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

fn checked_body_plan(
    source: &str,
    inputs: UnordMap<String, TensorType>,
    context: &RandomExecutionContext,
) -> EvaluationPlan {
    let parsed = chelis_surf::parser::parse_str(source).unwrap();
    let checked = chelis_types::check_ir_program(
        &chelis_surf::desugar::desugar_program(&parsed).expect("Surf fixture must desugar"),
    )
    .unwrap();
    // The ordinary host selector cuts at lexical handlers. This test lowers
    // the checked body itself; it does not claim full host/API transport.
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
    chelis_ir::lower::try_lower_subexpr_evaluation_plan(
        &children(sample)[1],
        inputs,
        UnordMap::new(),
        defs.clone(),
        context,
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
                    "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} i64)}} {seed}) \
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
                "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} i64)}} 42) (app {{}} (var {{}} dropout) (var {{}} x) (lit {{type: (t-prim {{}} {dtype})}} 0.1)))"
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
        let mut context = RandomExecutionContext::new(RandomLoweringState {
            seed: Some(7),
            counter: 13,
        });
        let plan = checked_body_plan(
            &source,
            [(
                "x".into(),
                TensorType {
                    dims: vec![DimInfo::Lit(32)],
                    precision: Prim::F32,
                },
            )]
            .into_iter()
            .collect(),
            &context,
        );
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

struct SpecialWords {
    prim: Prim,
    tag: &'static str,
    ctype: &'static str,
    input: [u64; 6],
    doubled: [u64; 6],
    one: u64,
}

fn special_words() -> [SpecialWords; 4] {
    // Independently encoded [-0, -3, +0, +inf, -inf, canonical NaN] and
    // division by the exact stored denominator 0.5. No evaluator oracle.
    [
        SpecialWords {
            prim: Prim::F16,
            tag: "CHELIS_DTYPE_F16",
            ctype: "uint16_t",
            input: [0x8000, 0xc200, 0, 0x7c00, 0xfc00, 0x7e00],
            doubled: [0x8000, 0xc600, 0, 0x7c00, 0xfc00, 0x7e00],
            one: 0x3c00,
        },
        SpecialWords {
            prim: Prim::Bf16,
            tag: "CHELIS_DTYPE_BF16",
            ctype: "uint16_t",
            input: [0x8000, 0xc040, 0, 0x7f80, 0xff80, 0x7fc0],
            doubled: [0x8000, 0xc0c0, 0, 0x7f80, 0xff80, 0x7fc0],
            one: 0x3f80,
        },
        SpecialWords {
            prim: Prim::F32,
            tag: "CHELIS_DTYPE_F32",
            ctype: "uint32_t",
            input: [
                0x80000000, 0xc0400000, 0, 0x7f800000, 0xff800000, 0x7fc00000,
            ],
            doubled: [
                0x80000000, 0xc0c00000, 0, 0x7f800000, 0xff800000, 0x7fc00000,
            ],
            one: 0x3f800000,
        },
        SpecialWords {
            prim: Prim::F64,
            tag: "CHELIS_DTYPE_F64",
            ctype: "uint64_t",
            input: [
                0x8000000000000000,
                0xc008000000000000,
                0,
                0x7ff0000000000000,
                0xfff0000000000000,
                0x7ff8000000000000,
            ],
            doubled: [
                0x8000000000000000,
                0xc018000000000000,
                0,
                0x7ff0000000000000,
                0xfff0000000000000,
                0x7ff8000000000000,
            ],
            one: 0x3ff0000000000000,
        },
    ]
}

fn special_word_driver(words: &SpecialWords, labels: &[String], expected: [[u64; 4]; 6]) -> String {
    let input: [[u64; 4]; 6] =
        std::array::from_fn(|offset| std::array::from_fn(|i| words.input[(offset + i) % 6]));
    native_word_driver(words, labels, &input, &expected)
}

fn native_word_driver(
    words: &SpecialWords,
    labels: &[String],
    input: &[[u64; 4]],
    expected: &[[u64; 4]],
) -> String {
    assert!(!input.is_empty());
    assert_eq!(input.len(), expected.len());
    let special = if labels.len() == 1 { "x" } else { "weights" };
    assert!(labels.iter().all(|name| name == "x" || name == "weights"));
    let special_slot = labels.iter().position(|name| name == special).unwrap();
    let encode = |values: &[u64]| {
        values
            .iter()
            .map(|v| format!("UINT64_C(0x{v:x})"))
            .collect::<Vec<_>>()
            .join(",")
    };
    let rows = |values: &[[u64; 4]]| {
        values
            .iter()
            .map(|row| format!("{{{}}}", encode(row)))
            .collect::<Vec<_>>()
            .join(",\n")
    };
    format!(
        r#"
int main(void) {{
    int64_t n = 4;
    const {ctype} input_cases[{cases}][4] = {{{input_cases}}};
    const {ctype} expected[{cases}][4] = {{{expected}}};
    for (int offset = 0; offset < {cases}; ++offset) {{
        chelis_tensor *inputs[{count}];
        {ctype} original[{count}][4];
        for (int slot = 0; slot < {count}; ++slot) {{
            chelis_scalar elements[4];
            for (int i = 0; i < 4; ++i) {{
                original[slot][i] = slot == {special_slot} ? input_cases[offset][i] : UINT64_C(0x{one:x});
                elements[i] = chelis_scalar_from_bits({tag}, original[slot][i]);
            }}
            inputs[slot] = chelis_alloc(1, &n, {tag});
            chelis_tensor_write *write = chelis_tensor_begin_write(inputs[slot]);
            chelis_tensor_write_literal(write, chelis_scalar_from_bits(CHELIS_DTYPE_I64, n), elements);
            chelis_tensor_end_write(write);
        }}
        for (int repeat = 0; repeat < 4; ++repeat) {{
            chelis_tensor *outputs[1];
            sample(inputs, {count}, outputs, 1);
            assert(chelis_tensor_rank(outputs[0]) == 1 && chelis_tensor_shape(outputs[0], 0) == n);
            chelis_read_view view = chelis_tensor_read_view(outputs[0]);
            assert(view.dtype == {tag} && view.count == n);
            assert(memcmp(view.data, expected[offset], sizeof(expected[offset])) == 0);
            chelis_tensor_release(outputs[0]);
            for (int slot = 0; slot < {count}; ++slot) {{
                assert(chelis_tensor_rank(inputs[slot]) == 1 && chelis_tensor_shape(inputs[slot], 0) == n);
                chelis_read_view original_view = chelis_tensor_read_view(inputs[slot]);
                assert(original_view.dtype == {tag} && original_view.count == n);
                assert(memcmp(original_view.data, original[slot], sizeof(original[slot])) == 0);
            }}
        }}
        for (int slot = 0; slot < {count}; ++slot) chelis_tensor_release(inputs[slot]);
    }}
    return 0;
}}
"#,
        ctype = words.ctype,
        cases = input.len(),
        input_cases = rows(input),
        expected = rows(expected),
        count = labels.len(),
        tag = words.tag,
        one = words.one
    )
}

fn corrupt_contiguous_dropout(
    source: &str,
    node: chelis_ir::dag::NodeId,
    corrupt: impl FnOnce(&str) -> String,
) -> String {
    let prefix = format!("__out_{}[i] = ", node.0);
    let lines = source
        .lines()
        .filter(|line| {
            line.trim_start().starts_with(&prefix) && line.contains("chelis_dropout_unit")
        })
        .collect::<Vec<_>>();
    assert_eq!(lines.len(), 1, "actual emitted dropout assignment");
    let replacement = corrupt(lines[0]);
    assert_ne!(
        replacement, lines[0],
        "mutation must alter the real assignment"
    );
    source.replacen(lines[0], &replacement, 1)
}

fn source_ad_plan(prim: Prim, rate: &str) -> (EvaluationPlan, chelis_ir::dag::NodeId) {
    use chelis_ir::evaluation::RandomSite;
    let dtype = prim.name();
    let source = format!(
        "def sample(x: tensor[4, {dtype}], weights: tensor[4, {dtype}]) -> tensor[4, {dtype}] = with seed(42i64) {{\n loss = fn (v: tensor[4, {dtype}]) -> tensor_to_scalar(sum(mul(dropout(v, {rate}{dtype}), weights), 0i32))\n grad(loss)(x)\n}}\n"
    );
    let inputs = ["x", "weights"]
        .into_iter()
        .map(|name| {
            (
                name.into(),
                TensorType {
                    dims: vec![DimInfo::Lit(4)],
                    precision: prim,
                },
            )
        })
        .collect();
    let plan = checked_body_plan(
        &source,
        inputs,
        &RandomExecutionContext::new(RandomLoweringState {
            seed: Some(7),
            counter: 13,
        }),
    );
    let replay = plan
        .clone()
        .verify_ownership()
        .unwrap()
        .with_emission(|owned, execution| {
            let sites = owned
                .emission()
                .nodes()
                .iter()
                .filter_map(|node| execution.site(node.id).map(|site| (node.id, site)))
                .collect::<Vec<_>>();
            match sites.as_slice() {
                [
                    (_, RandomSite::Forward { draw: forward, .. }),
                    (node, RandomSite::Replay { draw }),
                ] if draw == forward => *node,
                other => panic!("actual forward/replay pair: {other:?}"),
            }
        });
    (plan, replay)
}

#[test]
fn native_source_ad_replays_signed_and_nonfinite_stored_words() {
    for words in special_words() {
        let (plan, replay) = source_ad_plan(words.prim, "0.5");
        let artifact = chelis_backend_c::codegen_evaluation_with_options(
            plan.verify_ownership().unwrap(),
            "sample",
            Default::default(),
        )
        .unwrap();
        let mut labels = artifact.input_labels.clone();
        labels.sort();
        assert_eq!(labels, ["weights", "x"]);
        // Seed42 ordinal0 keeps coordinate1. §2.4 adds +0 before replay and
        // again at the gradient root, so a -0 source contribution becomes +0.
        let expected = std::array::from_fn(|offset| {
            let class = (offset + 1) % 6;
            [0, if class == 0 { 0 } else { words.doubled[class] }, 0, 0]
        });
        let driver = special_word_driver(&words, &artifact.input_labels, expected);
        ownership_support::balanced(&ownership_support::run(&artifact.c_source, &driver));
        let value = if matches!(words.prim, Prim::F16 | Prim::Bf16) {
            "__av".into()
        } else {
            format!("__in_a_{}[i]", replay.0)
        };
        let abs = if words.prim == Prim::F64 {
            "fabs"
        } else {
            "fabsf"
        };
        let mutant = corrupt_contiguous_dropout(&artifact.c_source, replay, |line| {
            line.replace(&format!("({value}) /"), &format!("{abs}({value}) /"))
        });
        assert_native_value_failure(&mutant, &driver);
    }
}

#[test]
fn native_special_words_reject_mask_sign_and_nan_corruption() {
    // [05-OP-37], [04-NUM-2/8]: the exact primal words include dropped +0,
    // kept -0, and canonical NaN; class-only comparison is insufficient.
    for words in special_words() {
        for rate in ["0.0", "0.5"] {
            let dtype = words.prim.name();
            let source = format!(
                "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} i64)}} 42) (app {{}} (var {{}} dropout) (var {{}} x) (lit {{type: (t-prim {{}} {dtype})}} {rate})))"
            );
            let plan = plan(&source, words.prim, 4);
            let node = plan.dag_for_inspection().roots()[0];
            let artifact = chelis_backend_c::codegen_evaluation_with_options(
                plan.verify_ownership().unwrap(),
                "sample",
                Default::default(),
            )
            .unwrap();
            assert_eq!(artifact.input_labels, ["x"]);
            let expected = std::array::from_fn(|offset| {
                std::array::from_fn(|i| {
                    if rate == "0.0" {
                        words.input[(offset + i) % 6]
                    } else if i == 1 {
                        words.doubled[(offset + i) % 6]
                    } else {
                        0
                    }
                })
            });
            let driver = special_word_driver(&words, &artifact.input_labels, expected);
            ownership_support::balanced(&ownership_support::run(&artifact.c_source, &driver));
            if rate == "0.5" {
                let value = if matches!(words.prim, Prim::F16 | Prim::Bf16) {
                    "__av".into()
                } else {
                    format!("__in_a_{}[i]", node.0)
                };
                let abs = if words.prim == Prim::F64 {
                    "fabs"
                } else {
                    "fabsf"
                };
                // Normalize finite zeros so only dropped nonfinite values
                // discriminate this multiplication mutant, not the -0 case.
                let masked = corrupt_contiguous_dropout(&artifact.c_source, node, |line| {
                    line.replace("? 0 :", &format!("? {abs}(0.0f * ({value})) :"))
                });
                assert_native_value_failure(&masked, &driver);
                let sign = corrupt_contiguous_dropout(&artifact.c_source, node, |line| {
                    line.replace(
                        &format!("({value}) /"),
                        &format!("(({value}) == 0 ? 0 : ({value})) /"),
                    )
                });
                assert_native_value_failure(&sign, &driver);
                let nan = corrupt_contiguous_dropout(&artifact.c_source, node, |line| {
                    format!(
                        "{line}\n{{ {ctype} bits; memcpy(&bits, &__out_{id}[i], sizeof(bits)); if (bits == UINT64_C(0x{canonical:x})) {{ bits ^= 1; memcpy(&__out_{id}[i], &bits, sizeof(bits)); }} }}",
                        ctype = words.ctype,
                        id = node.0,
                        canonical = words.input[5]
                    )
                });
                assert_native_value_failure(&nan, &driver);
            }
        }
    }
}

#[test]
fn native_source_ad_replay_finalizes_nonbinary_rate_division() {
    // [05-OP-37]/[04-NUM-8]/spec06 §2.4: finalized denominator then
    // division, not a precomputed reciprocal or a different-width divisor.
    // Independent rational-RNE witnesses: two positive cotangents and their
    // div results, followed by rate/denominator/reciprocal at arithmetic width.
    // The reciprocal itself has first been finalized to operand storage.
    let cases = [
        (
            [0x3c05, 0x3c06],
            [0x3c77, 0x3c79],
            0x3dccc000,
            0x3f666000,
            0x3f8e4000,
        ),
        (
            [0x3f81, 0x3f81],
            [0x3f90, 0x3f90],
            0x3dcd0000,
            0x3f660000,
            0x3f8e0000,
        ),
        (
            [0x3f800005, 0x3f800007],
            [0x3f8e38e9, 0x3f8e38ec],
            0x3dcccccd,
            0x3f666666,
            0x3f8e38e4,
        ),
        (
            [0x3ff0000000000005, 0x3ff0000000000000],
            [0x3ff1c71c71c71c77, 0x3ff1c71c71c71c72],
            0x3fb999999999999a,
            0x3feccccccccccccd,
            0x3ff1c71c71c71c72,
        ),
    ];
    for (words, (input, result, rate, denominator, reciprocal)) in
        special_words().into_iter().zip(cases)
    {
        let literal = |bits: u64| {
            if words.prim == Prim::F64 {
                format!("chelis_f64_from_bits(UINT64_C(0x{bits:016x}))")
            } else {
                format!("chelis_f32_from_bits(0x{bits:08x}u)")
            }
        };
        let signs = |values: [u64; 2]| {
            [
                values[0],
                values[0] | words.input[0],
                values[1],
                values[1] | words.input[0],
            ]
        };
        let input = signs(input);
        let result = signs(result);
        let input: [[u64; 4]; 4] =
            std::array::from_fn(|offset| std::array::from_fn(|i| input[(offset + i) % 4]));
        // Independently calculated seed42/ordinal0 rate .1 mask: K K K D.
        // These finite nonzero cotangents are unchanged by either +0 base.
        let expected: [[u64; 4]; 4] = std::array::from_fn(|offset| {
            std::array::from_fn(|i| if i == 3 { 0 } else { result[(offset + i) % 4] })
        });
        let (plan, replay) = source_ad_plan(words.prim, "0.1");
        let artifact = chelis_backend_c::codegen_evaluation_with_options(
            plan.verify_ownership().unwrap(),
            "sample",
            Default::default(),
        )
        .unwrap();
        let mut labels = artifact.input_labels.clone();
        labels.sort();
        assert_eq!(labels, ["weights", "x"]);
        let driver = native_word_driver(&words, &artifact.input_labels, &input, &expected);
        ownership_support::balanced(&ownership_support::run(&artifact.c_source, &driver));

        let denominator = literal(denominator);
        let reciprocal = literal(reciprocal);
        let multiplied = corrupt_contiguous_dropout(&artifact.c_source, replay, |line| {
            line.replace(&format!(" / {denominator}"), &format!(" * {reciprocal}"))
        });
        assert_native_value_failure(&multiplied, &driver);

        let wrong_divisor = match words.prim {
            // Omit the required f16/bf16 denominator storage finalization.
            Prim::F16 | Prim::Bf16 => format!("(1.0f - {})", literal(rate)),
            // Compute denominator and division at f64, then narrow at store.
            Prim::F32 => format!("(1.0 - (double){})", literal(rate)),
            // Incorrectly funnel the f64 denominator through f32.
            Prim::F64 => format!("((double)(float){denominator})"),
            _ => unreachable!(),
        };
        let divisor = corrupt_contiguous_dropout(&artifact.c_source, replay, |line| {
            line.replace(&format!(" / {denominator}"), &format!(" / {wrong_divisor}"))
        });
        assert_native_value_failure(&divisor, &driver);
    }
}

#[test]
fn native_mask_threshold_uses_arithmetic_width_and_strict_less_than() {
    // [05-RNG-1]/[05-OP-37]: compare the exact arithmetic-width unit,
    // neither its pre-width rational value nor a storage-width surrogate.
    // Inverting [05-RNG-1]'s bijective map gives ordinal0/index0 words
    // 7fffffffffffffff, 8000000000000000, 8000000000000800, 7fff000000000000.
    // Units are .5-2^-53, .5, .5+2^-53, .5-2^-16 respectively. The first
    // and third round to .5 at f32; only the last storage-rounds to .5 at
    // f16/bf16 while staying strictly below at their f32 arithmetic width.
    let cases = [
        (
            7396636047707789066_i64,
            [true, false, true, true],
            [false, false, true, true],
        ),
        (
            4901139120565445618_i64,
            [true, false, true, false],
            [true, false, true, false],
        ),
        (
            6117835775437243522_i64,
            [true, false, false, false],
            [true, false, false, false],
        ),
        (3386422020048024308_i64, [false; 4], [false; 4]),
    ];
    for (words, two) in
        special_words()
            .into_iter()
            .zip([0x4000, 0x4000, 0x40000000, 0x4000000000000000])
    {
        for (case, (seed, narrow_mask, wide_mask)) in cases.into_iter().enumerate() {
            let dtype = words.prim.name();
            let source = format!(
                "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} i64)}} {seed}) (app {{}} (var {{}} dropout) (var {{}} x) (lit {{type: (t-prim {{}} {dtype})}} 0.5)))"
            );
            let plan = plan(&source, words.prim, 4);
            let node = plan.dag_for_inspection().roots()[0];
            let artifact = chelis_backend_c::codegen_evaluation_with_options(
                plan.verify_ownership().unwrap(),
                "sample",
                Default::default(),
            )
            .unwrap();
            assert_eq!(artifact.input_labels, ["x"]);
            let mask = if words.prim == Prim::F64 {
                wide_mask
            } else {
                narrow_mask
            };
            let input = [[words.one, words.input[1], words.one, words.input[1]]];
            let expected = [std::array::from_fn(|i| {
                if mask[i] {
                    if i % 2 == 0 { two } else { words.doubled[1] }
                } else {
                    0
                }
            })];
            let driver = native_word_driver(&words, &artifact.input_labels, &input, &expected);
            ownership_support::balanced(&ownership_support::run(&artifact.c_source, &driver));
            if case == 1 {
                let inclusive = corrupt_contiguous_dropout(&artifact.c_source, node, |line| {
                    line.replace(" < ", " <= ")
                });
                assert_native_value_failure(&inclusive, &driver);
            }
            if case == 0 {
                let (from, to) = if words.prim == Prim::F64 {
                    ("chelis_dropout_unit(", "chelis_dropout_unit_f32(")
                } else {
                    ("chelis_dropout_unit_f32(", "chelis_dropout_unit(")
                };
                let width = corrupt_contiguous_dropout(&artifact.c_source, node, |line| {
                    line.replace(from, to)
                });
                assert_native_value_failure(&width, &driver);
            }
            if case == 3 && matches!(words.prim, Prim::F16 | Prim::Bf16) {
                let unit =
                    "chelis_dropout_unit_f32(__chelis_draw_seed_0, __chelis_draw_ordinal_0, i)";
                let rounded = format!("chelis_{dtype}_to_f32(chelis_f32_to_{dtype}({unit}))");
                let storage = corrupt_contiguous_dropout(&artifact.c_source, node, |line| {
                    line.replace(unit, &rounded)
                });
                assert_native_value_failure(&storage, &driver);
            }
        }
    }
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
            "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} i64)}} 42) (app {{}} (var {{}} dropout) (var {{}} x) (lit {{type: (t-prim {{}} f32)}} {rate})))"
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
                "(handle-effect {{effect: random}} (lit {{type: (t-prim {{}} i64)}} 42) \
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
