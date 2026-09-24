//! API source-shell acceptance for the bounded fixed-control dropout plan.
#![allow(deprecated)] // Explicit compatibility/parity coverage for prepare_eval.
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use chelis_compiler_api::compiler::{eval_selected, prepare_eval};
use chelis_compiler_api::schema::{
    EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use std::collections::BTreeMap;

#[test]
fn concrete_static_rate_calls_keep_source_bindings_across_host_boundaries() {
    // [05-OP-37]/[05-RNG-1], #1764: source-static actuals are not runtime rates.
    for definition in [
        "def keep(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(x, rate)\ndef draw(x: tensor[4, f32]) -> tensor[4, f32] = keep(x, 0.5f32)",
        "def draw(x: tensor[4, f32]) -> tensor[4, f32] = { rate = 0.5f32\n alias = rate\n dropout(x, alias) }",
        "def draw(x: tensor[4, f32]) -> tensor[4, f32] = { keep = fn (v: tensor[4, f32], rate: f32) -> dropout(v, rate)\n keep(x, 0.5f32) }",
        "def keep(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(x, rate)\ndef draw(x: tensor[4, f32]) -> tensor[4, f32] = { keep = fn (v: tensor[4, f32], rate: f32) -> dropout(v, 0.5f32)\n keep(x, 0.0f32) }",
    ] {
        let source = format!(
            "{definition}\ndef loss(x: tensor[4, f32]) -> tensor[f32] = sum(draw(x), 0i32)\ndef main() = with seed(42i64) {{\n x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n first = draw(copy(x))\n backward = grad(loss)(copy(x))\n next = dropout(x, 0.5f32)\n (first, backward, next, x)\n}}\n"
        );
        let result = eval_selected(request(&source), &["main".into()])
            .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
        for (index, ordinal) in [0, 1, 2].into_iter().enumerate() {
            assert_eq!(
                tensor(&result, &format!("main.{index}")),
                mask(ordinal)[..4]
            );
        }
        assert_eq!(tensor(&result, "main.3"), vec![1.0; 4]);
    }
}

#[test]
fn concrete_static_rate_exported_library_call_survives_context_decode() {
    use chelis_compiler_api::compiler::{eval_in_context, prepare_eval_in_context};
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"static_rate\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
    std::fs::write(directory.path().join("src/draw.ch"), "module Probe.Draw\nexport (keep)\ndef keep(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(x, rate)\n").unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let decoded =
        chelis_compiler_api::context::CompiledContext::decode(&context.encode().unwrap()).unwrap();
    let source = "module Probe.Client\nimport Probe.Draw (keep)\ndef main() = with seed(42i64) {\n x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n first = keep(copy(x), 0.0f32)\n second = keep(copy(x), 0.5f32)\n (first, second, dropout(x, 0.5f32))\n}\n";
    for context in [&context, &decoded] {
        let result = eval_in_context(context, source).unwrap();
        assert_eq!(tensor(&result, "main.0"), vec![1.0; 4]);
        assert_eq!(tensor(&result, "main.1"), mask(1)[..4]);
        assert_eq!(tensor(&result, "main.2"), mask(2)[..4]);
        let prepared = prepare_eval_in_context(context, source).unwrap();
        for _ in 0..2 {
            let result = prepared.eval_root(BTreeMap::new(), "main").unwrap();
            assert_eq!(tensor(&result, "main.1"), mask(1)[..4]);
            assert_eq!(tensor(&result, "main.2"), mask(2)[..4]);
        }
    }
}

#[test]
fn compiled_static_rate_exported_library_call_survives_context_decode() {
    use chelis_compiler_api::compiler::compile_for_execution_in_context;
    use chelis_compiler_api::schema::CompileTarget;
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};

    // [05-OP-37]/[05-RNG-1]: library source must retain the selected entry's
    // fixed-control execution authority across the serialized context boundary.
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"static_rate\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
    std::fs::write(directory.path().join("src/draw.ch"), "module Probe.Draw\nexport (keep)\ndef keep(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(x, rate)\n").unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let decoded =
        chelis_compiler_api::context::CompiledContext::decode(&context.encode().unwrap()).unwrap();
    let source = "module Probe.Client\nimport Probe.Draw (keep)\ndef main(x: tensor[4, f32]) -> tensor[4, f32] = with seed(42i64) { keep(x, 0.5f32) }\n";
    for context in [&decoded, &context] {
        let artifact =
            compile_for_execution_in_context(context, source, CompileTarget::C, Some("main"))
                .expect("fixed-control library entry must compile through the context API");
        assert_eq!(artifact.inputs.len(), 1);
        assert_eq!(artifact.outputs.len(), 1);
    }
}

// chelis#2411: a runtime rate is an ordinary [05-OP-37] operand. The call
// used to be refused; it now draws ordinal 0 of the handler, which the
// independent reference below recomputes from the spec text.
#[test]
fn concrete_runtime_rate_actual_draws_the_conforming_stream() {
    let source = "def keep(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(x, rate)\ndef main() = with seed(42i64) { keep(to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]), tensor_to_scalar(scalar_to_tensor(0.5f32))) }\n";
    let result = eval_selected(request(source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    assert_eq!(tensor(&result, "main"), mask(0)[..4]);
}

#[test]
fn unrelated_scalar_capture_preserves_public_acceptance_and_next_draw() {
    // PR1807 R1: an unrelated rebound scalar is not a closure dependency.
    let ones = "to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])";
    for function_body in [
        "dropout(v, 0.5f32)",
        "{ ignored = unrelated\n dropout(v, 0.5f32) }",
    ] {
        for shadow in [false, true] {
            for next_draw in [false, true] {
                let source = format!(
                    "def sample(x: tensor[8, f32]) -> tensor[8, f32] = {{\n\
                 unrelated = 0.5f32\n\
                 f = fn (v: tensor[8, f32]) -> {function_body}\n\
                 {}\n f(x)\n}}\n\
                 def main() = with seed(42i64) {{\n first = sample({ones})\n {}\n}}",
                    if shadow { "unrelated = 0.25f32" } else { "" },
                    if next_draw {
                        format!("add(first, dropout({ones}, 0.5f32))")
                    } else {
                        "first".into()
                    },
                );
                let result = eval_selected(
                    EvalRequest {
                        source_kind: SourceKind::Surf,
                        source,
                        bindings: BTreeMap::new(),
                    },
                    &["main".into()],
                )
                .unwrap_or_else(|error| panic!("shadow={shadow}, next={next_draw}: {error:?}"));
                let expected = if next_draw {
                    mask(0)
                        .iter()
                        .zip(mask(1))
                        .take(8)
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>()
                } else {
                    mask(0)[..8].to_vec()
                };
                assert_eq!(tensor(&result, "main"), expected);
                assert!(result.transcript.is_empty());
            }
        }
    }
}

#[test]
fn generic_static_rate_cast_reaches_the_evaluator_plan() {
    // [05-OP-37]/[05-RNG-1], chelis#1764: specialize the source call while
    // its checked actual dtype and fixed control expression are still paired.
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let one = if matches!(dtype, "f32" | "f64") {
            format!("1.0{dtype}")
        } else {
            format!("cast(1.0, {dtype})")
        };
        let source = format!(
            "def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))\n\
             def main() = with seed(42i64) {{ keep(to_tensor([{one}, {one}, {one}, {one}])) }}"
        );
        let result = eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source,
                bindings: BTreeMap::new(),
            },
            &["main".into()],
        )
        .unwrap();
        assert!(result.transcript.is_empty());
        assert_eq!(tensor(&result, "main"), mask(0)[..4]);
        assert_tensor_dtype_shape(&result, "main", dtype);
    }
}

fn assert_tensor_dtype_shape(result: &EvalResult, name: &str, dtype: &str) {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap();
    let ExecutionValue::Tensor { value } = &root.value else {
        panic!("{root:?}")
    };
    assert_eq!(value.shape, [4]);
    assert_eq!(value.data.prim().name(), dtype);
}

#[test]
fn generic_static_rate_gradient_and_following_draw_share_the_handled_stream() {
    // [05-OP-37]/[05-RNG-1]: source AD replays ordinal1 without consuming
    // ordinal2. Expected complete masks come from the independent spec map.
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            "def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))\n\
             def loss(x: tensor[4, {dtype}]) -> {dtype} = tensor_to_scalar(sum(keep(x), 0i32))\n\
             def main() = with seed(42i64) {{\n\
               x = to_tensor([cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype})])\n\
               first = keep(copy(x))\n backward = grad(loss)(copy(x))\n next = keep(copy(x))\n\
               (first, backward, next, x)\n }}"
        );
        let prepared = prepare_eval(request(&source)).unwrap();
        for result in [
            eval_selected(request(&source), &["main".into()]),
            prepared.eval_root(BTreeMap::new(), "main"),
            prepared.eval_root(BTreeMap::new(), "main"),
        ] {
            let result = result.unwrap_or_else(|error| panic!("{dtype}: {error:?}"));
            for (index, ordinal) in [0, 1, 2].into_iter().enumerate() {
                let name = format!("main.{index}");
                assert_eq!(tensor(&result, &name), mask(ordinal)[..4]);
                assert_tensor_dtype_shape(&result, &name, dtype);
            }
            assert_eq!(tensor(&result, "main.3"), [1.0; 4]);
            assert_tensor_dtype_shape(&result, "main.3", dtype);
            assert!(result.transcript.is_empty());
        }
    }
}

#[test]
fn generic_static_rate_calls_do_not_reuse_another_calls_precision() {
    let source = "def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))\n\
        def main() = with seed(42i64) {\n\
          x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
          y = to_tensor([1.0f64, 1.0f64, 1.0f64, 1.0f64])\n\
          a = keep(copy(x))\n b = keep(y)\n c = keep(copy(x))\n\
          next = dropout(x, 0.5f32)\n\
          (a, b, c, next)\n }";
    let result = eval_selected(request(source), &["main".into()]).unwrap();
    for (index, ordinal, dtype) in [(0, 0, "f32"), (1, 1, "f64"), (2, 2, "f32"), (3, 3, "f32")] {
        let name = format!("main.{index}");
        assert_eq!(tensor(&result, &name), mask(ordinal)[..4]);
        assert_tensor_dtype_shape(&result, &name, dtype);
    }
}

#[test]
fn a_local_callable_still_shadows_the_generic_dropout_definition() {
    let source = "def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))\n\
        def main() = with seed(42i64) {\n\
          keep = fn (v: tensor[4, f32]) -> v\n\
          x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
          unchanged = keep(copy(x))\n next = dropout(x, 0.5f32)\n\
          (unchanged, next)\n }";
    let result = eval_selected(request(source), &["main".into()]).unwrap();
    assert_eq!(tensor(&result, "main.0"), [1.0; 4]);
    assert_eq!(tensor(&result, "main.1"), mask(0)[..4]);
}

// chelis#2411: a generic runtime rate draws the conforming stream at its own
// dtype's arithmetic width; it was refused before.
#[test]
fn generic_runtime_rate_draws_the_conforming_stream() {
    for dtype in ["f32", "f64"] {
        for definition in [
            "def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(tensor_to_scalar(scalar_to_tensor(0.5f32)), p))".to_string(),
            format!("def keep[p: Float](x: tensor[4, p], rate: p) -> tensor[4, p] = dropout(x, rate)\ndef run(x: tensor[4, {dtype}]) -> tensor[4, {dtype}] = keep(x, tensor_to_scalar(scalar_to_tensor(0.5{dtype})))"),
            format!("def keep[p: Float](x: tensor[4, p], rate: p) -> tensor[4, p] = dropout(x, rate)\ndef run(x: tensor[4, {dtype}]) -> tensor[4, {dtype}] = {{ rate = tensor_to_scalar(scalar_to_tensor(0.5{dtype}))\n keep(x, rate) }}"),
        ] {
            let callee = if definition.contains("def run") { "run" } else { "keep" };
            let source = format!("{definition}\ndef main() = with seed(42i64) {{ {callee}(to_tensor([1.0{dtype}, 1.0{dtype}, 1.0{dtype}, 1.0{dtype}])) }}");
            let result = eval_selected(request(&source), &["main".into()])
                .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
            assert_eq!(
                tensor(&result, "main"),
                mask_at_width(42, 0, dtype == "f64")[..4],
                "{source}"
            );
        }
    }
}

#[test]
fn generic_static_rate_effecting_argument_is_evaluated_once() {
    for dtype in ["f32", "f64"] {
        let source = format!(
            "def keep[p: Float](x: tensor[4, p]) -> tensor[4, p] = dropout(x, cast(0.5, p))\n\
            def main() = with seed(42i64) {{\n\
              x = to_tensor([1.0{dtype}, 1.0{dtype}, 1.0{dtype}, 1.0{dtype}])\n\
              first = keep(dropout(x, 0.0{dtype}))\n next = keep(x)\n\
              (first, next)\n }}"
        );
        let result = eval_selected(request(&source), &["main".into()]).unwrap();
        assert_eq!(tensor(&result, "main.0"), mask(1)[..4]);
        assert_eq!(tensor(&result, "main.1"), mask(2)[..4]);
    }
}

#[test]
fn static_rate_example_includes_generic_evaluator_dispatch() {
    let source = include_str!("../../../examples/dropout_static_rate.ch");
    let result = eval_selected(request(source), &["result".into()]).unwrap();
    for ordinal in 0..3 {
        assert_eq!(
            tensor(&result, &format!("result.{ordinal}")),
            mask(ordinal)[..4]
        );
    }
    assert_eq!(tensor(&result, "result.3"), [1.0; 4]);
}

fn generic_scalar_data_argument_source(copy_middle: bool) -> String {
    let middle_input = if copy_middle { "copy(x)" } else { "x" };
    format!(
        "def keep[p: Float](x: tensor[4, p], extra: p) -> tensor[4, p] = mul(dropout(x, cast(0.5, p)), insert(scalar_to_tensor(extra), 0i32, 4i64))\n\
         def main() = with seed(42i64) {{\n\
           x = to_tensor([1.0f64, 1.0f64, 1.0f64, 1.0f64])\n\
           y = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
           first = keep(copy(y), 1.0f32)\n\
           middle = keep({middle_input}, tensor_to_scalar(dropout(scalar_to_tensor(1.0f64), 0.0f64)))\n\
           next = keep(y, 1.0f32)\n (first, middle, next)\n }}"
    )
}

fn check_generic_scalar_data_argument_stream(copy_middle: bool) {
    let source = generic_scalar_data_argument_source(copy_middle);
    let prepared = prepare_eval(request(&source)).unwrap();
    // Independently encoded [05-RNG-1] seed42 ordinals0/2/3. The rate-zero
    // scalar actual consumes ordinal1 once, before the middle helper call.
    // Full stored words make every coordinate, dtype and zero sign observable.
    let expected = serde_json::json!([
        {"shape":[4], "data":{"dtype":"f32", "bits":["00000000","40000000","00000000","00000000"]}},
        {"shape":[4], "data":{"dtype":"f64", "bits":["4000000000000000","4000000000000000","0000000000000000","0000000000000000"]}},
        {"shape":[4], "data":{"dtype":"f32", "bits":["00000000","40000000","00000000","00000000"]}}
    ]);
    // Run all three entry invocations before asserting, so a red regression
    // records both immediate and reused-preparation behavior.
    let actual: Vec<_> = [
        eval_selected(request(&source), &["main".into()]),
        prepared.eval_root(BTreeMap::new(), "main"),
        prepared.eval_root(BTreeMap::new(), "main"),
    ]
    .into_iter()
    .map(|result| {
        let result = result.unwrap();
        assert!(result.transcript.is_empty());
        (0..3)
            .map(|index| {
                let name = format!("main.{index}");
                let root = result
                    .roots
                    .iter()
                    .find(|root| root.name.as_deref() == Some(&name))
                    .unwrap();
                let ExecutionValue::Tensor { value } = &root.value else {
                    panic!("{root:?}")
                };
                serde_json::json!({"shape":value.shape, "data":value.data})
            })
            .collect::<Vec<_>>()
    })
    .collect();
    assert_eq!(
        serde_json::json!(actual),
        serde_json::json!([expected, expected, expected])
    );
}

#[test]
fn generic_scalar_data_argument_consumes_its_draw_once() {
    check_generic_scalar_data_argument_stream(false);
}

#[test]
fn generic_scalar_data_argument_copy_control_keeps_the_same_stream() {
    check_generic_scalar_data_argument_stream(true);
}

#[test]
fn fixed_dropout_composes_with_host_produced_checked_reshape_targets() {
    use chelis_compiler_api::compiler::{eval_in_context, prepare_eval_in_context};
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
    let mut failures = Vec::new();
    for (target, source_draws) in [
        ("numel(source)", 0),
        ("len(to_list(source))", 0),
        ("bitand(shape(source, 0i32), 3i64)", 0),
        ("numel(dropout(source, 0.0f32))", 1),
    ] {
        for dropout in [false, true] {
            let body = if dropout {
                format!(
                    "{{\n dead = dropout(x, 0.0f32)\n _ = drop(dead)\n dropout(reshape(x, [{target}, 2i64]), 0.5f32)\n}}"
                )
            } else {
                format!("reshape(x, [{target}, 2i64])")
            };
            let definition = format!(
                "def loss[m, n](source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {body}\n"
            );
            let directory = tempfile::tempdir().unwrap();
            std::fs::create_dir(directory.path().join("src")).unwrap();
            std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"review_three_guard\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
            std::fs::write(
                directory.path().join("src/draw.ch"),
                format!("module Probe.Draw\nexport (loss)\n{definition}"),
            )
            .unwrap();
            let context = compile_reef_context(directory.path(), directory.path()).unwrap();
            let bytes = context.encode().unwrap();
            let decoded = chelis_compiler_api::context::CompiledContext::decode(&bytes).unwrap();
            for count in [2, 3] {
                let source_values = std::iter::repeat_n("1.0f32", count)
                    .collect::<Vec<_>>()
                    .join(", ");
                let x_values = std::iter::repeat_n("1.0f32", count * 2)
                    .collect::<Vec<_>>()
                    .join(", ");
                let main = format!(
                    "def main() = with seed(42i64) {{ loss(to_tensor([{source_values}]), to_tensor([{x_values}])) }}"
                );
                let source = format!("{definition}\n{main}");
                let request = EvalRequest {
                    source_kind: SourceKind::Surf,
                    source: source.clone(),
                    bindings: BTreeMap::new(),
                };
                let prepared = prepare_eval(request.clone()).unwrap();
                let mut results = vec![
                    ("ordinary", eval_selected(request, &["main".into()])),
                    ("prepared", prepared.eval_root(BTreeMap::new(), "main")),
                ];
                let client = format!("module Probe.Eval\nimport Probe.Draw (loss)\n{main}");
                for (label, context) in [("context", &context), ("decoded", &decoded)] {
                    results.push((label, eval_in_context(context, &client)));
                    let prepared = prepare_eval_in_context(context, &client).unwrap();
                    results.push((label, prepared.eval_root(BTreeMap::new(), "main")));
                }
                for (lane, result) in results {
                    if count == 2 {
                        let result = result.unwrap_or_else(|error| {
                            panic!("{target} dropout={dropout} {lane}: {error:?}")
                        });
                        let ExecutionValue::Tensor { value } = &result.roots[0].value else {
                            panic!("{result:?}")
                        };
                        assert_eq!(value.shape, vec![2, 2]);
                        assert_eq!(
                            value.data.to_f64_lossy_vec(),
                            if dropout {
                                mask(1 + source_draws)[..4].to_vec()
                            } else {
                                vec![1.0; 4]
                            }
                        );
                    } else {
                        match result {
                            Err(error)
                                if error.errors.iter().any(|error| {
                                    error.message.contains("claimed = 2")
                                        && error.message.contains("reshape axis 0 = 3")
                                        && error
                                            .message
                                            .contains("numeric trap: domain in reshape at i64")
                                }) =>
                            {
                                eprintln!("PASS negative {target} dropout={dropout} {lane}");
                            }
                            other => failures
                                .push(format!("{target} dropout={dropout} {lane}: {other:?}")),
                        }
                    }
                }
                assert_eq!(context.encode().unwrap(), bytes);
                assert_eq!(decoded.encode().unwrap(), bytes);
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn host_only_random_source_does_not_cache_the_first_callers_seed() {
    use chelis_compiler_api::compiler::{eval_in_context, prepare_eval_in_context};
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};

    fn ones(count: usize) -> String {
        format!(
            "to_tensor([{}])",
            std::iter::repeat_n("1.0f32", count)
                .collect::<Vec<_>>()
                .join(", ")
        )
    }

    fn check(result: EvalResult, first: u64, second: u64, lane: &str) {
        for (name, expected) in [
            ("main.0.0", vec![1.0; 4]),
            ("main.0.1", mask_with_seed(first, 1)[..4].to_vec()),
            ("main.1.0", vec![1.0; 4]),
            ("main.1.1", mask_with_seed(second, 1)[..4].to_vec()),
        ] {
            assert_eq!(
                tensor(&result, name),
                expected,
                "{lane} seeds={first},{second} root={name}"
            );
        }
    }

    let definition = "def draw[m, n](source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = reshape(x, [numel(dropout(source, 0.0f32)), 2i64])";
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("reef.toml"),
        format!(
            "[package]\nname = \"round_four_seed\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
        ),
    )
    .unwrap();
    std::fs::write(
        directory.path().join("src/draw.ch"),
        format!("module Probe.Draw\nexport (draw)\n{definition}\n"),
    )
    .unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let bytes = context.encode().unwrap();
    let decoded = chelis_compiler_api::context::CompiledContext::decode(&bytes).unwrap();

    for (first, second) in [(42, 42), (42, 7), (7, 42)] {
        let main = format!(
            "def main() = {{\n source = {}\n x = {}\n a = with seed({first}i64) {{\n shaped = draw(copy(source), copy(x))\n (shaped, dropout(copy(x), 0.5f32))\n }}\n b = with seed({second}i64) {{\n shaped = draw(source, copy(x))\n (shaped, dropout(x, 0.5f32))\n }}\n (a, b)\n}}",
            ones(2),
            ones(4)
        );
        let source = format!("{definition}\n{main}");
        check(
            eval_selected(request(&source), &["main".into()]).unwrap(),
            first,
            second,
            "ordinary",
        );
        let prepared = prepare_eval(request(&source)).unwrap();
        for repetition in 0..2 {
            check(
                prepared.eval_root(BTreeMap::new(), "main").unwrap(),
                first,
                second,
                &format!("prepared-{repetition}"),
            );
        }

        let client = format!("module Probe.Eval\nimport Probe.Draw (draw)\n{main}");
        for (label, checked_context) in [("context", &context), ("decoded", &decoded)] {
            check(
                eval_in_context(checked_context, &client).unwrap(),
                first,
                second,
                label,
            );
            let prepared = prepare_eval_in_context(checked_context, &client).unwrap();
            for repetition in 0..2 {
                check(
                    prepared.eval_root(BTreeMap::new(), "main").unwrap(),
                    first,
                    second,
                    &format!("{label}-prepared-{repetition}"),
                );
            }
            assert_eq!(checked_context.encode().unwrap(), bytes);
        }
    }
    assert_eq!(context.encode().unwrap(), bytes);
    assert_eq!(decoded.encode().unwrap(), bytes);
}

fn bindings() -> BTreeMap<String, TensorValue> {
    BTreeMap::from([(
        "x".into(),
        TensorValue {
            shape: vec![32],
            data: wire_values::storage_f32(vec![1.0; 32]),
        },
    )])
}

#[test]
fn staged_host_sources_interleave_input_ad_and_the_next_draw() {
    let source = "def loss[a, b](x: tensor[a, b, f32]) -> tensor[f32] = sum(sum(dropout(x, 0.5f32), 0i32), 0i32)\ndef checked[m, n](source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {\n first = dropout(source, 0.0f32)\n shaped = reshape(x, [numel(first), 2i64])\n gradient = grad(loss)(shaped)\n dropout(reshape(gradient, [len(to_list(source)), 2i64]), 0.5f32)\n}\ndef sample[m, n](source: tensor[m, f32], x: tensor[n, f32]) = with seed(42i64) {\n result = checked(source, copy(x))\n (result, dropout(x, 0.5f32))\n}";
    let prepared = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: BTreeMap::new(),
    })
    .unwrap();
    for count in [2, 3, 2] {
        let bindings = BTreeMap::from([
            (
                "source".into(),
                TensorValue {
                    shape: vec![count],
                    data: wire_values::storage_f32(vec![1.0; count as usize]),
                },
            ),
            (
                "x".into(),
                TensorValue {
                    shape: vec![count * 2],
                    data: wire_values::storage_f32(vec![1.0; count as usize * 2]),
                },
            ),
        ]);
        for result in [
            eval_selected(
                EvalRequest {
                    source_kind: SourceKind::Surf,
                    source: source.into(),
                    bindings: bindings.clone(),
                },
                &["sample".into()],
            ),
            prepared.eval_root(bindings, "sample"),
        ] {
            if count == 3 {
                let error = result.unwrap_err();
                assert!(
                    error
                        .errors
                        .iter()
                        .any(|error| error.message.contains("claimed = 2")
                            && error.message.contains("reshape axis 0 = 3")),
                    "{error:?}"
                );
            } else {
                let result = result.unwrap();
                assert_eq!(
                    tensor(&result, "sample.0"),
                    mask(1)
                        .iter()
                        .zip(mask(2))
                        .take(4)
                        .map(|(a, b)| a * b)
                        .collect::<Vec<_>>()
                );
                assert_eq!(tensor(&result, "sample.1"), mask(3)[..4]);
            }
        }
    }
}

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: bindings(),
    }
}

#[test]
fn staged_ad_local_seed_controls_restore_before_the_following_host_cut() {
    let source = "def loss[a, b](x: tensor[a, b, f32]) -> tensor[f32] = with seed(42i64) {\n identity = with seed(42i64) { x }\n sum(sum(dropout(identity, 0.5f32), 0i32), 0i32)\n}\ndef checked[m, n](source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {\n first = dropout(source, 0.0f32)\n shaped = reshape(x, [numel(first), 2i64])\n gradient = grad(loss)(shaped)\n dropout(reshape(gradient, [len(to_list(source)), 2i64]), 0.5f32)\n}\ndef sample() = with seed(42i64) {\n source = to_tensor(SOURCE)\n x = to_tensor(VALUES)\n result = checked(source, copy(x))\n (result, dropout(x, 0.5f32))\n}";
    for count in [2, 3, 2] {
        let values = |n| {
            format!(
                "[{}]",
                std::iter::repeat_n("1.0f32", n)
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        let source = source
            .replace("SOURCE", &values(count))
            .replace("VALUES", &values(count * 2));
        let request = EvalRequest {
            source_kind: SourceKind::Surf,
            source,
            bindings: BTreeMap::new(),
        };
        let prepared = prepare_eval(request.clone()).unwrap();
        for result in [
            eval_selected(request, &["sample".into()]),
            prepared.eval_root(BTreeMap::new(), "sample"),
        ] {
            if count == 3 {
                let error = result.unwrap_err();
                assert!(
                    error
                        .errors
                        .iter()
                        .any(|error| error.message.contains("claimed = 2")
                            && error.message.contains("reshape axis 0 = 3")),
                    "{error:?}"
                );
            } else {
                let result = result.unwrap();
                assert_eq!(
                    tensor(&result, "sample.0"),
                    mask(0)
                        .iter()
                        .zip(mask(1))
                        .take(4)
                        .map(|(a, b)| a * b)
                        .collect::<Vec<_>>()
                );
                assert_eq!(tensor(&result, "sample.1"), mask(2)[..4]);
            }
        }
    }
}

fn tensor(result: &EvalResult, name: &str) -> Vec<f64> {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap();
    let ExecutionValue::Tensor { value } = &root.value else {
        panic!("{root:?}")
    };
    (0..value.data.len())
        .map(|index| value.data.element_f64_lossy(index))
        .collect()
}

// [04] section 4.7 claims must survive the [05-OP-37] source plan,
// including retained dead draws and first-order replay.
#[test]
fn checked_extent_dropout_source_and_gradient_keep_computed_claims() {
    let helper = "def checked[n](x: tensor[n, f32]) -> tensor[16, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])\n";
    let loss = "def loss[n](x: tensor[n, f32]) -> tensor[f32] = sum(sum(dropout(checked(x), 0.5f32), 0), 0)\n";
    for gradient in [false, true] {
        let body = if gradient {
            "grad(loss)(x)"
        } else {
            "dropout(checked(x), 0.5f32)"
        };
        let source = format!(
            "{helper}{loss}def sample[n](x: tensor[n, f32]) = with seed(42i64) {{\n dead = dropout(x, 0.0f32)\n _ = drop(dead)\n {body}\n}}\n"
        );
        let prepared = prepare_eval(request(&source)).unwrap();
        for count in [32, 34, 32] {
            let inputs = BTreeMap::from([(
                "x".into(),
                TensorValue {
                    shape: vec![count],
                    data: wire_values::storage_f32(vec![1.0; count as usize]),
                },
            )]);
            for result in [
                eval_selected(
                    EvalRequest {
                        source_kind: SourceKind::Surf,
                        source: source.clone(),
                        bindings: inputs.clone(),
                    },
                    &["sample".into()],
                ),
                prepared.eval_root(inputs, "sample"),
            ] {
                if count == 32 {
                    assert_eq!(tensor(&result.unwrap(), "sample"), mask(1));
                } else {
                    let error = result.unwrap_err();
                    assert!(
                        error
                            .errors
                            .iter()
                            .any(|error| error.message.contains("claimed = 16")
                                && error.message.contains("reshape axis 0 = 17")
                                && error
                                    .message
                                    .contains("numeric trap: domain in reshape at i64")),
                        "gradient={gradient}: {error:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn checked_unit_axis_dropout_source_and_gradient_preserve_domain_checks() {
    for gradient in [false, true] {
        let result = if gradient { "grad(loss)(b)" } else { "draw(b)" };
        let source = format!(
            "def draw(b: tensor[unit, f32]) -> tensor[32, f32] = dropout(expand(b, 0i32, 32i64), 0.5f32)\ndef loss(b: tensor[unit, f32]) -> tensor[f32] = sum(draw(b), 0)\ndef sample(b: tensor[unit, f32]) = with seed(42i64) {{\n dead = dropout(b, 0.0f32)\n _ = drop(dead)\n {result}\n}}\n"
        );
        let prepared = prepare_eval(request(&source)).unwrap();
        for count in [1, 2, 1] {
            let inputs = BTreeMap::from([(
                "b".into(),
                TensorValue {
                    shape: vec![count],
                    data: wire_values::storage_f32(vec![1.0; count as usize]),
                },
            )]);
            for result in [
                eval_selected(
                    EvalRequest {
                        source_kind: SourceKind::Surf,
                        source: source.clone(),
                        bindings: inputs.clone(),
                    },
                    &["sample".into()],
                ),
                prepared.eval_root(inputs, "sample"),
            ] {
                if count == 1 {
                    let expected = if gradient {
                        vec![mask(1).iter().sum()]
                    } else {
                        mask(1)
                    };
                    assert_eq!(tensor(&result.unwrap(), "sample"), expected);
                } else {
                    let error = result.unwrap_err();
                    assert!(
                        error
                            .errors
                            .iter()
                            .any(|error| error.message.contains("claimed = 1")
                                && error.message.contains("axis 0 = 2")
                                && error
                                    .message
                                    .contains("numeric trap: domain in load at i64")),
                        "gradient={gradient}: {error:?}"
                    );
                }
            }
        }
    }
}

#[test]
fn checked_extent_dropout_context_cache_keeps_claims_and_fresh_replay() {
    use chelis_compiler_api::compiler::{eval_in_context, prepare_eval_in_context};
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!(
        "[package]\nname = \"extent_dropout_probe\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
    )).unwrap();
    std::fs::write(directory.path().join("src/draw.ch"),
        "module Probe.Draw\nexport (draw, loss)\ndef checked[n](x: tensor[n, f32]) -> tensor[16, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])\ndef draw[n](x: tensor[n, f32]) -> tensor[16, 2, f32] = dropout(checked(x), 0.5f32)\ndef loss[n](x: tensor[n, f32]) -> tensor[f32] = sum(sum(draw(x), 0), 0)\n"
    ).unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let wire = context.encode().unwrap();
    let decoded = chelis_compiler_api::context::CompiledContext::decode(&wire).unwrap();
    for context in [&context, &decoded] {
        for gradient in [false, true] {
            for count in [32, 34, 32] {
                let ones = std::iter::repeat_n("1.0f32", count)
                    .collect::<Vec<_>>()
                    .join(", ");
                let body = if gradient { "grad(loss)(x)" } else { "draw(x)" };
                let source = format!(
                    "module Probe.Eval\nimport Probe.Draw (draw, loss)\ndef main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n dead = dropout(x, 0.0f32)\n _ = drop(dead)\n {body}\n}}\n"
                );
                let prepared = prepare_eval_in_context(context, &source).unwrap();
                for result in [
                    eval_in_context(context, &source),
                    prepared.eval_root(BTreeMap::new(), "main"),
                    prepared.eval_root(BTreeMap::new(), "main"),
                ] {
                    if count == 32 {
                        assert_eq!(tensor(&result.unwrap(), "main"), mask(1));
                    } else {
                        let error = result.unwrap_err();
                        assert!(
                            error
                                .errors
                                .iter()
                                .any(|error| error.message.contains("claimed = 16")
                                    && error.message.contains("reshape axis 0 = 17")
                                    && error
                                        .message
                                        .contains("numeric trap: domain in reshape at i64")),
                            "gradient={gradient}: {error:?}"
                        );
                    }
                }
            }
        }
        assert_eq!(context.encode().unwrap(), wire);
    }
}

#[test]
fn checked_extent_dropout_helper_keeps_result_claim_and_source_trap_order() {
    for count in [32, 34] {
        let inputs = BTreeMap::from([(
            "x".into(),
            TensorValue {
                shape: vec![count],
                data: wire_values::storage_f32(vec![1.0; count as usize]),
            },
        )]);
        for (body, operation) in [
            (
                "reshape(dropout(x, 0.5f32), [floor_div(shape(x, 0i32), 2i64), 2i64])",
                "reshape",
            ),
            (
                "{\n dead = dropout(x, 1.0f32)\n _ = drop(dead)\n reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])\n}",
                "dropout",
            ),
            (
                "dropout(reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64]), 1.0f32)",
                if count == 32 { "dropout" } else { "reshape" },
            ),
        ] {
            let source = format!(
                "def draw[n](x: tensor[n, f32]) -> tensor[16, 2, f32] = {body}\ndef sample[n](x: tensor[n, f32]) = with seed(42i64) {{ draw(x) }}\n"
            );
            let prepared = prepare_eval(request(&source)).unwrap();
            for result in [
                eval_selected(
                    EvalRequest {
                        source_kind: SourceKind::Surf,
                        source: source.clone(),
                        bindings: inputs.clone(),
                    },
                    &["sample".into()],
                ),
                prepared.eval_root(inputs.clone(), "sample"),
            ] {
                if count == 32 && operation == "reshape" {
                    assert_eq!(tensor(&result.unwrap(), "sample"), mask(0));
                } else {
                    let error = result.unwrap_err();
                    let dtype = if operation == "dropout" { "f32" } else { "i64" };
                    assert!(
                        error.errors.iter().any(|error| error
                            .message
                            .contains(&format!("numeric trap: domain in {operation} at {dtype}"))),
                        "{body}: {error:?}"
                    );
                }
            }
        }
    }
}

// Independent transcription of [05-RNG-1], never an evaluator helper.
fn mask(ordinal: u64) -> Vec<f64> {
    mask_with_seed(42, ordinal)
}

/// [05-OP-37] with rate 0.5 over ones: the f64 comparison reads the exact
/// unit, every narrower dtype its f32 rounding.
fn mask_at_width(seed: u64, ordinal: u64, exact: bool) -> Vec<f64> {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    (0..32)
        .map(|index| {
            let word = mix(seed ^ mix(ordinal).rotate_left(17) ^ mix(index).rotate_left(41));
            let unit = (word >> 11) as f64 / 9007199254740992.0;
            let unit = if exact { unit } else { f64::from(unit as f32) };
            if unit < 0.5 { 0.0 } else { 2.0 }
        })
        .collect()
}

fn mask_with_seed(seed: u64, ordinal: u64) -> Vec<f64> {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    (0..32)
        .map(|index| {
            let word = mix(seed ^ mix(ordinal).rotate_left(17) ^ mix(index).rotate_left(41));
            let unit = ((word >> 11) as f64 / 9007199254740992.0) as f32;
            if unit < 0.5 { 0.0 } else { 2.0 }
        })
        .collect()
}

#[test]
fn selected_tensor_entry_and_reused_preparation_use_canonical_stream() {
    let source = "def sample(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) { dropout(x, 0.5f32) }\n";
    let result = eval_selected(request(source), &["sample".into()]).unwrap();
    assert_eq!(tensor(&result, "sample"), mask(0));
    let prepared = prepare_eval(request(source)).unwrap();
    for _ in 0..2 {
        assert_eq!(
            tensor(&prepared.eval_root(bindings(), "sample").unwrap(), "sample"),
            mask(0)
        );
    }
}

#[test]
fn unselected_invalid_sibling_is_not_an_entered_source_declaration() {
    let source = "def invalid(y: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) { dropout(y, 1.0f32) }\ndef sample(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) { dropout(x, 0.5f32) }\n";
    assert_eq!(
        tensor(
            &eval_selected(request(source), &["sample".into()]).unwrap(),
            "sample"
        ),
        mask(0)
    );
}

#[test]
fn discarded_forward_draw_advances_the_selected_source_stream() {
    let source = "def sample(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {\n dead = dropout(x, 0.0f32)\n dropout(x, 0.5f32)\n}\n";
    assert_eq!(
        tensor(
            &eval_selected(request(source), &["sample".into()]).unwrap(),
            "sample"
        ),
        mask(1)
    );
    assert_ne!(
        mask(0),
        mask(1),
        "the ordinal-removal control must discriminate"
    );
}

#[test]
fn dynamic_helper_applications_get_fresh_forward_keys() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def draw(x: tensor[32, f32]) -> tensor[32, f32] = dropout(x, 0.5f32)\ndef main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n dead = draw(x)\n draw(x)\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()]).unwrap();
    assert_eq!(tensor(&result, "main"), mask(1));
}

#[test]
fn actual_input_gradient_uses_forward_mask_without_advancing_next_draw() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def loss(x: tensor[32, f32]) -> tensor[f32] = sum(dropout(x, 0.5f32), 0)\ndef draw(x: tensor[32, f32]) -> tensor[32, f32] = dropout(x, 0.5f32)\ndef main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n gradient = grad(loss)(x)\n next = draw(x)\n (gradient, next)\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()]).unwrap();
    assert_eq!(tensor(&result, "main.0"), mask(0));
    assert_eq!(tensor(&result, "main.1"), mask(1));
}

#[test]
fn scalar_cotangent_repacking_keeps_dropout_replay_and_next_draw() {
    for dtype in ["f32", "f64"] {
        let ones = std::iter::repeat_n(format!("1.0{dtype}"), 32)
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "def loss(x: {dtype}) -> {dtype} = tensor_to_scalar(dropout(scalar_to_tensor(x), 0.5{dtype}))\ndef main() = with seed(42i64) {{\n derivative = grad(loss)(1.0{dtype})\n next = dropout(to_tensor([{ones}]), 0.5{dtype})\n (derivative, next)\n}}\n"
        );
        let result = eval_selected(request(&source), &["main".into()]).unwrap();
        let derivative = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("main.0"))
            .unwrap();
        let ExecutionValue::Scalar { value } = derivative.value else {
            panic!("{derivative:?}")
        };
        assert_eq!(value.get().prim().name(), dtype);
        assert_eq!(value.get().as_f64_lossy(), mask(0)[0]);
        assert_eq!(tensor(&result, "main.1"), mask(1));
    }
}

#[test]
fn empty_cotangent_repacking_still_executes_retained_dropout() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    for rate in ["0.0f32", "1.0f32"] {
        let source = format!(
            "def loss(xs: List[f32]) -> f32 = {{\n dead = dropout(to_tensor([{ones}]), {rate})\n _ = drop(dead)\n 0.0f32\n}}\ndef main() = with seed(42i64) {{\n _ = grad(loss)([])\n dropout(to_tensor([{ones}]), 0.5f32)\n}}\n"
        );
        let result = eval_selected(request(&source), &["main".into()]);
        if rate == "0.0f32" {
            assert_eq!(tensor(&result.unwrap(), "main"), mask(1));
        } else {
            let error = result.unwrap_err();
            assert!(
                error
                    .errors
                    .iter()
                    .any(|error| error.message == "numeric trap: domain in dropout at f32"),
                "{error:?}"
            );
        }
    }
}

#[test]
fn checked_library_context_and_prepared_context_preserve_raw_stream_binding() {
    use chelis_compiler_api::compiler::{eval_in_context, prepare_eval_in_context};
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!(
        "[package]\nname = \"dropout_probe\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
    )).unwrap();
    std::fs::write(directory.path().join("src/draw.ch"),
        "module Probe.Draw\nexport (draw)\ndef draw(x: tensor[32, f32]) -> tensor[32, f32] = dropout(x, 0.5f32)\n"
    ).unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let cold_wire = context.encode().unwrap();
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "module Probe.Eval\nimport Probe.Draw (draw)\ndef main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n dead = draw(x)\n draw(x)\n}}\n"
    );
    assert_eq!(
        tensor(&eval_in_context(&context, &source).unwrap(), "main"),
        mask(1)
    );
    let prepared = prepare_eval_in_context(&context, &source).unwrap();
    for _ in 0..2 {
        assert_eq!(
            tensor(
                &prepared.eval_root(BTreeMap::new(), "main").unwrap(),
                "main"
            ),
            mask(1)
        );
    }
    assert_eq!(
        context.encode().unwrap(),
        cold_wire,
        "warming the private source-plan memo must not alter cache bytes"
    );
    let decoded = chelis_compiler_api::context::CompiledContext::decode(&cold_wire).unwrap();
    for context in [&context, &decoded] {
        for (seed, ordinal) in [(42, 1), (7, 0), (42, 0), (7, 1)] {
            let mut source = source.replace("seed(42i64)", &format!("seed({seed}i64)"));
            if ordinal == 0 {
                source = source.replace("dead = draw(x)", "");
            }
            let prepared = prepare_eval_in_context(context, &source).unwrap();
            for _ in 0..2 {
                assert_eq!(
                    tensor(
                        &prepared.eval_root(BTreeMap::new(), "main").unwrap(),
                        "main"
                    ),
                    mask_with_seed(seed, ordinal)
                );
            }
        }
        assert_eq!(
            context.encode().unwrap(),
            cold_wire,
            "source-derived memo carries no invocation seed, counter, or realized key"
        );
    }
}

fn assert_explicit_drop_context_parity(
    definition: &str,
    body: &str,
    expected: &[(&str, Vec<f64>)],
) {
    use chelis_compiler_api::compiler::{eval_in_context, prepare_eval_in_context};
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};

    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let main = format!("def main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n {body}\n}}\n");
    let check = |result: EvalResult| {
        for (name, values) in expected {
            assert_eq!(tensor(&result, name), *values, "{definition}\n{main}");
        }
    };
    let monolithic = format!("{definition}\n{main}");
    check(eval_selected(request(&monolithic), &["main".into()]).unwrap());
    let prepared = prepare_eval(request(&monolithic)).unwrap();
    check(prepared.eval_root(BTreeMap::new(), "main").unwrap());

    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!(
        "[package]\nname = \"dropout_drop_probe\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
    )).unwrap();
    std::fs::write(
        directory.path().join("src/draw.ch"),
        format!("module Probe.Draw\nexport (draw)\n{definition}\n"),
    )
    .unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let wire = context.encode().unwrap();
    let decoded = chelis_compiler_api::context::CompiledContext::decode(&wire).unwrap();
    let client = format!("module Probe.Eval\nimport Probe.Draw (draw)\n{main}");
    for context in [&context, &decoded] {
        check(eval_in_context(context, &client).unwrap());
        let prepared = prepare_eval_in_context(context, &client).unwrap();
        for _ in 0..2 {
            check(prepared.eval_root(BTreeMap::new(), "main").unwrap());
        }
        assert_eq!(context.encode().unwrap(), wire);
    }
}

#[test]
fn explicit_library_drop_preserves_dead_forward_and_context_entry_remapping() {
    assert_explicit_drop_context_parity(
        "def draw(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {\n dropped = dropout(x, 0.0f32)\n _ = drop(dropped)\n dropout(x, 0.5f32)\n}",
        "draw(x)",
        &[("main", mask(1))],
    );
}

#[test]
fn explicit_library_drop_preserves_gradient_replay_and_caller_input() {
    let definition = "def draw(x: tensor[32, f32]) -> tensor[f32] = {\n dropped = dropout(x, 0.0f32)\n _ = drop(dropped)\n sum(dropout(x, 0.5f32), 0)\n}";
    assert_explicit_drop_context_parity(
        definition,
        "g = grad(draw)(x)\n (g, dropout(x, 0.5f32), x)",
        &[
            ("main.0", mask(1)),
            ("main.1", mask(2)),
            ("main.2", vec![1.0; 32]),
        ],
    );
    assert_explicit_drop_context_parity(
        definition,
        "g = grad(draw)(x)\n _ = drop(g)\n dropout(x, 0.5f32)",
        &[("main", mask(2))],
    );
}

#[test]
fn fixed_primitive_in_a_host_tuple_uses_the_same_plan_core() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n (dropout(x, 0.5f32), 7i64)\n}}\n"
    );
    assert_eq!(
        tensor(
            &eval_selected(request(&source), &["main".into()]).unwrap(),
            "main.0"
        ),
        mask(0)
    );
    let invalid = source.replace("0.5f32", "1.0f32");
    let error = eval_selected(request(&invalid), &["main".into()]).unwrap_err();
    assert!(
        error
            .errors
            .iter()
            .any(|diagnostic| diagnostic.message == "numeric trap: domain in dropout at f32"),
        "{error:?}"
    );
}

#[test]
fn dropout_failure_preserves_only_the_executed_output_prefix() {
    let source = "def run() -> tensor[1, f32] = with seed(42i64) {\n _ = print(\"before\")\n x = to_tensor([1.0f32])\n dead = dropout(x, 0.0f32)\n value = dropout(x, 1.0f32)\n _ = print(\"after\")\n value\n}\nout = run()\n";
    let error = eval_selected(request(source), &["out".into()]).unwrap_err();
    assert_eq!(error.transcript, ["before"], "{error:?}");
    assert_eq!(error.errors.len(), 1, "{error:?}");
    assert_eq!(
        error.errors[0].message,
        "numeric trap: domain in dropout at f32"
    );
}

#[test]
fn runtime_rate_and_dynamic_control_both_draw_the_handled_stream() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def excluded(x: tensor[32, f32], rate: f32) = (dropout(x, rate), 7i64)\ndef main() = with seed(42i64) {{ excluded(to_tensor([{ones}]), 0.5f32) }}\n"
    );
    // [05-OP-37]: a runtime rate is an ordinary operand, so the helper draws
    // the handled stream's ordinal 0 (chelis#2411).
    let result = eval_selected(request(&source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    assert_eq!(tensor(&result, "main.0"), mask(0));
    // chelis#2405: a dynamic caller no longer excludes its fixed-rate helper,
    // which runs its own plan on the handled stream.
    let source = format!(
        "def draw(x: tensor[32, f32]) -> tensor[32, f32] = dropout(x, 0.5f32)\ndef dynamic(x: tensor[32, f32], condition: bool) = if condition then (draw(x), 7i64) else (x, 7i64)\ndef main() = with seed(42i64) {{ dynamic(to_tensor([{ones}]), true) }}\n"
    );
    let result = eval_selected(request(&source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    assert_eq!(tensor(&result, "main.0"), mask(0));
}

#[test]
fn every_active_float_dtype_and_invalid_domain_use_the_source_plan() {
    for (prim, one_bits) in [
        ("f16", "3c00"),
        ("bf16", "3f80"),
        ("f32", "3f800000"),
        ("f64", "3ff0000000000000"),
    ] {
        for (count, rate) in [(32, "0.5"), (0, "0.0"), (0, "1.0"), (32, "-0.5")] {
            let source = format!(
                "def sample(x: tensor[{count}, {prim}]) -> tensor[{count}, {prim}] = with seed(42i64) {{ dropout(x, {rate}{prim}) }}\n"
            );
            let input = TensorValue {
                shape: vec![count],
                data: serde_json::from_value(
                    serde_json::json!({"dtype":prim,"bits":vec![one_bits; count as usize]}),
                )
                .unwrap(),
            };
            let result = eval_selected(
                EvalRequest {
                    source_kind: SourceKind::Surf,
                    source,
                    bindings: BTreeMap::from([("x".into(), input)]),
                },
                &["sample".into()],
            );
            if rate == "1.0" || rate == "-0.5" {
                let error = result.unwrap_err();
                assert!(
                    error.errors.iter().any(|diagnostic| diagnostic.message
                        == format!("numeric trap: domain in dropout at {prim}")),
                    "{prim}/{count}/{rate}: {error:?}"
                );
            } else {
                let result =
                    result.unwrap_or_else(|error| panic!("{prim}/{count}/{rate}: {error:?}"));
                assert_eq!(
                    tensor(&result, "sample"),
                    if count == 0 { vec![] } else { mask(0) }
                );
            }
        }
    }
}

#[test]
fn mismatch_nonfloat_and_alias_admission_follow_the_shared_signature() {
    use chelis_compiler_api::compiler::check;
    use chelis_compiler_api::schema::CheckRequest;
    for (input, rate) in [("f64", "0.5f32"), ("f16", "0.5bf16"), ("i32", "0i32")] {
        let source = format!(
            "def sample(x: tensor[4, {input}]) = with seed(42i64) {{ dropout(x, {rate}) }}\n"
        );
        assert!(
            !check(CheckRequest {
                source_kind: SourceKind::Surf,
                source
            })
            .unwrap()
            .errors
            .is_empty(),
            "{input}/{rate}"
        );
    }
    assert!(check(CheckRequest {source_kind:SourceKind::Surf,source:"type Values = tensor[4, f64]\ndef sample(x: Values) -> Values = with seed(42i64) { dropout(x, 0.5f64) }\n".into()}).unwrap().errors.is_empty());
}

#[test]
fn dead_draw_input_is_not_required_when_not_data_live_at_the_selected_root() {
    let source = "def sample(x: tensor[32, f32], y: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {\n dead = dropout(y, 0.0f32)\n x\n}\n";
    let mut request = request(source);
    request
        .bindings
        .insert("y".into(), request.bindings["x"].clone());
    let result = eval_selected(request, &["sample".into()]).unwrap();
    assert_eq!(tensor(&result, "sample"), vec![1.0; 32]);
    let entry = result
        .manifest
        .entries
        .iter()
        .find(|entry| entry.name == "sample")
        .unwrap();
    // The discarded draw still takes its ordinal through its key; its data is
    // never read, so the caller need not supply it.
    assert!(!entry.required_inputs.iter().any(|name| name == "y"));
}

#[test]
fn nonunit_cotangent_matches_same_seed_finite_differences() {
    let weights = std::iter::repeat_n("3.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def loss(x: tensor[32, f32]) -> tensor[f32] = sum(mul(dropout(x, 0.5f32), to_tensor([{weights}])), 0)\ndef value(x: tensor[32, f32]) -> tensor[f32] = with seed(42i64) {{ loss(x) }}\ndef derivative(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {{ grad(loss)(x) }}\n"
    );
    let prepared = prepare_eval(request(&source)).unwrap();
    let gradient = tensor(
        &prepared.eval_root(bindings(), "derivative").unwrap(),
        "derivative",
    );
    assert_eq!(
        gradient,
        mask(0).iter().map(|value| 3.0 * value).collect::<Vec<_>>()
    );
    for index in [0, 5, 31] {
        let evaluate = |delta: f32| {
            let mut values = vec![1.0; 32];
            values[index] += delta;
            let input = TensorValue {
                shape: vec![32],
                data: wire_values::storage_f32(values),
            };
            tensor(
                &prepared
                    .eval_root(BTreeMap::from([("x".into(), input)]), "value")
                    .unwrap(),
                "value",
            )[0]
        };
        let finite_difference = (evaluate(0.125) - evaluate(-0.125)) / 0.25;
        assert_eq!(finite_difference, gradient[index]);
    }
}

/// spec/06 §5.2 with [05-OP-37]: a draw inside an activation the selected
/// root runs validates its rate and traps whether or not anything reads its
/// result. The rows are the selected root's own handler (as a value root and
/// as a function root), a discarded inner handler, a discarded call of a
/// helper with its own handler, and a declaration the root references only
/// through a dead binding, both as an inlined function and as a value
/// declaration. The uniform row is [05-OP-8]'s bound check.
///
/// Evidentiary status: REGRESSION TEST. At 3b5f029d8 every row returned the
/// root's value, because a region with no live draw was dropped as unreached.
#[test]
fn a_selected_activations_discarded_draws_still_validate_and_trap() {
    let rows = [
        (
            "own handler, value root",
            "x: tensor[32, f32] = x\nselected = with seed(42i64) {\n dead = dropout(copy(x), 1.0f32)\n copy(x)\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "own handler, function root",
            "def selected(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {\n dead = dropout(copy(x), 1.0f32)\n x\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "discarded inner handler",
            "x: tensor[32, f32] = x\nselected = with seed(42i64) {\n dead = with seed(9i64) { dropout(copy(x), 1.0f32) }\n dropout(copy(x), 0.5f32)\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "discarded helper with its own handler",
            "def h(v: tensor[32, f32]) -> tensor[32, f32] = with seed(9i64) {\n dead = dropout(copy(v), 1.0f32)\n v\n}\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {\n a = h(copy(x))\n dropout(x, 0.5f32)\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "dead reference to a function declaration",
            "def sampled(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) { dropout(x, 1.0f32) }\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = {\n dead = sampled(copy(x))\n x\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "dead reference to a value declaration",
            "x: tensor[32, f32] = x\nsampled = with seed(42i64) { dropout(x, 1.0f32) }\nselected = {\n dead = sampled\n copy(x)\n}\nunrelated = with seed(7i64) { dropout(x, 0.5f32) }\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "discarded uniform with reversed bounds",
            "x: tensor[32, f32] = x\nselected = with seed(42i64) {\n dead = uniform_like(copy(x), 1.0f32, 0.0f32)\n copy(x)\n}\n",
            "numeric trap: domain in uniform_like at f32",
        ),
    ];
    for (row, source, trap) in rows {
        let error = eval_selected(request(source), &["selected".into()])
            .map(|result| format!("{:?}", result.roots))
            .expect_err(row);
        assert!(
            error.errors.iter().any(|error| error.message == trap),
            "{row}: {error:?}"
        );
    }
}

/// The complement of the trap rows: selecting one root never runs another
/// declaration's activation, so its invalid rate does not trap and its draws
/// take no ordinal of the selected stream.
#[test]
fn an_unselected_declarations_invalid_draw_does_not_run() {
    let source = "x: tensor[32, f32] = x\nselected = with seed(42i64) { dropout(copy(x), 0.5f32) }\nunrelated = with seed(42i64) { dropout(x, 1.0f32) }\n";
    let result = eval_selected(request(source), &["selected".into()]).unwrap();
    assert_eq!(tensor(&result, "selected"), mask(0));
    let source = "def unrelated(y: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) { dropout(y, 1.0f32) }\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {\n dead = dropout(copy(x), 0.5f32)\n dropout(x, 0.5f32)\n}\n";
    let result = eval_selected(request(source), &["selected".into()]).unwrap();
    assert_eq!(tensor(&result, "selected"), mask(1));
}

/// [05-RNG-1]: keeping a selected activation's discarded draws shifts no
/// ordinal. A discarded draw takes the ordinal it owns, and a helper's
/// discarded draw is taken on the stream it inherits.
#[test]
fn a_selected_activations_discarded_draws_keep_their_ordinals() {
    let helper_dead_only = "def h(v: tensor[32, f32]) -> tensor[32, f32] ! { Random } = {\n dead = dropout(copy(v), 0.5f32)\n v\n}\n";
    let rows = [
        (
            "value root, discarded then live",
            "x: tensor[32, f32] = x\nselected = with seed(42i64) {\n dead = dropout(copy(x), 0.5f32)\n dropout(copy(x), 0.5f32)\n}\n".to_string(),
            mask(1),
        ),
        (
            "inherited helper whose only draw is discarded",
            format!("{helper_dead_only}def selected(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {{\n a = h(copy(x))\n dropout(x, 0.5f32)\n}}\n"),
            mask(1),
        ),
        (
            "discarded call of an inherited helper",
            "def h(v: tensor[32, f32]) -> tensor[32, f32] ! { Random } = dropout(v, 0.5f32)\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {\n dead = h(copy(x))\n dropout(x, 0.5f32)\n}\n".to_string(),
            mask(1),
        ),
        (
            "inherited helper, discarded then live",
            "def h(v: tensor[32, f32]) -> tensor[32, f32] ! { Random } = {\n dead = dropout(copy(v), 0.5f32)\n dropout(v, 0.5f32)\n}\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) { h(x) }\n".to_string(),
            mask(1),
        ),
        (
            "runtime if whose unselected arm draws",
            "def pick(v: tensor[32, f32], flag: bool) -> tensor[32, f32] ! { Random } = if flag then dropout(v, 0.5f32) else v\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {\n flag = lt(tensor_to_scalar(sum(copy(x), 0i32)), 0.0f32)\n a = pick(copy(x), flag)\n add(a, dropout(x, 0.5f32))\n}\n".to_string(),
            mask(0).iter().map(|value| value + 1.0).collect(),
        ),
    ];
    assert_ne!(mask(0), mask(1), "the ordinal rows must discriminate");
    for (row, source, expected) in rows {
        let result = eval_selected(request(&source), &["selected".into()])
            .unwrap_or_else(|error| panic!("{row}\n{error:?}"));
        assert_eq!(tensor(&result, "selected"), expected, "{row}");
    }
}

#[test]
fn constant_loss_preserves_dead_forward_draw_and_every_zero_gradient_coordinate() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def loss(x: tensor[32, f32]) -> f32 = {{\n dead = dropout(x, 0.5f32)\n 3.0f32\n}}\ndef draw(x: tensor[32, f32]) -> tensor[32, f32] = dropout(x, 0.5f32)\ndef main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n gradient = grad(loss)(x)\n (gradient, draw(x))\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()]).unwrap();
    let gradient = tensor(&result, "main.0");
    assert_eq!(gradient, vec![0.0; 32]);
    assert!(gradient.iter().all(|value| value.to_bits() == 0));
    assert_eq!(tensor(&result, "main.1"), mask(1));
}

fn ones32() -> String {
    std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ")
}

fn scalar_root(result: &EvalResult, name: &str) -> f64 {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap();
    let ExecutionValue::Scalar { value } = &root.value else {
        panic!("{root:?}")
    };
    value.get().as_f64_lossy()
}

/// chelis#2405: a fixed-rate draw beneath a recursive or dynamic caller runs
/// on the handled stream. The caller's control flow used to become an
/// execution exclusion inherited by every nested dispatch, which sent the
/// draw to the host interpreter's builtin table, where `dropout` does not
/// exist, and eval failed with "unknown runtime name `dropout`" on programs
/// the C lane runs.
///
/// Evidentiary status: REGRESSION TEST (each row fails on the base with that
/// error).
#[test]
fn issue_2405_dropout_beneath_recursion_and_runtime_if_runs_on_the_handled_stream() {
    let ones = ones32();
    let recursion = format!(
        "def rep(x: tensor[32, f32], n: i64) -> tensor[32, f32] ! {{ Random }} = if eq(n, 0i64) then x else rep(dropout(x, 0.5f32), sub(n, 1i64))\ndef main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n r = rep(copy(x), 3i64)\n after = dropout(x, 0.5f32)\n (r, after)\n}}\n"
    );
    let result = eval_selected(request(&recursion), &["main".into()])
        .unwrap_or_else(|error| panic!("{recursion}\n{error:?}"));
    let kept = (0..32)
        .map(|index| mask(0)[index] * mask(1)[index] * mask(2)[index])
        .collect::<Vec<_>>();
    assert!(kept.contains(&8.0) && kept.contains(&0.0));
    assert_eq!(tensor(&result, "main.0"), kept);
    assert_eq!(tensor(&result, "main.1"), mask(3));

    // An untaken branch holding the draw consumes no ordinal; a taken one
    // consumes one.
    for (flag, first, next) in [("false", vec![1.0; 32], 0), ("true", mask(0), 1)] {
        let branch = format!(
            "def pick(x: tensor[32, f32], flag: bool) -> tensor[32, f32] ! {{ Random }} = if flag then dropout(x, 0.5f32) else x\ndef main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n a = pick(copy(x), {flag})\n b = dropout(x, 0.5f32)\n (a, b)\n}}\n"
        );
        let result = eval_selected(request(&branch), &["main".into()])
            .unwrap_or_else(|error| panic!("{branch}\n{error:?}"));
        assert_eq!(tensor(&result, "main.0"), first, "{flag}");
        assert_eq!(tensor(&result, "main.1"), mask(next), "{flag}");
    }
}

/// chelis#2405: a `match` in a definition that draws nothing no longer
/// removes `dropout` from the rest of the program, whether the draw sits
/// beside the call inside the handler or in a separate handler.
///
/// Evidentiary status: REGRESSION TEST (both rows fail on the base with
/// "unknown runtime name `dropout`").
#[test]
fn issue_2405_unrelated_match_leaves_dropout_on_the_handled_stream() {
    let ones = ones32();
    let scale = "type Mode = | Train | Infer\ndef scale(m: Mode) -> f32 =\n  match m with {\n    | Train => 2.0f32\n    | Infer => 1.0f32\n  }\n";
    for main in [
        format!(
            "def main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n s = scale(Train)\n d = dropout(x, 0.5f32)\n (d, s)\n}}\n"
        ),
        format!(
            "def main() = {{\n s = scale(Train)\n d = with seed(42i64) {{ dropout(to_tensor([{ones}]), 0.5f32) }}\n (d, s)\n}}\n"
        ),
    ] {
        let source = format!("{scale}{main}");
        let result = eval_selected(request(&source), &["main".into()])
            .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
        assert_eq!(tensor(&result, "main.0"), mask(0));
        assert_eq!(scalar_root(&result, "main.1"), 2.0);
    }
}

// Independent transcription of [05-RNG-1] and [05-OP-8] for `uniform_like`
// over [0, 1), never an evaluator helper: the drawn value is the unit value
// rounded to f32.
fn spec_unit_uniform(seed: u64, ordinal: u64, count: u64) -> Vec<f32> {
    fn mix(mut x: u64) -> u64 {
        x = x.wrapping_add(0x9e3779b97f4a7c15);
        x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
        x ^ (x >> 31)
    }
    (0..count)
        .map(|index| {
            let word = mix(seed ^ mix(ordinal).rotate_left(17) ^ mix(index).rotate_left(41));
            ((word >> 11) as f64 / 9007199254740992.0) as f32
        })
        .collect()
}

/// chelis#2405 retired the execution exclusion that a recursive program's
/// helpers used to run under, so a drawing helper beneath recursion now
/// takes the planned kernel entry instead of the legacy one. Its
/// `uniform_like` draws take one ordinal per application in execution order,
/// and the counter carries to the draws that follow: `a` sums the draws at
/// ordinals 0 to 2, and `b` and `c` are ordinals 3 and 4.
///
/// Evidentiary status: DISPOSITION LOCK for the ordinals (chelis#2405's base
/// consumed the same ones). The values are [05-RNG-1]'s since chelis#2408;
/// the pinned bits are exact-rational evaluations from the assessment's
/// `rng_ref.py uniform 7 ORDINAL 4 0 1 f32`.
#[test]
fn uniform_draws_beneath_recursion_keep_their_stream() {
    let source = "def draw(x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = uniform_like(x, 0.0f32, 1.0f32)\ndef walk(n: i64, x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = if eq(n, 0i64) then x else add(draw(copy(x)), walk(sub(n, 1i64), x))\ndef main() = with seed(7i64) {\n x = to_tensor([0.0f32, 0.0f32, 0.0f32, 0.0f32])\n a = walk(3i64, copy(x))\n b = draw(copy(x))\n c = uniform_like(x, 0.0f32, 1.0f32)\n (a, b, c)\n}\n";
    let result = eval_selected(request(source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    let draws = (0..5)
        .map(|ordinal| spec_unit_uniform(7, ordinal, 4))
        .collect::<Vec<_>>();
    let bits = |values: &[f32]| {
        values
            .iter()
            .map(|value| value.to_bits())
            .collect::<Vec<_>>()
    };
    let observed = |root: &str| {
        tensor(&result, root)
            .iter()
            .map(|value| (*value as f32).to_bits())
            .collect::<Vec<_>>()
    };
    let walked = (0..4)
        .map(|index| draws[0][index] + (draws[1][index] + draws[2][index]))
        .collect::<Vec<_>>();
    for (root, expected, pinned) in [
        (
            "main.0",
            bits(&walked),
            [0x3f2e_b639, 0x4022_8700, 0x3f70_aea6, 0x3fac_e925],
        ),
        (
            "main.1",
            bits(&draws[3]),
            [0x3edf_9140, 0x3d0d_592a, 0x3f12_e0b1, 0x3f72_3538],
        ),
        (
            "main.2",
            bits(&draws[4]),
            [0x3f0f_dbaa, 0x3f2e_03e3, 0x3f4f_a3f4, 0x3d21_7fa8],
        ),
    ] {
        assert_eq!(
            expected, pinned,
            "{root}: the transcription and rng_ref.py agree"
        );
        assert_eq!(observed(root), expected, "{root}");
    }
}

/// chelis#2405: `grad` of a function whose draw sits under runtime control,
/// and `vmap` of a drawing function, lower their `dropout` without a
/// fixed-control plan. The evaluator refuses that draw loudly. Before the
/// fence it returned a mask from the pre-[05-RNG-1] formula, and retiring
/// the inherited exclusion had removed the unrelated error that used to hide
/// that value behind a later draw.
///
/// Evidentiary status: REGRESSION TEST (every row returns values, not an
/// error, without the fence).
#[test]
fn dropout_under_grad_of_dynamic_control_draws_and_vmap_is_refused() {
    let ones = ones32();
    let row = "[1.0f32, 1.0f32, 1.0f32, 1.0f32]";
    // Dropout under a runtime `if` inside `grad` draws the taken arm's key at
    // ordinal 0 and replays it for the pathwise adjoint; the following draw
    // takes ordinal 1.
    let source = format!(
        "def loss(x: tensor[32, f32]) -> tensor[f32] ! {{ Random }} = if gt(tensor_to_scalar(sum(copy(x), 0i32)), 0.0f32) then sum(dropout(x, 0.5f32), 0i32) else sum(x, 0i32)\ndef main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n g = grad(loss)(copy(x))\n after = dropout(x, 0.5f32)\n (g, after)\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    assert_eq!(tensor(&result, "main.0"), mask(0));
    assert_eq!(tensor(&result, "main.1"), mask(1));
    // chelis#2409: vmap over a function that draws has no conforming stream
    // until explicit keys, so it is refused rather than drawn at seed zero.
    let vmap_keep =
        "def keep(x: tensor[4, f32]) -> tensor[4, f32] ! { Random } = dropout(x, 0.5f32)\n";
    for source in [
        format!(
            "{vmap_keep}def main() = with seed(42i64) {{\n xs = to_tensor([{row}, {row}, {row}])\n ys = vmap(keep)(xs)\n after = dropout(to_tensor({row}), 0.5f32)\n (ys, after)\n}}\n"
        ),
        format!(
            "{vmap_keep}def main() = with seed(42i64) {{\n xs = to_tensor([{row}, {row}, {row}])\n vmap(keep)(xs)\n}}\n"
        ),
    ] {
        let error = eval_selected(request(&source), &["main".into()]).unwrap_err();
        assert!(
            error.errors.iter().any(|error| {
                error
                    .message
                    .contains("`vmap` over a function that draws from `Random`")
                    && error.message.contains("unimplemented chelis#2409")
            }),
            "{source}\n{error:?}"
        );
    }
}

const CUBE_SLOPE_HESS: &str = "def cube(z: tensor[4, f32]) -> tensor[f32] = sum(mul(mul(copy(z), copy(z)), z), 0i32)\ndef slope(y: tensor[4, f32]) -> tensor[f32] = sum(grad(cube)(y), 0i32)\ndef hess(x: tensor[4, f32]) -> tensor[4, f32] = grad(slope)(x)\n";

/// chelis#2405 round 1: a draw-free nested `grad` has no draw to plan, but
/// the execution spine still cannot lower it, so the evaluation program
/// keeps it on the compatibility route. Admitting it to the spine failed
/// the whole program, even when the Hessian helper was never called.
///
/// Evidentiary status: REGRESSION TEST (both rows fail on the round-1 head
/// with "execution spine lost source node").
#[test]
fn issue_2405_draw_free_nested_grad_is_not_admitted_to_the_spine() {
    let unused = format!(
        "{CUBE_SLOPE_HESS}def main() = add(to_tensor([1.0f32, 2.0f32]), to_tensor([3.0f32, 4.0f32]))\n"
    );
    let result = eval_selected(request(&unused), &["main".into()])
        .unwrap_or_else(|error| panic!("{unused}\n{error:?}"));
    assert_eq!(tensor(&result, "main"), vec![4.0, 6.0]);

    let used = format!(
        "{CUBE_SLOPE_HESS}def main() = {{\n x = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])\n hess(x)\n}}\n"
    );
    let result = eval_selected(request(&used), &["main".into()])
        .unwrap_or_else(|error| panic!("{used}\n{error:?}"));
    assert_eq!(tensor(&result, "main"), vec![6.0, 12.0, 18.0, 24.0]);
}

/// chelis#2405 round 1: the same through a Reef library. A context package
/// exporting a draw-free Hessian must not break its importers, including
/// one that only calls an unrelated export.
///
/// Evidentiary status: REGRESSION TEST (every row fails on the round-1
/// head).
#[test]
fn issue_2405_library_exporting_a_draw_free_hessian_keeps_its_importers() {
    use chelis_compiler_api::compiler::eval_in_context;
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"hessian\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
    std::fs::write(
        directory.path().join("src/calc.ch"),
        format!("module Probe.Calc\nexport (cube, hess)\n{CUBE_SLOPE_HESS}"),
    )
    .unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let decoded =
        chelis_compiler_api::context::CompiledContext::decode(&context.encode().unwrap()).unwrap();
    let cube_only = "module Probe.Client\nimport Probe.Calc (cube)\ndef main() = tensor_to_scalar(cube(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])))\n";
    let both = "module Probe.Client\nimport Probe.Calc (cube, hess)\ndef main() = {\n x = to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32])\n (tensor_to_scalar(cube(copy(x))), hess(x))\n}\n";
    for context in [&context, &decoded] {
        let result = eval_in_context(context, cube_only)
            .unwrap_or_else(|error| panic!("{cube_only}\n{error:?}"));
        assert_eq!(scalar_root(&result, "main"), 100.0);
        let result =
            eval_in_context(context, both).unwrap_or_else(|error| panic!("{both}\n{error:?}"));
        assert_eq!(scalar_root(&result, "main.0"), 100.0);
        assert_eq!(tensor(&result, "main.1"), vec![6.0, 12.0, 18.0, 24.0]);
    }
}

/// chelis#2405 round 2: a transform whose target is a captured local
/// closure classifies against a definition table that includes the closure,
/// so a draw inside it is still seen and planned. The scope's cached
/// snapshot, which the uncaptured case reads, does not contain the closure.
///
/// Evidentiary status: DISPOSITION LOCK (it passes before and after the
/// round-2 change, and fails if the cached snapshot is read for a captured
/// closure).
#[test]
fn grad_of_a_captured_drawing_closure_still_sees_its_draw() {
    let ones = ones32();
    let source = format!(
        "def main() = with seed(42i64) {{\n x = to_tensor([{ones}])\n keep = fn (v: tensor[32, f32]) -> sum(dropout(v, 0.5f32), 0i32)\n g = grad(keep)(copy(x))\n next = dropout(x, 0.5f32)\n (g, next)\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()])
        .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
    assert_eq!(tensor(&result, "main.0"), mask(0));
    assert_eq!(tensor(&result, "main.1"), mask(1));
}

/// [05-RNG-1] enters only the selected arm of a runtime `if`, so in the DAG
/// evaluator a draw in an unselected arm neither validates its controls nor
/// takes an ordinal (chelis#2410): the whole-program graph computes both arms
/// of a `where`, and each draw there carries its arm's path condition as its
/// activation. Each row adds the arm's value to a later draw, so a shifted
/// ordinal changes the result. Flags come from data, so lowering cannot fold
/// them.
///
/// Evidentiary status: REGRESSION TEST for the row taking `grad` through an
/// unselected drawing arm, which dcc9256c4 refused with the #2410 rejection.
/// The other rows passed there and lock this lane's stream beside the
/// lowering change.
#[test]
fn a_draw_in_an_unselected_arm_takes_no_ordinal_in_the_dag_evaluator() {
    let sum = "tensor_to_scalar(sum(copy(x), 0i32))";
    let selected = |body: &str| {
        format!(
            "def selected(x: tensor[32, f32]) -> tensor[32, f32] = with seed(42i64) {{\n s = {sum}\n{body}}}\n"
        )
    };
    let unit = |ordinal| spec_unit_uniform(42, ordinal, 32);
    let add = |left: &[f32], right: &[f32]| -> Vec<f32> {
        left.iter()
            .zip(right)
            .map(|(left, right)| left + right)
            .collect()
    };
    let as_f32 = |values: Vec<f64>| {
        values
            .into_iter()
            .map(|value| value as f32)
            .collect::<Vec<_>>()
    };
    let ones = vec![1.0_f32; 32];
    // [05-OP-37] at rate 0.25 over ones: a kept element is `1 / 0.75` at
    // binary32.
    let quarter_rate = unit(0)
        .into_iter()
        .map(|unit| if unit < 0.25 { 0.0 } else { 1.0_f32 / 0.75_f32 })
        .collect::<Vec<_>>();
    let noisy = "def layer(x: tensor[32, f32], noisy: bool, eps: f32) -> tensor[32, f32] ! { Random } = if noisy then add(copy(x), uniform_like(x, neg(eps), eps)) else x\n";
    let loss = |comparison: &str| {
        format!(
            "def loss(x: tensor[32, f32]) -> tensor[f32] ! {{ Random }} = if {comparison}({sum}, 0.0f32) then sum(dropout(x, 0.5f32), 0i32) else sum(x, 0i32)\n"
        )
    };
    let rows = [
        (
            "uniform, invalid run-time bounds, unselected",
            format!(
                "{noisy}{}",
                selected(" layer(x, lt(s, 0.0f32), sub(0.0f32, s))\n")
            ),
            ones.clone(),
        ),
        (
            "uniform, valid run-time bounds, unselected, then a draw",
            format!(
                "{noisy}{}",
                selected(
                    " add(layer(copy(x), lt(s, 0.0f32), s), uniform_like(x, 0.0f32, 1.0f32))\n"
                )
            ),
            add(&ones, &unit(0)),
        ),
        (
            "uniform in a nested arm, then a draw",
            format!(
                "def pick(x: tensor[32, f32], a: bool, b: bool) -> tensor[32, f32] ! {{ Random }} = if a then if b then uniform_like(x, 0.0f32, 1.0f32) else x else x\n{}",
                selected(
                    " add(pick(copy(x), gt(s, 0.0f32), lt(s, 0.0f32)), uniform_like(x, 0.0f32, 1.0f32))\n"
                )
            ),
            add(&ones, &unit(0)),
        ),
        (
            "dropout in both arms, then a draw",
            format!(
                "def pick(x: tensor[32, f32], flag: bool) -> tensor[32, f32] ! {{ Random }} = if flag then dropout(x, 0.5f32) else dropout(x, 0.25f32)\n{}",
                selected(" add(pick(copy(x), lt(s, 0.0f32)), dropout(x, 0.5f32))\n")
            ),
            add(&quarter_rate, &as_f32(mask(1))),
        ),
        (
            "grad through an unselected drawing arm, then a draw",
            format!(
                "{}{}",
                loss("lt"),
                selected(" add(grad(loss)(copy(x)), dropout(x, 0.5f32))\n")
            ),
            add(&ones, &as_f32(mask(0))),
        ),
        (
            "grad through a selected drawing arm, then a draw",
            format!(
                "{}{}",
                loss("gt"),
                selected(" add(grad(loss)(copy(x)), dropout(x, 0.5f32))\n")
            ),
            add(&as_f32(mask(0)), &as_f32(mask(1))),
        ),
    ];
    assert_ne!(unit(0), unit(1));
    let mut failures = Vec::new();
    for (row, source, expected) in rows {
        match eval_selected(request(&source), &["selected".into()]) {
            Ok(result) => {
                let actual = as_f32(tensor(&result, "selected"));
                let bits = |values: &[f32]| {
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                if bits(&actual) != bits(&expected) {
                    failures.push(format!("{row}: {actual:?}"));
                }
            }
            Err(error) => failures.push(format!("{row}: {error:?}")),
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}
