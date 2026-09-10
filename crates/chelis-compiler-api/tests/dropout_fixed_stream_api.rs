//! API source-shell acceptance for the bounded fixed-control dropout plan.
#![allow(deprecated)] // Explicit compatibility/parity coverage for prepare_eval.
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

use chelis_compiler_api::compiler::{eval_selected, prepare_eval};
use chelis_compiler_api::schema::{
    EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use std::collections::BTreeMap;

fn bindings() -> BTreeMap<String, TensorValue> {
    BTreeMap::from([(
        "x".into(),
        TensorValue {
            shape: vec![32],
            data: wire_values::storage_f32(vec![1.0; 32]),
        },
    )])
}

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: bindings(),
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

// Independent transcription of [05-RNG-1], never an evaluator helper.
fn mask(ordinal: u64) -> Vec<f64> {
    mask_with_seed(42, ordinal)
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
fn runtime_rate_and_dynamic_control_keep_explicit_legacy_dispatch() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    for (helper, stage, diagnostic) in [
        (
            "def excluded(x: tensor[32, f32], rate: f32) = (dropout(x, rate), 7i64)\ndef main() = with seed(42i64) { excluded(to_tensor(ONES), 0.5f32) }\n",
            "lower",
            "requires a statically-resolvable rate",
        ),
        (
            "def draw(x: tensor[32, f32]) -> tensor[32, f32] = dropout(x, 0.5f32)\ndef excluded(x: tensor[32, f32], condition: bool) = if condition then (draw(x), 7i64) else (x, 7i64)\ndef main() = with seed(42i64) { excluded(to_tensor(ONES), true) }\n",
            "eval",
            "unknown runtime name `dropout`",
        ),
    ] {
        let source = helper.replace("ONES", &format!("[{ones}]"));
        let error = eval_selected(request(&source), &["main".into()]).unwrap_err();
        assert_eq!(error.stage, stage, "{error:?}");
        assert!(
            error
                .errors
                .iter()
                .any(|error| error.message.contains(diagnostic)),
            "{error:?}"
        );
    }
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
    for (input, rate) in [("f64", "0.5f32"), ("f16", "0.5bf16"), ("int32", "0i32")] {
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
fn dead_draw_input_is_required_even_when_not_data_live_at_the_selected_root() {
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
    assert!(entry.required_inputs.iter().any(|name| name == "y"));
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

#[test]
fn selected_declaration_keeps_a_dead_reference_to_an_effecting_declaration() {
    let source = "x: tensor[32, f32] = x\nsampled = with seed(42i64) { dropout(x, 1.0f32) }\nselected = {\n dead = sampled\n copy(x)\n}\nunrelated = with seed(7i64) { dropout(x, 0.5f32) }\n";
    let error = eval_selected(request(source), &["selected".into()]).unwrap_err();
    assert!(
        error
            .errors
            .iter()
            .any(|error| error.message == "numeric trap: domain in dropout at f32"),
        "{error:?}"
    );
    let result = eval_selected(request(source), &["unrelated".into()]).unwrap();
    assert!(
        result
            .roots
            .iter()
            .any(|root| root.name.as_deref() == Some("unrelated"))
    );
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
