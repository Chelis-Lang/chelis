//! API source-shell acceptance for keyed dropout: static and runtime rates,
//! generic specialization, host boundaries, library contexts, gradients, and
//! discarded draws (chelis#2413 moved every draw onto an explicit key).
//!
//! Every expected draw comes from `key_reference`, the transcription of
//! `key_ref.py`, never from an evaluator helper. The programs name their keys
//! as `key_reference` does: `k1` and `k2` split `key_from_seed(42i64)`; with
//! three or four keys, `k1` is the left half and the rest come from
//! splitting the remainder. A single-key program draws with
//! `key_from_seed(7i64)`, whose first four elements are mixed; the first four
//! of `key_from_seed(42i64)` all drop, which no 4-element test could read.
#![allow(deprecated)] // Explicit compatibility/parity coverage for prepare_eval.
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

mod key_reference;
mod ownership_support;

use chelis_compiler_api::compiler::{
    CompilerError, compile_for_execution_in_context, eval_in_context, eval_selected, prepare_eval,
};
use chelis_compiler_api::context::CompiledContext;
use chelis_compiler_api::schema::{
    CompileTarget, EvalRequest, EvalResult, ExecutionValue, SourceKind, TensorValue,
};
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
use chelis_types::types::Lane;
use key_reference::{FOUR_KEYS, THREE_KEYS, TWO_KEYS, four_keys, three_keys, two_keys};
use std::collections::BTreeMap;

/// `key_from_seed(7i64)`.
fn key7() -> u64 {
    key_reference::key_from_seed(7)
}

/// The rate-0.5 dropout of 32 ones under `key` at a width narrower than f64.
fn mask(key: u64) -> Vec<f64> {
    key_reference::mask(key, 32)
}

/// The rate-0.5 dropout of 32 ones under `key`, read at f64's exact width
/// when `exact`.
fn mask_at_width(key: u64, exact: bool) -> Vec<f64> {
    key_reference::mask_at_width(key, 32, exact)
}

/// The reference agrees with the values `key_ref.py` prints: its worked
/// values, and the seed-7 and split masks these tests read (printed by
/// `key_ref.py` with `unit` rounded to f32 and compared with 0.5).
#[test]
fn key_reference_matches_key_ref_py() {
    let key = key7();
    assert_eq!(
        key_reference::split(key),
        (0xaa38_9617_2f9a_3213, 0x8fd0_6b2e_7bad_8630)
    );
    assert_eq!(key_reference::fold_in(key, 3), 0x53c6_f7e8_3810_b049);
    assert_eq!(key_reference::word(key, 0), 0x2065_4588_fcd2_5740);
    let bits = |mask: Vec<f64>| {
        mask.iter()
            .map(|value| if *value == 2.0 { '1' } else { '0' })
            .collect::<String>()
    };
    assert_eq!(bits(mask(key)), "01110111101001111111110100000100");
    assert_eq!(bits(mask(two_keys().1)), "11001101110110001111111110111100");
    assert_eq!(
        bits(key_reference::mask(key_reference::key_from_seed(42), 4)),
        "0000"
    );
}

#[test]
fn concrete_static_rate_calls_keep_source_bindings_across_host_boundaries() {
    // [05-OP-37], #1764: source-static actuals are not runtime rates.
    for definition in [
        "def keep(k: key, x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(k, x, rate)\ndef draw(k: key, x: tensor[4, f32]) -> tensor[4, f32] = keep(k, x, 0.5f32)",
        "def draw(k: key, x: tensor[4, f32]) -> tensor[4, f32] = { rate = 0.5f32\n alias = rate\n dropout(k, x, alias) }",
        "def draw(k: key, x: tensor[4, f32]) -> tensor[4, f32] = { keep = fn (j: key, v: tensor[4, f32], rate: f32) -> dropout(j, v, rate)\n keep(k, x, 0.5f32) }",
        "def keep(k: key, x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(k, x, rate)\ndef draw(k: key, x: tensor[4, f32]) -> tensor[4, f32] = { keep = fn (j: key, v: tensor[4, f32], rate: f32) -> dropout(j, v, 0.5f32)\n keep(k, x, 0.0f32) }",
    ] {
        let source = format!(
            "{definition}\ndef loss(k: key, x: tensor[4, f32]) -> tensor[f32] = sum(draw(k, x), 0i32)\ndef main() = {{\n {THREE_KEYS}\n x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n first = draw(k1, copy(x))\n backward = grad(loss, wrt=x)(k2, copy(x))\n next = dropout(k3, x, 0.5f32)\n (first, backward, next, x)\n}}\n"
        );
        let result = eval_selected(request(&source), &["main".into()])
            .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
        let (k1, k2, k3) = three_keys();
        for (index, key) in [k1, k2, k3].into_iter().enumerate() {
            assert_eq!(tensor(&result, &format!("main.{index}")), mask(key)[..4]);
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
    std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"static-rate\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
    std::fs::write(directory.path().join("src/draw.ch"), "module Probe.Draw\nexport (keep)\ndef keep(k: key, x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(k, x, rate)\n").unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let decoded =
        chelis_compiler_api::context::CompiledContext::decode(&context.encode().unwrap()).unwrap();
    let source = format!(
        "module Probe.Client\nimport Probe.Draw (keep)\ndef main() = {{\n {THREE_KEYS}\n x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n first = keep(k1, copy(x), 0.0f32)\n second = keep(k2, copy(x), 0.5f32)\n (first, second, dropout(k3, x, 0.5f32))\n}}\n"
    );
    let (_, k2, k3) = three_keys();
    for context in [&context, &decoded] {
        let result = eval_in_context(context, &source).unwrap();
        assert_eq!(tensor(&result, "main.0"), vec![1.0; 4]);
        assert_eq!(tensor(&result, "main.1"), mask(k2)[..4]);
        assert_eq!(tensor(&result, "main.2"), mask(k3)[..4]);
        let prepared = prepare_eval_in_context(context, &source).unwrap();
        for _ in 0..2 {
            let result = prepared.eval_root(BTreeMap::new(), "main").unwrap();
            assert_eq!(tensor(&result, "main.1"), mask(k2)[..4]);
            assert_eq!(tensor(&result, "main.2"), mask(k3)[..4]);
        }
    }
}

#[test]
fn compiled_static_rate_exported_library_call_survives_context_decode() {
    use chelis_compiler_api::compiler::compile_for_execution_in_context;
    use chelis_compiler_api::schema::CompileTarget;
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};

    // [05-OP-37]: library source must retain the selected entry's execution
    // authority across the serialized context boundary.
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"static-rate\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
    std::fs::write(directory.path().join("src/draw.ch"), "module Probe.Draw\nexport (keep)\ndef keep(k: key, x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(k, x, rate)\n").unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let decoded =
        chelis_compiler_api::context::CompiledContext::decode(&context.encode().unwrap()).unwrap();
    let source = "module Probe.Client\nimport Probe.Draw (keep)\ndef main(x: tensor[4, f32]) -> tensor[4, f32] = keep(key_from_seed(7i64), x, 0.5f32)\n";
    for context in [&decoded, &context] {
        let artifact =
            compile_for_execution_in_context(context, source, CompileTarget::C, Some("main"))
                .expect("a keyed library entry must compile through the context API");
        assert_eq!(artifact.inputs.len(), 1);
        assert_eq!(artifact.outputs.len(), 1);
    }
}

// chelis#2411: a runtime rate is an ordinary [05-OP-37] operand; the call
// draws its key's mask, which the reference recomputes from the spec text.
#[test]
fn concrete_runtime_rate_actual_draws_its_keys_mask() {
    let source = "def keep(k: key, x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(k, x, rate)\ndef main() = keep(key_from_seed(7i64), to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]), tensor_to_scalar(scalar_to_tensor(0.5f32)))\n";
    let result = eval_selected(request(source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    assert_eq!(tensor(&result, "main"), mask(key7())[..4]);
}

#[test]
fn unrelated_scalar_capture_preserves_public_acceptance_and_next_draw() {
    // PR1807 R1: an unrelated rebound scalar is not a closure dependency.
    let ones = "to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32, 1.0f32])";
    let (k1, k2) = two_keys();
    for function_body in [
        "dropout(j, v, 0.5f32)",
        "{ ignored = unrelated\n dropout(j, v, 0.5f32) }",
    ] {
        for shadow in [false, true] {
            for next_draw in [false, true] {
                let source = format!(
                    "def sample(k: key, x: tensor[8, f32]) -> tensor[8, f32] = {{\n\
                 unrelated = 0.5f32\n\
                 f = fn (j: key, v: tensor[8, f32]) -> {function_body}\n\
                 {}\n f(k, x)\n}}\n\
                 def main() = {{\n {TWO_KEYS}\n first = sample(k1, {ones})\n {}\n}}",
                    if shadow { "unrelated = 0.25f32" } else { "" },
                    if next_draw {
                        format!("add(first, dropout(k2, {ones}, 0.5f32))")
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
                    mask(k1)
                        .iter()
                        .zip(mask(k2))
                        .take(8)
                        .map(|(a, b)| a + b)
                        .collect::<Vec<_>>()
                } else {
                    mask(k1)[..8].to_vec()
                };
                assert_eq!(tensor(&result, "main"), expected);
                assert!(result.transcript.is_empty());
            }
        }
    }
}

#[test]
fn generic_static_rate_cast_reaches_the_evaluator_plan() {
    // [05-OP-37], chelis#1764: specialize the source call while its checked
    // actual dtype and fixed control expression are still paired.
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let one = if matches!(dtype, "f32" | "f64") {
            format!("1.0{dtype}")
        } else {
            format!("cast(1.0, {dtype})")
        };
        let source = format!(
            "def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))\n\
             def main() = keep(key_from_seed(7i64), to_tensor([{one}, {one}, {one}, {one}]))"
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
        assert_eq!(
            tensor(&result, "main"),
            mask_at_width(key7(), dtype == "f64")[..4]
        );
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
fn generic_static_rate_gradient_replays_its_keys_mask_beside_the_following_draw() {
    // [05-OP-37]: source AD of the generic helper reuses the forward mask of
    // the key it is given, and the following draw is its own key's.
    for dtype in ["f16", "bf16", "f32", "f64"] {
        let source = format!(
            "def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))\n\
             def loss(k: key, x: tensor[4, {dtype}]) -> {dtype} = tensor_to_scalar(sum(keep(k, x), 0i32))\n\
             def main() = {{\n {THREE_KEYS}\n\
               x = to_tensor([cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype}), cast(1.0, {dtype})])\n\
               first = keep(k1, copy(x))\n backward = grad(loss, wrt=x)(k2, copy(x))\n next = keep(k3, copy(x))\n\
               (first, backward, next, x)\n }}"
        );
        let prepared = prepare_eval(request(&source)).unwrap();
        let (k1, k2, k3) = three_keys();
        for result in [
            eval_selected(request(&source), &["main".into()]),
            prepared.eval_root(BTreeMap::new(), "main"),
            prepared.eval_root(BTreeMap::new(), "main"),
        ] {
            let result = result.unwrap_or_else(|error| panic!("{dtype}: {error:?}"));
            for (index, key) in [k1, k2, k3].into_iter().enumerate() {
                let name = format!("main.{index}");
                assert_eq!(
                    tensor(&result, &name),
                    mask_at_width(key, dtype == "f64")[..4]
                );
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
    let source = format!(
        "def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))\n\
        def main() = {{\n {FOUR_KEYS}\n\
          x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
          y = to_tensor([1.0f64, 1.0f64, 1.0f64, 1.0f64])\n\
          a = keep(k1, copy(x))\n b = keep(k2, y)\n c = keep(k3, copy(x))\n\
          next = dropout(k4, x, 0.5f32)\n\
          (a, b, c, next)\n }}"
    );
    let result = eval_selected(request(&source), &["main".into()]).unwrap();
    let (k1, k2, k3, k4) = four_keys();
    for (index, key, dtype) in [
        (0, k1, "f32"),
        (1, k2, "f64"),
        (2, k3, "f32"),
        (3, k4, "f32"),
    ] {
        let name = format!("main.{index}");
        assert_eq!(
            tensor(&result, &name),
            mask_at_width(key, dtype == "f64")[..4]
        );
        assert_tensor_dtype_shape(&result, &name, dtype);
    }
}

#[test]
fn a_local_callable_still_shadows_the_generic_dropout_definition() {
    let source = "def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))\n\
        def main() = {\n\
          keep = fn (v: tensor[4, f32]) -> v\n\
          x = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
          unchanged = keep(copy(x))\n next = dropout(key_from_seed(7i64), x, 0.5f32)\n\
          (unchanged, next)\n }";
    let result = eval_selected(request(source), &["main".into()]).unwrap();
    assert_eq!(tensor(&result, "main.0"), [1.0; 4]);
    assert_eq!(tensor(&result, "main.1"), mask(key7())[..4]);
}

// chelis#2411: a generic runtime rate draws at its own dtype's arithmetic
// width.
#[test]
fn generic_runtime_rate_draws_at_its_dtypes_width() {
    for dtype in ["f32", "f64"] {
        for definition in [
            "def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(tensor_to_scalar(scalar_to_tensor(0.5f32)), p))".to_string(),
            format!("def keep[p: Float](k: key, x: tensor[4, p], rate: p) -> tensor[4, p] = dropout(k, x, rate)\ndef run(k: key, x: tensor[4, {dtype}]) -> tensor[4, {dtype}] = keep(k, x, tensor_to_scalar(scalar_to_tensor(0.5{dtype})))"),
            format!("def keep[p: Float](k: key, x: tensor[4, p], rate: p) -> tensor[4, p] = dropout(k, x, rate)\ndef run(k: key, x: tensor[4, {dtype}]) -> tensor[4, {dtype}] = {{ rate = tensor_to_scalar(scalar_to_tensor(0.5{dtype}))\n keep(k, x, rate) }}"),
        ] {
            let callee = if definition.contains("def run") { "run" } else { "keep" };
            let source = format!("{definition}\ndef main() = {callee}(key_from_seed(7i64), to_tensor([1.0{dtype}, 1.0{dtype}, 1.0{dtype}, 1.0{dtype}]))");
            let result = eval_selected(request(&source), &["main".into()])
                .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
            assert_eq!(
                tensor(&result, "main"),
                mask_at_width(key7(), dtype == "f64")[..4],
                "{source}"
            );
        }
    }
}

/// The generic helper's specialization evaluates its argument once. A draw
/// is no longer an effect, so the argument also prints, and the transcript
/// is the witness: a second evaluation would print twice. The value root
/// `out` runs the printing `main`; an effectful zero-argument definition is
/// not itself surfaced as a root.
#[test]
fn generic_static_rate_effecting_argument_is_evaluated_once() {
    for dtype in ["f32", "f64"] {
        let source = format!(
            "def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))\n\
            def main() = {{\n {THREE_KEYS}\n\
              x = to_tensor([1.0{dtype}, 1.0{dtype}, 1.0{dtype}, 1.0{dtype}])\n\
              first = keep(k2, {{ _ = print(\"argument\")\n dropout(k1, x, 0.0{dtype}) }})\n next = keep(k3, x)\n\
              (first, next)\n }}\nout = main()\n"
        );
        let result = eval_selected(request(&source), &["out".into()])
            .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
        let (_, k2, k3) = three_keys();
        assert_eq!(
            tensor(&result, "out.0"),
            mask_at_width(k2, dtype == "f64")[..4]
        );
        assert_eq!(
            tensor(&result, "out.1"),
            mask_at_width(k3, dtype == "f64")[..4]
        );
        assert_eq!(result.transcript, ["argument"]);
    }
}

#[test]
fn static_rate_example_includes_generic_evaluator_dispatch() {
    // The example draws with `k1`, then `k2` and `k3` from splitting the
    // remainder of `key_from_seed(42i64)`.
    let source = include_str!("../../../examples/dropout_static_rate.ch");
    let result = eval_selected(request(source), &["result".into()]).unwrap();
    let (k1, k2, k3) = three_keys();
    for (index, key) in [k1, k2, k3].into_iter().enumerate() {
        assert_eq!(tensor(&result, &format!("result.{index}")), mask(key)[..4]);
    }
    assert_eq!(tensor(&result, "result.3"), [1.0; 4]);
}

fn generic_scalar_data_argument_source(copy_middle: bool) -> String {
    let middle_input = if copy_middle { "copy(x)" } else { "x" };
    format!(
        "def keep[p: Float](k: key, x: tensor[4, p], extra: p) -> tensor[4, p] = mul(dropout(k, x, cast(0.5, p)), insert(scalar_to_tensor(extra), 0i32, 4i64))\n\
         def main() = {{\n {FOUR_KEYS}\n\
           x = to_tensor([1.0f64, 1.0f64, 1.0f64, 1.0f64])\n\
           y = to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32])\n\
           first = keep(k1, copy(y), 1.0f32)\n\
           middle = keep(k3, {middle_input}, tensor_to_scalar({{ _ = print(\"extra\")\n dropout(k2, scalar_to_tensor(1.0f64), 0.0f64) }}))\n\
           next = keep(k4, y, 1.0f32)\n (first, middle, next)\n }}\nout = main()\n"
    )
}

/// The stored words of `mask` at `dtype`, as the execution schema spells them.
fn stored_bits(mask: &[f64], dtype: &str) -> Vec<String> {
    mask.iter()
        .map(|value| match dtype {
            "f32" => format!("{:08x}", (*value as f32).to_bits()),
            "f64" => format!("{:016x}", value.to_bits()),
            other => panic!("{other}"),
        })
        .collect()
}

fn check_generic_scalar_data_argument_draws(copy_middle: bool) {
    let source = generic_scalar_data_argument_source(copy_middle);
    let prepared = prepare_eval(request(&source)).unwrap();
    // The rate-zero scalar actual draws with `k2` once, before the middle
    // helper call, which draws with `k3`; its print is the witness that it
    // runs once (the value root `out` runs the printing `main`). Full stored
    // words make every coordinate, dtype and zero sign observable.
    let (k1, _, k3, k4) = four_keys();
    let expected = serde_json::json!([
        {"shape":[4], "data":{"dtype":"f32", "bits":stored_bits(&mask(k1)[..4], "f32")}},
        {"shape":[4], "data":{"dtype":"f64", "bits":stored_bits(&mask_at_width(k3, true)[..4], "f64")}},
        {"shape":[4], "data":{"dtype":"f32", "bits":stored_bits(&mask(k4)[..4], "f32")}}
    ]);
    // Run all three entry invocations before asserting, so a red regression
    // records both immediate and reused-preparation behavior.
    let actual: Vec<_> = [
        eval_selected(request(&source), &["out".into()]),
        prepared.eval_root(BTreeMap::new(), "out"),
        prepared.eval_root(BTreeMap::new(), "out"),
    ]
    .into_iter()
    .map(|result| {
        let result = result.unwrap();
        assert_eq!(result.transcript, ["extra"]);
        (0..3)
            .map(|index| {
                let name = format!("out.{index}");
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
    check_generic_scalar_data_argument_draws(false);
}

#[test]
fn generic_scalar_data_argument_copy_control_keeps_the_same_draws() {
    check_generic_scalar_data_argument_draws(true);
}

#[test]
fn fixed_dropout_composes_with_host_produced_checked_reshape_targets() {
    use chelis_compiler_api::compiler::{eval_in_context, prepare_eval_in_context};
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
    let mut failures = Vec::new();
    let (_, _, k3) = three_keys();
    for target in [
        "numel(source)",
        "len(to_list(source))",
        "bitand(shape(source, 0i32), 3i64)",
        "numel(dropout(kt, source, 0.0f32))",
    ] {
        for dropout in [false, true] {
            let body = if dropout {
                format!(
                    "{{\n dead = dropout(kd, x, 0.0f32)\n _ = drop(dead)\n dropout(kf, reshape(x, [{target}, 2i64]), 0.5f32)\n}}"
                )
            } else {
                format!("reshape(x, [{target}, 2i64])")
            };
            let definition = format!(
                "def loss[m, n](kt: key, kd: key, kf: key, source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {body}\n"
            );
            let directory = tempfile::tempdir().unwrap();
            std::fs::create_dir(directory.path().join("src")).unwrap();
            std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"review-three-guard\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
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
                    "def main() = {{\n {THREE_KEYS}\n loss(k1, k2, k3, to_tensor([{source_values}]), to_tensor([{x_values}]))\n}}"
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
                        let result = match result {
                            Ok(result) => result,
                            Err(error) => {
                                failures
                                    .push(format!("{target} dropout={dropout} {lane}: {error:?}"));
                                continue;
                            }
                        };
                        let ExecutionValue::Tensor { value } = &result.roots[0].value else {
                            panic!("{result:?}")
                        };
                        assert_eq!(value.shape, vec![2, 2]);
                        assert_eq!(
                            value.data.to_f64_lossy_vec(),
                            if dropout {
                                mask(k3)[..4].to_vec()
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

/// A library helper's draws take the key each caller passes. Its source plan
/// is memoized per context, so a plan that baked the first caller's key
/// would give the second caller the first caller's mask.
#[test]
fn host_only_random_source_does_not_cache_the_first_callers_key() {
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

    /// `(ks, rest) = split_key(key_from_seed(seed))`, then
    /// `(kx, knext) = split_key(rest)`: the helper's draw and the next one.
    fn keys(seed: i64) -> (u64, u64) {
        let (_, rest) = key_reference::split(key_reference::key_from_seed(seed));
        key_reference::split(rest)
    }

    fn check(result: EvalResult, first: i64, second: i64, lane: &str) {
        let (first_draw, first_next) = keys(first);
        let (second_draw, second_next) = keys(second);
        for (name, expected) in [
            ("main.0.0", mask(first_draw)[..4].to_vec()),
            ("main.0.1", mask(first_next)[..4].to_vec()),
            ("main.1.0", mask(second_draw)[..4].to_vec()),
            ("main.1.1", mask(second_next)[..4].to_vec()),
        ] {
            assert_eq!(
                tensor(&result, name),
                expected,
                "{lane} seeds={first},{second} root={name}"
            );
        }
    }

    let definition = "def draw[m, n](ks: key, kx: key, source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = dropout(kx, reshape(x, [numel(dropout(ks, source, 0.0f32)), 2i64]), 0.5f32)";
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("reef.toml"),
        format!(
            "[package]\nname = \"round-four-seed\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
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
    assert_ne!(mask(keys(42).0)[..4], mask(keys(7).0)[..4]);
    assert_ne!(mask(keys(42).1)[..4], mask(keys(7).1)[..4]);

    for (first, second) in [(42, 42), (42, 7), (7, 42)] {
        let main = format!(
            "def main() = {{\n source = {}\n x = {}\n a = {{\n (ks, rest) = split_key(key_from_seed({first}i64))\n (kx, knext) = split_key(rest)\n shaped = draw(ks, kx, copy(source), copy(x))\n (shaped, dropout(knext, copy(x), 0.5f32))\n }}\n b = {{\n (ks, rest) = split_key(key_from_seed({second}i64))\n (kx, knext) = split_key(rest)\n shaped = draw(ks, kx, source, copy(x))\n (shaped, dropout(knext, x, 0.5f32))\n }}\n (a, b)\n}}",
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
    let source = format!(
        "def loss[a, b](k: key, x: tensor[a, b, f32]) -> tensor[f32] = sum(sum(dropout(k, x, 0.5f32), 0i32), 0i32)\ndef checked[m, n](k1: key, k2: key, k3: key, source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {{\n first = dropout(k1, source, 0.0f32)\n shaped = reshape(x, [numel(first), 2i64])\n gradient = grad(loss, wrt=x)(k2, shaped)\n dropout(k3, reshape(gradient, [len(to_list(source)), 2i64]), 0.5f32)\n}}\ndef sample[m, n](source: tensor[m, f32], x: tensor[n, f32]) = {{\n {FOUR_KEYS}\n result = checked(k1, k2, k3, source, copy(x))\n (result, dropout(k4, x, 0.5f32))\n}}"
    );
    let prepared = prepare_eval(EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.clone(),
        bindings: BTreeMap::new(),
    })
    .unwrap();
    let (_, k2, k3, k4) = four_keys();
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
                    source: source.clone(),
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
                    mask(k2)
                        .iter()
                        .zip(mask(k3))
                        .take(4)
                        .map(|(a, b)| a * b)
                        .collect::<Vec<_>>()
                );
                assert_eq!(tensor(&result, "sample.1"), mask(k4)[..4]);
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

/// The differentiated function makes its own key, as it once entered its
/// own seed handler: the key stays local to the function and its replay,
/// and the host cut that follows draws with the caller's keys.
#[test]
fn staged_ad_local_key_source_stays_local_before_the_following_host_cut() {
    let source = format!(
        "def loss[a, b](x: tensor[a, b, f32]) -> tensor[f32] = sum(sum(dropout(key_from_seed(7i64), x, 0.5f32), 0i32), 0i32)\ndef checked[m, n](k1: key, k2: key, source: tensor[m, f32], x: tensor[n, f32]) -> tensor[2, 2, f32] = {{\n first = dropout(k1, source, 0.0f32)\n shaped = reshape(x, [numel(first), 2i64])\n gradient = grad(loss)(shaped)\n dropout(k2, reshape(gradient, [len(to_list(source)), 2i64]), 0.5f32)\n}}\ndef sample() = {{\n {THREE_KEYS}\n source = to_tensor(SOURCE)\n x = to_tensor(VALUES)\n result = checked(k1, k2, source, copy(x))\n (result, dropout(k3, x, 0.5f32))\n}}"
    );
    let (_, k2, k3) = three_keys();
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
                    mask(key7())
                        .iter()
                        .zip(mask(k2))
                        .take(4)
                        .map(|(a, b)| a * b)
                        .collect::<Vec<_>>()
                );
                assert_eq!(tensor(&result, "sample.1"), mask(k3)[..4]);
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
    let loss = "def loss[n](k: key, x: tensor[n, f32]) -> tensor[f32] = sum(sum(dropout(k, checked(x), 0.5f32), 0), 0)\n";
    let (_, k2) = two_keys();
    for gradient in [false, true] {
        let body = if gradient {
            "grad(loss, wrt=x)(k2, x)"
        } else {
            "dropout(k2, checked(x), 0.5f32)"
        };
        let source = format!(
            "{helper}{loss}def sample[n](x: tensor[n, f32]) = {{\n {TWO_KEYS}\n dead = dropout(k1, x, 0.0f32)\n _ = drop(dead)\n {body}\n}}\n"
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
                    assert_eq!(tensor(&result.unwrap(), "sample"), mask(k2));
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
    let (_, k2) = two_keys();
    for gradient in [false, true] {
        let result = if gradient {
            "grad(loss, wrt=b)(k2, b)"
        } else {
            "draw(k2, b)"
        };
        let source = format!(
            "def draw(k: key, b: tensor[unit, f32]) -> tensor[32, f32] = dropout(k, expand(b, 0i32, 32i64), 0.5f32)\ndef loss(k: key, b: tensor[unit, f32]) -> tensor[f32] = sum(draw(k, b), 0)\ndef sample(b: tensor[unit, f32]) = {{\n {TWO_KEYS}\n dead = dropout(k1, b, 0.0f32)\n _ = drop(dead)\n {result}\n}}\n"
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
                        vec![mask(k2).iter().sum()]
                    } else {
                        mask(k2)
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
        "[package]\nname = \"extent-dropout-probe\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
    )).unwrap();
    std::fs::write(directory.path().join("src/draw.ch"),
        "module Probe.Draw\nexport (draw, loss)\ndef checked[n](x: tensor[n, f32]) -> tensor[16, 2, f32] = reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])\ndef draw[n](k: key, x: tensor[n, f32]) -> tensor[16, 2, f32] = dropout(k, checked(x), 0.5f32)\ndef loss[n](k: key, x: tensor[n, f32]) -> tensor[f32] = sum(sum(draw(k, x), 0), 0)\n"
    ).unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let wire = context.encode().unwrap();
    let decoded = chelis_compiler_api::context::CompiledContext::decode(&wire).unwrap();
    let (_, k2) = two_keys();
    for context in [&context, &decoded] {
        for gradient in [false, true] {
            for count in [32, 34, 32] {
                let ones = std::iter::repeat_n("1.0f32", count)
                    .collect::<Vec<_>>()
                    .join(", ");
                let body = if gradient {
                    "grad(loss, wrt=x)(k2, x)"
                } else {
                    "draw(k2, x)"
                };
                let source = format!(
                    "module Probe.Eval\nimport Probe.Draw (draw, loss)\ndef main() = {{\n {TWO_KEYS}\n x = to_tensor([{ones}])\n dead = dropout(k1, x, 0.0f32)\n _ = drop(dead)\n {body}\n}}\n"
                );
                let prepared = prepare_eval_in_context(context, &source).unwrap();
                for result in [
                    eval_in_context(context, &source),
                    prepared.eval_root(BTreeMap::new(), "main"),
                    prepared.eval_root(BTreeMap::new(), "main"),
                ] {
                    if count == 32 {
                        assert_eq!(tensor(&result.unwrap(), "main"), mask(k2));
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
                "reshape(dropout(k, x, 0.5f32), [floor_div(shape(x, 0i32), 2i64), 2i64])",
                "reshape",
            ),
            (
                "{\n dead = dropout(k, x, 1.0f32)\n _ = drop(dead)\n reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64])\n}",
                "dropout",
            ),
            (
                "dropout(k, reshape(x, [floor_div(shape(x, 0i32), 2i64), 2i64]), 1.0f32)",
                if count == 32 { "dropout" } else { "reshape" },
            ),
        ] {
            let source = format!(
                "def draw[n](k: key, x: tensor[n, f32]) -> tensor[16, 2, f32] = {body}\ndef sample[n](x: tensor[n, f32]) = draw(key_from_seed(7i64), x)\n"
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
                    assert_eq!(tensor(&result.unwrap(), "sample"), mask(key7()));
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

#[test]
fn selected_tensor_entry_and_reused_preparation_draw_the_keys_mask() {
    let source = "def sample(x: tensor[32, f32]) -> tensor[32, f32] = dropout(key_from_seed(7i64), x, 0.5f32)\n";
    let result = eval_selected(request(source), &["sample".into()]).unwrap();
    assert_eq!(tensor(&result, "sample"), mask(key7()));
    let prepared = prepare_eval(request(source)).unwrap();
    for _ in 0..2 {
        assert_eq!(
            tensor(&prepared.eval_root(bindings(), "sample").unwrap(), "sample"),
            mask(key7())
        );
    }
}

#[test]
fn unselected_invalid_sibling_is_not_an_entered_source_declaration() {
    let source = "def invalid(y: tensor[32, f32]) -> tensor[32, f32] = dropout(key_from_seed(7i64), y, 1.0f32)\ndef sample(x: tensor[32, f32]) -> tensor[32, f32] = dropout(key_from_seed(7i64), x, 0.5f32)\n";
    assert_eq!(
        tensor(
            &eval_selected(request(source), &["sample".into()]).unwrap(),
            "sample"
        ),
        mask(key7())
    );
}

/// The discarded draw consumes its own key, and the live draw's mask is
/// its own key's (the counter stream's version asserted that the discarded
/// draw advanced the ordinal the live draw read).
#[test]
fn discarded_forward_draw_consumes_its_own_key_not_the_live_draws() {
    let source = format!(
        "def sample(x: tensor[32, f32]) -> tensor[32, f32] = {{\n {TWO_KEYS}\n dead = dropout(k1, x, 0.0f32)\n dropout(k2, x, 0.5f32)\n}}\n"
    );
    let (k1, k2) = two_keys();
    assert_eq!(
        tensor(
            &eval_selected(request(&source), &["sample".into()]).unwrap(),
            "sample"
        ),
        mask(k2)
    );
    assert_ne!(mask(k1), mask(k2), "the key control must discriminate");
}

#[test]
fn dynamic_helper_applications_get_fresh_forward_keys() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def draw(k: key, x: tensor[32, f32]) -> tensor[32, f32] = dropout(k, x, 0.5f32)\ndef main() = {{\n {TWO_KEYS}\n x = to_tensor([{ones}])\n dead = draw(k1, x)\n draw(k2, x)\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()]).unwrap();
    let (k1, k2) = two_keys();
    assert_eq!(tensor(&result, "main"), mask(k2));
    assert_ne!(mask(k1), mask(k2), "a reused first key must be visible");
}

#[test]
fn actual_input_gradient_uses_the_forward_mask_of_its_key() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def loss(k: key, x: tensor[32, f32]) -> tensor[f32] = sum(dropout(k, x, 0.5f32), 0)\ndef draw(k: key, x: tensor[32, f32]) -> tensor[32, f32] = dropout(k, x, 0.5f32)\ndef main() = {{\n {TWO_KEYS}\n x = to_tensor([{ones}])\n gradient = grad(loss, wrt=x)(k1, x)\n next = draw(k2, x)\n (gradient, next)\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()]).unwrap();
    let (k1, k2) = two_keys();
    assert_eq!(tensor(&result, "main.0"), mask(k1));
    assert_eq!(tensor(&result, "main.1"), mask(k2));
}

#[test]
fn scalar_cotangent_repacking_keeps_dropout_replay_and_next_draw() {
    let (k1, k2) = two_keys();
    for dtype in ["f32", "f64"] {
        let ones = std::iter::repeat_n(format!("1.0{dtype}"), 32)
            .collect::<Vec<_>>()
            .join(", ");
        let source = format!(
            "def loss(k: key, x: {dtype}) -> {dtype} = tensor_to_scalar(dropout(k, scalar_to_tensor(x), 0.5{dtype}))\ndef main() = {{\n {TWO_KEYS}\n derivative = grad(loss, wrt=x)(k1, 1.0{dtype})\n next = dropout(k2, to_tensor([{ones}]), 0.5{dtype})\n (derivative, next)\n}}\n"
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
        assert_eq!(
            value.get().as_f64_lossy(),
            mask_at_width(k1, dtype == "f64")[0]
        );
        assert_eq!(tensor(&result, "main.1"), mask_at_width(k2, dtype == "f64"));
    }
}

#[test]
fn empty_cotangent_repacking_still_executes_retained_dropout() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let (_, k2) = two_keys();
    for rate in ["0.0f32", "1.0f32"] {
        let source = format!(
            "def loss(k: key, xs: List[f32]) -> f32 = {{\n dead = dropout(k, to_tensor([{ones}]), {rate})\n _ = drop(dead)\n 0.0f32\n}}\ndef main() = {{\n {TWO_KEYS}\n _ = grad(loss, wrt=xs)(k1, [])\n dropout(k2, to_tensor([{ones}]), 0.5f32)\n}}\n"
        );
        let result = eval_selected(request(&source), &["main".into()]);
        if rate == "0.0f32" {
            assert_eq!(tensor(&result.unwrap(), "main"), mask(k2));
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

/// A warmed library context's source plan carries no realized key: callers
/// with other seeds, and a caller drawing with the other half, each get
/// their own key's mask, and warming never changes the cache bytes.
#[test]
fn checked_library_context_and_prepared_context_preserve_raw_key_binding() {
    use chelis_compiler_api::compiler::{eval_in_context, prepare_eval_in_context};
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!(
        "[package]\nname = \"dropout-probe\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
    )).unwrap();
    std::fs::write(directory.path().join("src/draw.ch"),
        "module Probe.Draw\nexport (draw)\ndef draw(k: key, x: tensor[32, f32]) -> tensor[32, f32] = dropout(k, x, 0.5f32)\n"
    ).unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let cold_wire = context.encode().unwrap();
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "module Probe.Eval\nimport Probe.Draw (draw)\ndef main() = {{\n x = to_tensor([{ones}])\n (k1, k2) = split_key(key_from_seed(42i64))\n dead = draw(k1, x)\n draw(k2, x)\n}}\n"
    );
    let (_, k2) = two_keys();
    assert_eq!(
        tensor(&eval_in_context(&context, &source).unwrap(), "main"),
        mask(k2)
    );
    let prepared = prepare_eval_in_context(&context, &source).unwrap();
    for _ in 0..2 {
        assert_eq!(
            tensor(
                &prepared.eval_root(BTreeMap::new(), "main").unwrap(),
                "main"
            ),
            mask(k2)
        );
    }
    assert_eq!(
        context.encode().unwrap(),
        cold_wire,
        "warming the private source-plan memo must not alter cache bytes"
    );
    let decoded = chelis_compiler_api::context::CompiledContext::decode(&cold_wire).unwrap();
    for context in [&context, &decoded] {
        for (seed, right_half) in [(42, true), (7, false), (42, false), (7, true)] {
            let mut source = source.replace("seed(42i64)", &format!("seed({seed}i64)"));
            if !right_half {
                source = source.replace("dead = draw(k1, x)\n draw(k2, x)", "draw(k1, x)");
            }
            let (left, right) = key_reference::split(key_reference::key_from_seed(seed));
            let prepared = prepare_eval_in_context(context, &source).unwrap();
            for _ in 0..2 {
                assert_eq!(
                    tensor(
                        &prepared.eval_root(BTreeMap::new(), "main").unwrap(),
                        "main"
                    ),
                    mask(if right_half { right } else { left })
                );
            }
        }
        assert_eq!(
            context.encode().unwrap(),
            cold_wire,
            "the source-derived memo carries no invocation key"
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
    let main = format!("def main() = {{\n x = to_tensor([{ones}])\n {body}\n}}\n");
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
        "[package]\nname = \"dropout-drop-probe\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
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

/// The library definition makes its own keys, as it once entered its own
/// seed handler.
#[test]
fn explicit_library_drop_preserves_dead_forward_and_context_entry_remapping() {
    assert_explicit_drop_context_parity(
        &format!(
            "def draw(x: tensor[32, f32]) -> tensor[32, f32] = {{\n {TWO_KEYS}\n dropped = dropout(k1, x, 0.0f32)\n _ = drop(dropped)\n dropout(k2, x, 0.5f32)\n}}"
        ),
        "draw(x)",
        &[("main", mask(two_keys().1))],
    );
}

#[test]
fn explicit_library_drop_preserves_gradient_replay_and_caller_input() {
    let definition = "def draw(kd: key, kl: key, x: tensor[32, f32]) -> tensor[f32] = {\n dropped = dropout(kd, x, 0.0f32)\n _ = drop(dropped)\n sum(dropout(kl, x, 0.5f32), 0)\n}";
    let (_, k2, k3) = three_keys();
    assert_explicit_drop_context_parity(
        definition,
        &format!(
            "{THREE_KEYS}\n g = grad(draw, wrt=x)(k1, k2, x)\n (g, dropout(k3, x, 0.5f32), x)"
        ),
        &[
            ("main.0", mask(k2)),
            ("main.1", mask(k3)),
            ("main.2", vec![1.0; 32]),
        ],
    );
    assert_explicit_drop_context_parity(
        definition,
        &format!(
            "{THREE_KEYS}\n g = grad(draw, wrt=x)(k1, k2, x)\n _ = drop(g)\n dropout(k3, x, 0.5f32)"
        ),
        &[("main", mask(k3))],
    );
}

#[test]
fn fixed_primitive_in_a_host_tuple_uses_the_same_plan_core() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def main() = {{\n x = to_tensor([{ones}])\n (dropout(key_from_seed(7i64), x, 0.5f32), 7i64)\n}}\n"
    );
    assert_eq!(
        tensor(
            &eval_selected(request(&source), &["main".into()]).unwrap(),
            "main.0"
        ),
        mask(key7())
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
    let source = format!(
        "def run() -> tensor[1, f32] = {{\n _ = print(\"before\")\n x = to_tensor([1.0f32])\n {TWO_KEYS}\n dead = dropout(k1, x, 0.0f32)\n value = dropout(k2, x, 1.0f32)\n _ = print(\"after\")\n value\n}}\nout = run()\n"
    );
    let error = eval_selected(request(&source), &["out".into()]).unwrap_err();
    assert_eq!(error.transcript, ["before"], "{error:?}");
    assert_eq!(error.errors.len(), 1, "{error:?}");
    assert_eq!(
        error.errors[0].message,
        "numeric trap: domain in dropout at f32"
    );
}

#[test]
fn runtime_rate_and_dynamic_control_both_draw_their_keys_mask() {
    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def excluded(k: key, x: tensor[32, f32], rate: f32) = (dropout(k, x, rate), 7i64)\ndef main() = excluded(key_from_seed(7i64), to_tensor([{ones}]), 0.5f32)\n"
    );
    // [05-OP-37]: a runtime rate is an ordinary operand (chelis#2411).
    let result = eval_selected(request(&source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    assert_eq!(tensor(&result, "main.0"), mask(key7()));
    // chelis#2405: a dynamic caller does not exclude its fixed-rate helper,
    // which runs its own plan.
    let source = format!(
        "def draw(k: key, x: tensor[32, f32]) -> tensor[32, f32] = dropout(k, x, 0.5f32)\ndef dynamic(k: key, x: tensor[32, f32], condition: bool) = if condition then (draw(k, x), 7i64) else (x, 7i64)\ndef main() = dynamic(key_from_seed(7i64), to_tensor([{ones}]), true)\n"
    );
    let result = eval_selected(request(&source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    assert_eq!(tensor(&result, "main.0"), mask(key7()));
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
                "def sample(x: tensor[{count}, {prim}]) -> tensor[{count}, {prim}] = dropout(key_from_seed(7i64), x, {rate}{prim})\n"
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
                    if count == 0 {
                        vec![]
                    } else {
                        mask_at_width(key7(), prim == "f64")
                    }
                );
            }
        }
    }
}

#[test]
fn mismatch_nonfloat_and_alias_admission_follow_the_shared_signature() {
    use chelis_compiler_api::compiler::check;
    use chelis_compiler_api::schema::CheckRequest;
    let errors = |input: &str, rate: &str| {
        check(CheckRequest {
            source_kind: SourceKind::Surf,
            source: format!(
                "def sample(x: tensor[4, {input}]) = dropout(key_from_seed(7i64), x, {rate})\n"
            ),
        })
        .unwrap()
        .errors
    };
    // Each rejected pair has an accepted twin that differs only in the
    // mismatched operand, so a parse or key error cannot satisfy it.
    for (input, rate, twin) in [
        ("f64", "0.5f32", ("f64", "0.5f64")),
        ("f16", "0.5bf16", ("f16", "0.5f16")),
        ("i32", "0i32", ("f32", "0.0f32")),
    ] {
        assert!(!errors(input, rate).is_empty(), "{input}/{rate}");
        let accepted = errors(twin.0, twin.1);
        assert!(accepted.is_empty(), "{twin:?}: {accepted:?}");
    }
    assert!(check(CheckRequest {source_kind:SourceKind::Surf,source:"type Values = tensor[4, f64]\ndef sample(x: Values) -> Values = dropout(key_from_seed(7i64), x, 0.5f64)\n".into()}).unwrap().errors.is_empty());
}

#[test]
fn dead_draw_input_is_not_required_when_not_data_live_at_the_selected_root() {
    let source = "def sample(x: tensor[32, f32], y: tensor[32, f32]) -> tensor[32, f32] = {\n dead = dropout(key_from_seed(7i64), y, 0.0f32)\n x\n}\n";
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
    // The discarded draw still consumes its key; its data is never read, so
    // the caller need not supply it.
    assert!(!entry.required_inputs.iter().any(|name| name == "y"));
}

/// A parameter is its declaration and its name (chelis#2413 B2): two defs
/// that each name their key parameter `k` take two keys. (a) Selecting one
/// root of such a module evaluates it with its own key, and `lower` of the
/// module keeps the two `k`s apart, each attributed to its declaration on the
/// wire.
///
/// Evidentiary status: REGRESSION TEST for `lower`, which rejected the module
/// at ff8957386 ("key 0 is consumed twice") because the whole-program graph
/// identified a key `Load` by its name alone. The `eval_selected` row is a
/// disposition lock: it already evaluated at ff8957386.
#[test]
fn two_declarations_key_parameters_of_one_name_are_two_keys() {
    use chelis_compiler_api::schema::{LowerRequest, WireRiscOp};
    let source = "def sample(k: key, v: tensor[32, f32]) -> tensor[32, f32] = dropout(k, v, 0.5f32)\ndef other(k: key, v: tensor[32, f32]) -> tensor[32, f32] = dropout(k, v, 0.25f32)\ndef main(x: tensor[32, f32]) -> tensor[32, f32] = sample(key_from_seed(7i64), x)\n";
    let result = eval_selected(request(source), &["main".into()]).unwrap();
    assert_eq!(tensor(&result, "main"), mask(key7()));
    let lowered = chelis_compiler_api::compiler::lower(LowerRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        entry: None,
    })
    .unwrap();
    let key_loads = lowered
        .dag
        .nodes
        .iter()
        .filter(|node| matches!(&node.op, WireRiscOp::Load { name } if name.as_str() == "k"))
        .map(|node| lowered.dag.declarations[usize::try_from(node.declaration).unwrap()].as_str())
        .collect::<Vec<_>>();
    assert_eq!(key_loads, ["sample", "other"]);
}

#[test]
fn nonunit_cotangent_matches_same_key_finite_differences() {
    let weights = std::iter::repeat_n("3.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let source = format!(
        "def loss(k: key, x: tensor[32, f32]) -> tensor[f32] = sum(mul(dropout(k, x, 0.5f32), to_tensor([{weights}])), 0)\ndef value(x: tensor[32, f32]) -> tensor[f32] = loss(key_from_seed(7i64), x)\ndef derivative(x: tensor[32, f32]) -> tensor[32, f32] = grad(loss, wrt=x)(key_from_seed(7i64), x)\n"
    );
    let prepared = prepare_eval(request(&source)).unwrap();
    let gradient = tensor(
        &prepared.eval_root(bindings(), "derivative").unwrap(),
        "derivative",
    );
    assert_eq!(
        gradient,
        mask(key7())
            .iter()
            .map(|value| 3.0 * value)
            .collect::<Vec<_>>()
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
/// result. The rows are a draw keyed in the selected root itself (as a
/// value root and as a function root), a discarded inner block that makes
/// its own key, a discarded call of a helper that makes its own key, and a
/// declaration the root references only through a dead binding, both as an
/// inlined function and as a value declaration. The uniform row is
/// [05-OP-8]'s bound check.
///
/// Evidentiary status: REGRESSION TEST. At 3b5f029d8 every row returned the
/// root's value, because a region with no live draw was dropped as unreached.
#[test]
fn a_selected_activations_discarded_draws_still_validate_and_trap() {
    let rows = [
        (
            "own key, value root",
            "x: tensor[32, f32] = x\nselected = {\n dead = dropout(key_from_seed(7i64), copy(x), 1.0f32)\n copy(x)\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "own key, function root",
            "def selected(x: tensor[32, f32]) -> tensor[32, f32] = {\n dead = dropout(key_from_seed(7i64), copy(x), 1.0f32)\n x\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "discarded inner block with its own key",
            "x: tensor[32, f32] = x\nselected = {\n dead = { k = key_from_seed(9i64)\n dropout(k, copy(x), 1.0f32) }\n dropout(key_from_seed(7i64), copy(x), 0.5f32)\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "discarded helper with its own key",
            "def h(v: tensor[32, f32]) -> tensor[32, f32] = {\n dead = dropout(key_from_seed(9i64), copy(v), 1.0f32)\n v\n}\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = {\n a = h(copy(x))\n dropout(key_from_seed(7i64), x, 0.5f32)\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "dead reference to a function declaration",
            "def sampled(x: tensor[32, f32]) -> tensor[32, f32] = dropout(key_from_seed(7i64), x, 1.0f32)\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = {\n dead = sampled(copy(x))\n x\n}\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "dead reference to a value declaration",
            "x: tensor[32, f32] = x\nsampled = dropout(key_from_seed(7i64), x, 1.0f32)\nselected = {\n dead = sampled\n copy(x)\n}\nunrelated = dropout(key_from_seed(8i64), x, 0.5f32)\n",
            "numeric trap: domain in dropout at f32",
        ),
        (
            "discarded uniform with reversed bounds",
            "x: tensor[32, f32] = x\nselected = {\n dead = uniform_like(key_from_seed(7i64), copy(x), 1.0f32, 0.0f32)\n copy(x)\n}\n",
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
/// declaration's activation, so its invalid rate does not trap.
#[test]
fn an_unselected_declarations_invalid_draw_does_not_run() {
    let source = "x: tensor[32, f32] = x\nselected = dropout(key_from_seed(7i64), copy(x), 0.5f32)\nunrelated = dropout(key_from_seed(7i64), x, 1.0f32)\n";
    let result = eval_selected(request(source), &["selected".into()]).unwrap();
    assert_eq!(tensor(&result, "selected"), mask(key7()));
    let source = format!(
        "def unrelated(y: tensor[32, f32]) -> tensor[32, f32] = dropout(key_from_seed(7i64), y, 1.0f32)\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = {{\n {TWO_KEYS}\n dead = dropout(k1, copy(x), 0.5f32)\n dropout(k2, x, 0.5f32)\n}}\n"
    );
    let result = eval_selected(request(&source), &["selected".into()]).unwrap();
    assert_eq!(tensor(&result, "selected"), mask(two_keys().1));
}

/// `sampled`, a value declaration whose draw has an invalid rate.
const SAMPLED: &str = "sampled = dropout(key_from_seed(9i64), scalar_to_tensor(1.0f32), 1.0f32)\n";

/// [`SAMPLED`] and `f`, a function whose body names `sampled` in a dead
/// binding.
const NAMES_SAMPLED: &str = "sampled = dropout(key_from_seed(9i64), scalar_to_tensor(1.0f32), 1.0f32)\ndef f(v: tensor[32, f32]) -> tensor[32, f32] = {\n  dead = sampled\n  v\n}\n";

/// A selected root that binds `g` to `f`: unapplied, or applied to a copy of
/// `argument` when `applied`.
fn binds_f(applied: bool, argument: &str) -> String {
    if applied {
        format!("g = f(copy({argument}))")
    } else {
        "g = f".to_string()
    }
}

/// The DAG evaluator's rows (spec/03 §4.4, spec/06 §5.2 with [05-OP-37]):
/// each program selects `selected`, and traps exactly when `traps`. The
/// first four are the `dead = f` row and its applied twin, as a value root
/// and as a function root. The rest pin how the references are recorded: a
/// value named inside a `grad` or `vmap` body belongs to the declaration the
/// body is spliced into; a value declaration whose value is another's still
/// runs its own initializer when named; and a local binding that shadows a
/// value declaration's name names only itself.
fn dag_evaluator_rows() -> Vec<(&'static str, String, bool)> {
    let input = "x: tensor[32, f32] = x\n";
    let mut rows = Vec::new();
    for applied in [false, true] {
        rows.push((
            if applied {
                "value root, f applied"
            } else {
                "value root, f unapplied"
            },
            format!(
                "{input}{NAMES_SAMPLED}selected = {{\n  {}\n  copy(x)\n}}\n",
                binds_f(applied, "x")
            ),
            applied,
        ));
        rows.push((
            if applied {
                "function root, f applied"
            } else {
                "function root, f unapplied"
            },
            format!(
                "{NAMES_SAMPLED}def selected(x: tensor[32, f32]) -> tensor[32, f32] = {{\n  {}\n  x\n}}\n",
                binds_f(applied, "x")
            ),
            applied,
        ));
    }
    rows.extend([
        (
            "named inside a differentiated body",
            format!(
                "{input}{SAMPLED}def loss(v: tensor[32, f32]) -> tensor[f32] = {{\n  dead = sampled\n  sum(v, 0i32)\n}}\nselected = grad(loss)(copy(x))\n"
            ),
            true,
        ),
        (
            "named inside a mapped body",
            format!(
                "{input}{SAMPLED}def row(v: tensor[f32]) -> tensor[f32] = {{\n  dead = sampled\n  v\n}}\nselected = vmap(row)(copy(x))\n"
            ),
            true,
        ),
        (
            "a value whose value is another's",
            format!(
                "{input}kept = dropout(key_from_seed(1i64), scalar_to_tensor(1.0f32), 0.0f32)\nalias = {{\n  dead = dropout(key_from_seed(8i64), scalar_to_tensor(1.0f32), 1.0f32)\n  kept\n}}\nselected = {{\n  dead = alias\n  copy(x)\n}}\n"
            ),
            true,
        ),
        (
            "a local binding shadowing the value's name",
            format!(
                "{input}{SAMPLED}selected = {{\n  sampled = copy(x)\n  dead = sampled\n  copy(x)\n}}\n"
            ),
            false,
        ),
    ]);
    rows
}

/// A compiled context whose library module exports `f` (its `sampled` is
/// private), with its decoded copy.
fn function_library_contexts() -> [CompiledContext; 2] {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("reef.toml"),
        format!("[package]\nname = \"fnlib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Fnlib\"\n"),
    )
    .unwrap();
    std::fs::write(
        directory.path().join("src/draw.ch"),
        format!("module Fnlib.Draw\nexport (f)\n{NAMES_SAMPLED}"),
    )
    .unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let decoded = CompiledContext::decode(&context.encode().unwrap()).unwrap();
    [context, decoded]
}

/// Compile `client`'s `main(x: tensor[32, f32])` in `context` as a C entry
/// and run it on the driver's `input(32)`, returning `Ok(())` when it
/// returns `x` and the trap text when it aborts.
fn run_in_context_c(context: &CompiledContext, client: &str) -> Result<(), String> {
    let artifact = compile_for_execution_in_context(context, client, CompileTarget::C, None)
        .unwrap_or_else(|error| panic!("{client}: {error:?}"));
    assert_eq!(
        artifact
            .inputs
            .iter()
            .map(|input| input.name.as_str())
            .collect::<Vec<_>>(),
        ["x"]
    );
    let file = |path: &str| {
        artifact
            .compile_result
            .files
            .iter()
            .find(|file| file.path == path)
            .unwrap_or_else(|| panic!("no generated `{path}`"))
            .contents
            .clone()
    };
    let program =
        ownership_support::GeneratedProgram::new(file("chelis_main.c"), file("chelis_main.h"));
    let driver = "int main(void) {\n    chelis_tensor *x = input(32);\n    chelis_tensor *inputs[] = {x};\n    chelis_tensor *outputs[] = {NULL};\n    chelis_main(inputs, 1, outputs, 1);\n    float expected[32];\n    for (int i = 0; i < 32; ++i) expected[i] = (float)(2*i-3);\n    tensor_bits(outputs[0], 32, expected);\n    chelis_tensor_release(outputs[0]);\n    chelis_tensor_release(x);\n    return 0;\n}\n";
    let traps =
        std::panic::catch_unwind(|| ownership_support::run_failure_stderr(&program, driver));
    match traps {
        Ok(stderr) => Err(stderr),
        Err(_) => {
            ownership_support::balanced(&ownership_support::run(&program, driver));
            Ok(())
        }
    }
}

/// `g = f`, a function named as a value and not applied, evaluates to the
/// function and runs nothing: neither `f`'s body nor the value declaration
/// `sampled` that `f` names, so `sampled`'s invalid rate does not trap.
/// Applying `f` inlines its body, which names `sampled`, so the applied twin
/// traps (spec/03 §4.4). Both hold in the DAG evaluator (`eval_selected` of
/// a Tensor-lane root), the host interpreter (a Host-lane `main`), and
/// compiled C (the in-context C entry, run natively, with `f` a library
/// function).
///
/// Evidentiary status: REGRESSION TEST for the unapplied rows in the DAG
/// evaluator and in C, each of which trapped at 441e5c8b2, because the
/// declarations a selection entered were closed over every name a body
/// mentions, applied or not. The host-lane rows, every applied twin, and the
/// `dag_evaluator_rows` after the first four are disposition locks: each
/// held at 441e5c8b2 too. They pin the recording: a mutation that records
/// the declaration of the node a name resolves to fails the alias row and
/// the `grad` and `vmap` rows, one that drops what a sub-context recorded
/// fails the `grad` and `vmap` rows, and one that ignores a shadowing local
/// fails the shadow row.
#[test]
fn a_function_named_as_a_value_and_not_applied_runs_nothing_in_any_lane() {
    for (row, source, traps) in dag_evaluator_rows() {
        let outcome = eval_selected(request(&source), &["selected".into()]);
        if traps {
            assert_domain_trap(outcome, row);
            continue;
        }
        let result = outcome.unwrap_or_else(|error| panic!("{row}: {error:?}"));
        assert_eq!(tensor(&result, "selected"), vec![1.0; 32], "{row}");
        if let Some(entry) = result
            .manifest
            .entries
            .iter()
            .find(|entry| entry.name == "selected")
        {
            assert_eq!(entry.lane, Lane::Tensor, "{row}");
        }
    }

    let ones = std::iter::repeat_n("1.0f32", 32)
        .collect::<Vec<_>>()
        .join(", ");
    let host = |applied: bool| {
        eval_selected(
            EvalRequest {
                source_kind: SourceKind::Surf,
                source: format!(
                    "{NAMES_SAMPLED}def main() -> tensor[32, f32] = {{\n  t = to_tensor([{ones}])\n  {}\n  t\n}}\n",
                    binds_f(applied, "t")
                ),
                bindings: BTreeMap::new(),
            },
            &["main".into()],
        )
    };
    let result = host(false).unwrap_or_else(|error| panic!("host lane: {error:?}"));
    assert_eq!(lane_of(&result, "main"), Lane::Host);
    assert_eq!(tensor(&result, "main"), vec![1.0; 32]);
    assert_domain_trap(host(true), "host lane, f applied");

    for context in &function_library_contexts() {
        let client = |applied: bool| {
            format!(
                "module Fnlib.Client\nimport Fnlib.Draw (f)\ndef main(x: tensor[32, f32]) -> tensor[32, f32] = {{\n  {}\n  x\n}}\n",
                binds_f(applied, "x")
            )
        };
        assert_eq!(run_in_context_c(context, &client(false)), Ok(()));
        let stderr = run_in_context_c(context, &client(true)).expect_err("C, f applied");
        assert!(stderr.contains(DOMAIN_TRAP), "{stderr}");
    }
}

const DOMAIN_TRAP: &str = "numeric trap: domain in dropout at f32";

/// A value declaration whose draw has the rate `rate`.
fn sampled_value(rate: &str) -> String {
    format!("sampled = dropout(key_from_seed(9i64), to_tensor([1.0f32, 1.0f32]), {rate})\n")
}

fn lane_of(result: &EvalResult, name: &str) -> Lane {
    result
        .manifest
        .entries
        .iter()
        .find(|entry| entry.name == name)
        .unwrap_or_else(|| panic!("no manifest entry `{name}`: {:?}", result.manifest))
        .lane
}

/// A root's value, a scalar read as one element.
fn root_values(result: &EvalResult, name: &str) -> Vec<f64> {
    let root = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some(name))
        .unwrap_or_else(|| panic!("no root `{name}`: {:?}", result.roots));
    match &root.value {
        ExecutionValue::Scalar { value } => vec![value.get().as_f64_lossy()],
        ExecutionValue::Tensor { .. } => tensor(result, name),
        other => panic!("{other:?}"),
    }
}

fn assert_domain_trap(outcome: Result<EvalResult, CompilerError>, row: &str) {
    let error = outcome
        .map(|result| format!("{:?}", result.roots))
        .expect_err(row);
    assert!(
        error
            .errors
            .iter()
            .any(|error| error.message == DOMAIN_TRAP),
        "{row}: {error:?}"
    );
}

/// spec/03 §4.4 with spec/06 §5.2 and [05-OP-37]: a binding's initializer is
/// evaluated whether or not the binding is read, so a value declaration that
/// a selected Host-lane declaration names is initialized, and its invalid
/// rate traps, though nothing reads it. The rows are #2463's API witnesses
/// in `eval_selected`: the value named from a function reached only through
/// `grad` (#2463's witness 3, which C and `chelis eval --file` trap on in
/// `issue_2463_key_dead_draw_traps`), from a plain call of that function,
/// from a body the host runs as one kernel, and from a helper that kernel
/// body calls. Each row's valid-rate twin returns the root's value from the
/// Host lane, so the trap is the only difference.
///
/// Evidentiary status: REGRESSION TEST for the `grad`, kernel-body and
/// kernel-helper rows, each of which returned the root's value at
/// 727e74b41. The plain-call row is a disposition lock: the host interpreter
/// already walked its dead binding there.
#[test]
fn a_dead_reference_to_a_value_declaration_initializes_it_in_the_host_lane() {
    let rows = [
        (
            "reached only through grad",
            "def f(v: tensor[4, f32]) -> f32 = {\n  dead = sampled\n  tensor_to_scalar(sum(v, 0i32))\n}\ndef main() -> tensor[4, f32] = grad(f)(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n",
            vec![1.0; 4],
        ),
        (
            "plain call",
            "def f(v: tensor[4, f32]) -> f32 = {\n  dead = sampled\n  tensor_to_scalar(sum(v, 0i32))\n}\ndef main() -> f32 = f(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]))\n",
            vec![10.0],
        ),
        (
            "kernel body",
            "def main() -> tensor[2, f32] = {\n  dead = sampled\n  to_tensor([1.0f32, 1.0f32])\n}\n",
            vec![1.0; 2],
        ),
        (
            "helper of a kernel body",
            "def h(v: tensor[2, f32]) -> tensor[2, f32] = {\n  dead = sampled\n  v\n}\ndef main() -> tensor[2, f32] = h(to_tensor([1.0f32, 1.0f32]))\n",
            vec![1.0; 2],
        ),
    ];
    for (row, body, expected) in rows {
        for inputs in [BTreeMap::new(), bindings()] {
            let evaluate = |rate: &str| {
                eval_selected(
                    EvalRequest {
                        source_kind: SourceKind::Surf,
                        source: format!("{}{body}", sampled_value(rate)),
                        bindings: inputs.clone(),
                    },
                    &["main".into()],
                )
            };
            let control = evaluate("0.5f32").unwrap_or_else(|error| panic!("{row}: {error:?}"));
            assert_eq!(lane_of(&control, "main"), Lane::Host, "{row}");
            assert_eq!(root_values(&control, "main"), expected, "{row}");
            assert_domain_trap(evaluate("1.0f32"), row);
        }
    }
}

/// A compiled context whose library module `Drawlib.Draw` exports `sampled`
/// (an invalid rate) and `kept` (a valid one), with its decoded copy.
fn draw_library_contexts() -> [CompiledContext; 2] {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("reef.toml"),
        format!("[package]\nname = \"drawlib\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Drawlib\"\n"),
    )
    .unwrap();
    std::fs::write(
        directory.path().join("src/draw.ch"),
        format!(
            "module Drawlib.Draw\nexport (sampled, kept)\n{}kept = dropout(key_from_seed(1i64), to_tensor([1.0f32, 1.0f32]), 0.0f32)\n",
            sampled_value("1.0f32")
        ),
    )
    .unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let decoded = CompiledContext::decode(&context.encode().unwrap()).unwrap();
    [context, decoded]
}

/// #2463's witness 2 in the in-context lane: a client whose only reference
/// to the library value `sampled` is dead initializes it, so its invalid
/// rate traps, as it does in C (`issue_2463_key_dead_draw_traps`). The
/// valid-rate twin `kept`, named the same way, returns the client's value.
///
/// Evidentiary status: REGRESSION TEST. At 727e74b41 the `sampled` client
/// returned `[1, 1]` from `eval_in_context`, in both contexts.
#[test]
fn a_dead_reference_to_a_library_value_declaration_initializes_it_in_context() {
    let client = |name: &str| {
        format!(
            "module Drawlib.Client\nimport Drawlib.Draw ({name})\ndef main() -> tensor[2, f32] = {{\n  dead = {name}\n  to_tensor([1.0f32, 1.0f32])\n}}\n"
        )
    };
    for context in &draw_library_contexts() {
        let result = eval_in_context(context, &client("kept")).unwrap();
        assert_eq!(lane_of(&result, "main"), Lane::Host);
        assert_eq!(tensor(&result, "main"), vec![1.0; 2]);
        assert_domain_trap(
            eval_in_context(context, &client("sampled")),
            "library value",
        );
    }
}

/// The complement of the two tests above: a value declaration that no
/// declaration the evaluation runs names is never initialized, whether it
/// sits beside the selected root, is named only by an unselected
/// declaration, or is a library value the client imports without naming. An
/// input declaration a kernel body names has no initializer to run.
///
/// Evidentiary status: DISPOSITION LOCK (every row returned its value at
/// 727e74b41 too); the `other` selection is its regression half.
#[test]
fn a_value_declaration_no_run_declaration_names_is_not_initialized() {
    let source = format!(
        "{}def other() -> tensor[2, f32] = {{\n  dead = sampled\n  to_tensor([1.0f32, 1.0f32])\n}}\ndef main() -> tensor[2, f32] = to_tensor([1.0f32, 1.0f32])\n",
        sampled_value("1.0f32")
    );
    let result = eval_selected(request(&source), &["main".into()]).unwrap();
    assert_eq!(lane_of(&result, "main"), Lane::Host);
    assert_eq!(tensor(&result, "main"), vec![1.0; 2]);
    assert_domain_trap(
        eval_selected(request(&source), &["other".into()]),
        "the naming declaration, selected",
    );

    let source = "x: tensor[32, f32] = x\ndef main() -> tensor[2, f32] = {\n  dead = x\n  to_tensor([1.0f32, 1.0f32])\n}\n";
    let result = eval_selected(request(source), &["main".into()]).unwrap();
    assert_eq!(lane_of(&result, "main"), Lane::Host);
    assert_eq!(tensor(&result, "main"), vec![1.0; 2]);

    let client = "module Drawlib.Client\nimport Drawlib.Draw (sampled)\ndef main() -> tensor[2, f32] = to_tensor([1.0f32, 1.0f32])\n";
    for context in &draw_library_contexts() {
        let result = eval_in_context(context, client).unwrap();
        assert_eq!(tensor(&result, "main"), vec![1.0; 2]);
    }
}

/// A discarded draw consumes only its own key, so keeping a selected
/// activation's discarded draws leaves the live draw's key alone, whether
/// the discarded draw sits in the root, in a helper, in a discarded helper
/// call, beside the live draw in one helper, or in a runtime arm that is
/// not taken (the counter stream's version counted ordinals).
#[test]
fn a_selected_activations_discarded_draws_leave_the_live_draws_key_alone() {
    let (k1, k2) = two_keys();
    let rows = [
        (
            "value root, discarded then live",
            "x: tensor[32, f32] = x\nselected = {\n dead = dropout(key_from_seed(8i64), copy(x), 0.5f32)\n dropout(key_from_seed(7i64), copy(x), 0.5f32)\n}\n".to_string(),
            mask(key7()),
        ),
        (
            "helper whose only draw is discarded",
            format!("def h(k: key, v: tensor[32, f32]) -> tensor[32, f32] = {{\n dead = dropout(k, copy(v), 0.5f32)\n v\n}}\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = {{\n {TWO_KEYS}\n a = h(k1, copy(x))\n dropout(k2, x, 0.5f32)\n}}\n"),
            mask(k2),
        ),
        (
            "discarded call of a helper",
            format!("def h(k: key, v: tensor[32, f32]) -> tensor[32, f32] = dropout(k, v, 0.5f32)\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = {{\n {TWO_KEYS}\n dead = h(k1, copy(x))\n dropout(k2, x, 0.5f32)\n}}\n"),
            mask(k2),
        ),
        (
            "helper, discarded then live",
            format!("def h(kd: key, kl: key, v: tensor[32, f32]) -> tensor[32, f32] = {{\n dead = dropout(kd, copy(v), 0.5f32)\n dropout(kl, v, 0.5f32)\n}}\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = {{\n {TWO_KEYS}\n h(k1, k2, x)\n}}\n"),
            mask(k2),
        ),
        (
            "runtime if whose unselected arm draws",
            format!("def pick(k: key, v: tensor[32, f32], flag: bool) -> tensor[32, f32] = if flag then dropout(k, v, 0.5f32) else v\ndef selected(x: tensor[32, f32]) -> tensor[32, f32] = {{\n {TWO_KEYS}\n flag = lt(tensor_to_scalar(sum(copy(x), 0i32)), 0.0f32)\n a = pick(k1, copy(x), flag)\n add(a, dropout(k2, x, 0.5f32))\n}}\n"),
            mask(k2).iter().map(|value| value + 1.0).collect(),
        ),
    ];
    assert_ne!(mask(k1), mask(k2), "the key rows must discriminate");
    assert_ne!(mask(key_reference::key_from_seed(8)), mask(key7()));
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
        "def loss(k: key, x: tensor[32, f32]) -> f32 = {{\n dead = dropout(k, x, 0.5f32)\n 3.0f32\n}}\ndef draw(k: key, x: tensor[32, f32]) -> tensor[32, f32] = dropout(k, x, 0.5f32)\ndef main() = {{\n {TWO_KEYS}\n x = to_tensor([{ones}])\n gradient = grad(loss, wrt=x)(k1, x)\n (gradient, draw(k2, x))\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()]).unwrap();
    let gradient = tensor(&result, "main.0");
    assert_eq!(gradient, vec![0.0; 32]);
    assert!(gradient.iter().all(|value| value.to_bits() == 0));
    assert_eq!(tensor(&result, "main.1"), mask(two_keys().1));
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

/// The keys `rep` and `walk` split off at each recursion step: `now` draws,
/// `later` recurses.
fn recursion_keys(key: u64, steps: usize) -> Vec<u64> {
    let mut later = key;
    (0..steps)
        .map(|_| {
            let (now, next) = key_reference::split(later);
            later = next;
            now
        })
        .collect()
}

/// chelis#2405: a fixed-rate draw beneath a recursive or dynamic caller
/// runs. The caller's control flow used to become an execution exclusion
/// inherited by every nested dispatch, which sent the draw to the host
/// interpreter's builtin table, where `dropout` does not exist, and eval
/// failed with "unknown runtime name `dropout`" on programs the C lane runs.
///
/// Evidentiary status: REGRESSION TEST (each row failed on the chelis#2405
/// base with that error).
#[test]
fn issue_2405_dropout_beneath_recursion_and_runtime_if_draws_its_keys() {
    let ones = ones32();
    let (k1, k2) = two_keys();
    let recursion = format!(
        "def rep(k: key, x: tensor[32, f32], n: i64) -> tensor[32, f32] = if eq(n, 0i64) then x else {{\n (now, later) = split_key(k)\n rep(later, dropout(now, x, 0.5f32), sub(n, 1i64))\n}}\ndef main() = {{\n {TWO_KEYS}\n x = to_tensor([{ones}])\n r = rep(k1, copy(x), 3i64)\n after = dropout(k2, x, 0.5f32)\n (r, after)\n}}\n"
    );
    let result = eval_selected(request(&recursion), &["main".into()])
        .unwrap_or_else(|error| panic!("{recursion}\n{error:?}"));
    let steps = recursion_keys(k1, 3);
    let kept = (0..32)
        .map(|index| mask(steps[0])[index] * mask(steps[1])[index] * mask(steps[2])[index])
        .collect::<Vec<_>>();
    assert!(kept.contains(&8.0) && kept.contains(&0.0));
    assert_eq!(tensor(&result, "main.0"), kept);
    assert_eq!(tensor(&result, "main.1"), mask(k2));

    // An untaken branch holding the draw draws nothing; a taken one draws
    // its key's mask. The following draw is its own key's either way.
    for (flag, first) in [("false", vec![1.0; 32]), ("true", mask(k1))] {
        let branch = format!(
            "def pick(k: key, x: tensor[32, f32], flag: bool) -> tensor[32, f32] = if flag then dropout(k, x, 0.5f32) else x\ndef main() = {{\n {TWO_KEYS}\n x = to_tensor([{ones}])\n a = pick(k1, copy(x), {flag})\n b = dropout(k2, x, 0.5f32)\n (a, b)\n}}\n"
        );
        let result = eval_selected(request(&branch), &["main".into()])
            .unwrap_or_else(|error| panic!("{branch}\n{error:?}"));
        assert_eq!(tensor(&result, "main.0"), first, "{flag}");
        assert_eq!(tensor(&result, "main.1"), mask(k2), "{flag}");
    }
}

/// chelis#2405: a `match` in a definition that draws nothing does not remove
/// `dropout` from the rest of the program, whether the draw sits beside the
/// call or in a separate block.
///
/// Evidentiary status: REGRESSION TEST (both rows failed on the chelis#2405
/// base with "unknown runtime name `dropout`").
#[test]
fn issue_2405_unrelated_match_leaves_dropout_runnable() {
    let ones = ones32();
    let scale = "type Mode = | Train | Infer\ndef scale(m: Mode) -> f32 =\n  match m with {\n    | Train => 2.0f32\n    | Infer => 1.0f32\n  }\n";
    for main in [
        format!(
            "def main() = {{\n x = to_tensor([{ones}])\n s = scale(Train)\n d = dropout(key_from_seed(7i64), x, 0.5f32)\n (d, s)\n}}\n"
        ),
        format!(
            "def main() = {{\n s = scale(Train)\n d = {{ k = key_from_seed(7i64)\n dropout(k, to_tensor([{ones}]), 0.5f32) }}\n (d, s)\n}}\n"
        ),
    ] {
        let source = format!("{scale}{main}");
        let result = eval_selected(request(&source), &["main".into()])
            .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
        assert_eq!(tensor(&result, "main.0"), mask(key7()));
        assert_eq!(scalar_root(&result, "main.1"), 2.0);
    }
}

/// chelis#2405 retired the execution exclusion that a recursive program's
/// helpers used to run under, so a drawing helper beneath recursion takes
/// the planned kernel entry. Each application draws with the key its caller
/// split off: `a` sums the three draws `walk` makes, and `b` and `c` draw
/// with the other two keys.
///
/// Evidentiary status: DISPOSITION LOCK. The pinned bits are
/// `key_ref.py`'s `unit` rounded to f32 over the same key tree (for `a`,
/// summed in f32 in the program's order).
#[test]
fn uniform_draws_beneath_recursion_take_their_split_keys() {
    let source = "def draw(k: key, x: tensor[4, f32]) -> tensor[4, f32] = uniform_like(k, x, 0.0f32, 1.0f32)\ndef walk(k: key, n: i64, x: tensor[4, f32]) -> tensor[4, f32] = if eq(n, 0i64) then x else {\n (now, later) = split_key(k)\n add(draw(now, copy(x)), walk(later, sub(n, 1i64), x))\n}\ndef main() = {\n (kw, rest) = split_key(key_from_seed(7i64))\n (kb, kc) = split_key(rest)\n x = to_tensor([0.0f32, 0.0f32, 0.0f32, 0.0f32])\n a = walk(kw, 3i64, copy(x))\n b = draw(kb, copy(x))\n c = uniform_like(kc, x, 0.0f32, 1.0f32)\n (a, b, c)\n}\n";
    let result = eval_selected(request(source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    let (kw, rest) = key_reference::split(key7());
    let (kb, kc) = key_reference::split(rest);
    let draws = recursion_keys(kw, 3)
        .into_iter()
        .map(|key| key_reference::unit_f32(key, 4))
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
            [0x3faa_a737, 0x3f6c_5d04, 0x3fe2_48ac, 0x3fc8_e28a],
        ),
        (
            "main.1",
            bits(&key_reference::unit_f32(kb, 4)),
            [0x3f66_a91e, 0x3e8b_fd70, 0x3ef1_f093, 0x3f67_3d83],
        ),
        (
            "main.2",
            bits(&key_reference::unit_f32(kc, 4)),
            [0x3e05_c531, 0x3d54_7ad6, 0x3e3f_ee26, 0x3f04_9c3c],
        ),
    ] {
        assert_eq!(
            expected, pinned,
            "{root}: the transcription and key_ref.py agree"
        );
        assert_eq!(observed(root), expected, "{root}");
    }
}

/// chelis#2405: `grad` of a function whose draw sits under runtime control
/// draws the taken arm with its key and replays that mask for the pathwise
/// adjoint. `vmap` of a drawing function, which the counter stream refused
/// (chelis#2409), maps a `tensor[n, key]` of keys: row `j` draws with
/// `fold_in(k, j)`, and a draw after the `vmap` is its own key's.
///
/// Evidentiary status: REGRESSION TEST for the `grad` row (it returned a
/// mask from the pre-[05-RNG-1] formula before the chelis#2405 fence).
#[test]
fn dropout_under_grad_of_dynamic_control_draws_and_vmap_maps_its_keys() {
    let ones = ones32();
    let row = "[1.0f32, 1.0f32, 1.0f32, 1.0f32]";
    let (k1, k2) = two_keys();
    let source = format!(
        "def loss(k: key, x: tensor[32, f32]) -> tensor[f32] = if gt(tensor_to_scalar(sum(copy(x), 0i32)), 0.0f32) then sum(dropout(k, x, 0.5f32), 0i32) else sum(x, 0i32)\ndef main() = {{\n {TWO_KEYS}\n x = to_tensor([{ones}])\n g = grad(loss, wrt=x)(k1, copy(x))\n after = dropout(k2, x, 0.5f32)\n (g, after)\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()])
        .unwrap_or_else(|error| panic!("{error:?}"));
    assert_eq!(tensor(&result, "main.0"), mask(k1));
    assert_eq!(tensor(&result, "main.1"), mask(k2));

    let vmap_keep =
        "def keep(k: key, x: tensor[4, f32]) -> tensor[4, f32] = dropout(k, x, 0.5f32)\n";
    let source = format!(
        "{vmap_keep}def main() = {{\n {TWO_KEYS}\n xs = to_tensor([{row}, {row}, {row}])\n ys = vmap(keep)(split_keys(k1, 3i64), xs)\n after = dropout(k2, to_tensor({row}), 0.5f32)\n (ys, after)\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()])
        .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
    let rows = (0..3)
        .flat_map(|j| key_reference::mask(key_reference::fold_in(k1, j), 4))
        .collect::<Vec<_>>();
    assert_eq!(tensor(&result, "main.0"), rows);
    assert_eq!(tensor(&result, "main.1"), mask(k2)[..4]);
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
/// Evidentiary status: DISPOSITION LOCK (it passed before and after the
/// round-2 change, and fails if the cached snapshot is read for a captured
/// closure).
#[test]
fn grad_of_a_captured_drawing_closure_still_sees_its_draw() {
    let ones = ones32();
    let source = format!(
        "def main() = {{\n {TWO_KEYS}\n x = to_tensor([{ones}])\n keep = fn (j: key, v: tensor[32, f32]) -> sum(dropout(j, v, 0.5f32), 0i32)\n g = grad(keep, wrt=v)(k1, copy(x))\n next = dropout(k2, x, 0.5f32)\n (g, next)\n}}\n"
    );
    let result = eval_selected(request(&source), &["main".into()])
        .unwrap_or_else(|error| panic!("{source}\n{error:?}"));
    let (k1, k2) = two_keys();
    assert_eq!(tensor(&result, "main.0"), mask(k1));
    assert_eq!(tensor(&result, "main.1"), mask(k2));
}

/// Only the selected arm of a runtime `if` draws, so in the DAG evaluator a
/// draw in an unselected arm neither validates its controls nor produces a
/// value (chelis#2410): the whole-program graph computes both arms of a
/// `where`, and each draw there carries its arm's path condition as its
/// activation. Each row adds the arm's value to a later draw keyed by `k2`.
/// Flags come from data, so lowering cannot fold them.
///
/// Evidentiary status: REGRESSION TEST for the row taking `grad` through an
/// unselected drawing arm, which dcc9256c4 refused with the #2410 rejection.
/// The other rows lock this lane's draws beside the lowering change.
#[test]
fn a_draw_in_an_unselected_arm_neither_validates_nor_draws_in_the_dag_evaluator() {
    let sum = "tensor_to_scalar(sum(copy(x), 0i32))";
    let selected = |body: &str| {
        format!(
            "def selected(x: tensor[32, f32]) -> tensor[32, f32] = {{\n s = {sum}\n {TWO_KEYS}\n{body}}}\n"
        )
    };
    let (k1, k2) = two_keys();
    let unit = |key| key_reference::unit_f32(key, 32);
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
    let quarter_rate = unit(k1)
        .into_iter()
        .map(|unit| if unit < 0.25 { 0.0 } else { 1.0_f32 / 0.75_f32 })
        .collect::<Vec<_>>();
    let noisy = "def layer(k: key, x: tensor[32, f32], noisy: bool, eps: f32) -> tensor[32, f32] = if noisy then add(copy(x), uniform_like(k, x, neg(eps), eps)) else x\n";
    let loss = |comparison: &str| {
        format!(
            "def loss(k: key, x: tensor[32, f32]) -> tensor[f32] = if {comparison}({sum}, 0.0f32) then sum(dropout(k, x, 0.5f32), 0i32) else sum(x, 0i32)\n"
        )
    };
    let rows = [
        (
            "uniform, invalid run-time bounds, unselected",
            format!(
                "{noisy}{}",
                selected(" layer(k1, x, lt(s, 0.0f32), sub(0.0f32, s))\n")
            ),
            ones.clone(),
        ),
        (
            "uniform, valid run-time bounds, unselected, then a draw",
            format!(
                "{noisy}{}",
                selected(
                    " add(layer(k1, copy(x), lt(s, 0.0f32), s), uniform_like(k2, x, 0.0f32, 1.0f32))\n"
                )
            ),
            add(&ones, &unit(k2)),
        ),
        (
            "uniform in a nested arm, then a draw",
            format!(
                "def pick(k: key, x: tensor[32, f32], a: bool, b: bool) -> tensor[32, f32] = if a then if b then uniform_like(k, x, 0.0f32, 1.0f32) else x else x\n{}",
                selected(
                    " add(pick(k1, copy(x), gt(s, 0.0f32), lt(s, 0.0f32)), uniform_like(k2, x, 0.0f32, 1.0f32))\n"
                )
            ),
            add(&ones, &unit(k2)),
        ),
        (
            "dropout in both arms, then a draw",
            format!(
                "def pick(k: key, x: tensor[32, f32], flag: bool) -> tensor[32, f32] = if flag then dropout(k, x, 0.5f32) else dropout(k, x, 0.25f32)\n{}",
                selected(" add(pick(k1, copy(x), lt(s, 0.0f32)), dropout(k2, x, 0.5f32))\n")
            ),
            add(&quarter_rate, &as_f32(mask(k2))),
        ),
        (
            "grad through an unselected drawing arm, then a draw",
            format!(
                "{}{}",
                loss("lt"),
                selected(" add(grad(loss, wrt=x)(k1, copy(x)), dropout(k2, x, 0.5f32))\n")
            ),
            add(&ones, &as_f32(mask(k2))),
        ),
        (
            "grad through a selected drawing arm, then a draw",
            format!(
                "{}{}",
                loss("gt"),
                selected(" add(grad(loss, wrt=x)(k1, copy(x)), dropout(k2, x, 0.5f32))\n")
            ),
            add(&as_f32(mask(k1)), &as_f32(mask(k2))),
        ),
    ];
    assert_ne!(unit(k1), unit(k2));
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
