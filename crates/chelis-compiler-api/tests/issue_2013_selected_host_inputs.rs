//! #2013: already-admitted selected Host roots receive their live tensor actuals.
//! Root admission and profile dispatch remain owned by the existing compiler.
//! Tagged values / traps follow spec/04-type-system.md [04-NUM-8/9]; seeded
//! transcript and draw observations follow spec/05-risc-primitives.md.
use chelis_compiler_api::compiler::{
    PreparedEvalInContext, check_in_context, prepare_eval_in_context,
};
use chelis_compiler_api::schema::{EvalResult, TensorValue};
use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
use chelis_types::types::Lane;
use serde_json::{Value, json};
use std::collections::BTreeMap;

const LIBRARY: &str = "weights = { _ = print(\"initialize\")\n\
    to_tensor([3.0f32, 5.0f32]) }\n\
    def total[pre, post](weights: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(weights, seq)\n";

// Reuse the owning public prepared-context harness, with a parameterized root.
fn prepare(client: &str) -> PreparedEvalInContext {
    prepare_with_library(LIBRARY, client)
}

fn prepare_with_library(library: &str, client: &str) -> PreparedEvalInContext {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!(
        "[package]\nname = \"declaration_ownership\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n"
    )).unwrap();
    std::fs::write(
        directory.path().join("src/values.ch"),
        format!("module Probe.Values\nexport (total)\n{library}"),
    )
    .unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let client = format!("module Probe.Client\n{client}");
    let checked = check_in_context(&context, &client).unwrap();
    assert_eq!(checked.score.get(), 1.0);
    assert!(checked.errors.is_empty(), "{checked:?}");
    assert!(checked.unresolved_names.is_empty(), "{checked:?}");
    prepare_eval_in_context(&context, &client).unwrap()
}

fn tensor(shape: &[i64], dtype: &str, bits: &[&str]) -> TensorValue {
    TensorValue {
        shape: shape.into(),
        data: serde_json::from_value(json!({"dtype":dtype,"bits":bits})).unwrap(),
    }
}

fn input(bits: &[&str]) -> BTreeMap<String, TensorValue> {
    BTreeMap::from([("x".into(), tensor(&[2], "f32", bits))])
}

fn tagged(shape: &[i64], dtype: &str, bits: &[&str]) -> Value {
    json!({"type":"tensor","value":{"shape":shape,"data":{"dtype":dtype,"bits":bits}}})
}

fn assert_host(result: &EvalResult, expected: &[(&str, Value)], transcript: &[&str]) {
    assert_host_with_inputs(result, expected, transcript, &["x"]);
}

fn assert_host_with_inputs(
    result: &EvalResult,
    expected: &[(&str, Value)],
    transcript: &[&str],
    required_inputs: &[&str],
) {
    assert_eq!(result.roots.len(), expected.len(), "{result:?}");
    assert_eq!(
        result
            .manifest
            .entries
            .iter()
            .map(|entry| entry.name.as_str())
            .collect::<Vec<_>>(),
        expected.iter().map(|(name, _)| *name).collect::<Vec<_>>(),
        "{result:?}"
    );
    for (root, (name, value)) in result.roots.iter().zip(expected) {
        assert_eq!(root.name.as_deref(), Some(*name), "{result:?}");
        let entry = result
            .manifest
            .entries
            .iter()
            .find(|entry| entry.name == *name)
            .unwrap();
        assert_eq!(entry.lane, Lane::Host, "{result:?}");
        assert_eq!(entry.required_inputs, required_inputs, "{result:?}");
        assert_eq!(serde_json::to_value(&root.value).unwrap(), *value);
    }
    assert_eq!(result.transcript, transcript);
}

fn sampled_client(params: &str, total: &str) -> String {
    // Five discarded draws on their own keys; under explicit keys they do not
    // move the returned draw, which is keyed by `key_from_seed(42)` alone.
    let draws = (0..5)
        .map(|n| {
            format!(
                "_ = uniform_like(fold_in(key_from_seed(7i64), {n}i64), copy(x), 0.0f32, 1.0f32)\n"
            )
        })
        .collect::<String>();
    format!(
        "def main({params}) = {{ _ = print(\"entry\")\n\
         {draws} result = {total}\n (result, uniform_like(key_from_seed(42i64), x, 0.0f32, 1.0f32)) }}\n"
    )
}

fn assert_sample(result: &EvalResult, sum: &str) {
    assert_host(
        result,
        &[
            ("main.0", tagged(&[], "f32", &[sum])),
            // [05-OP-8] keyed by `key_from_seed(42)`, [0, 1), f32: the f32
            // rounding of key_ref.py's `unit(key_from_seed(42), i)`, i = 0, 1.
            ("main.1", tagged(&[2], "f32", &["3efa06fe", "3e762d86"])),
        ],
        &["entry"],
    );
}

#[test]
fn original_library_client_preserves_repeated_and_changed_actuals() {
    let prepared = prepare(&format!(
        "import Probe.Values (total)\n{}",
        sampled_client("x: tensor[seq, f32]", "total(x)")
    ));
    let cases = [
        (["40e00000", "41300000"], "41900000"), // [7, 11] -> 18
        (["40400000", "40a00000"], "41000000"), // [3, 5] -> 8
        (["40e00000", "41300000"], "41900000"),
    ];
    // Evaluate all calls before asserting so an early error cannot hide reuse.
    let results = cases.map(|(bits, _)| prepared.eval_root(input(&bits), "main"));
    for (result, (_, sum)) in results.iter().zip(cases) {
        assert_sample(
            result.as_ref().unwrap_or_else(|_| panic!("{results:?}")),
            sum,
        );
    }
}

#[test]
fn capture_free_legacy_uniform_host_twin_receives_actuals() {
    let prepared = prepare(&sampled_client("x: tensor[seq, f32]", "sum(x, seq)"));
    let result = prepared
        .eval_root(input(&["40e00000", "41300000"]), "main")
        .unwrap();
    assert_sample(&result, "41900000");
}

#[test]
fn dead_before_live_parameter_preserves_authored_arity() {
    let prepared = prepare(&sampled_client(
        "dead: tensor[seq, f32], x: tensor[seq, f32]",
        "sum(x, seq)",
    ));
    // No dead actual: its positional slot must remain present without evaluation.
    let result = prepared
        .eval_root(input(&["40e00000", "41300000"]), "main")
        .unwrap();
    assert_sample(&result, "41900000");
}

#[test]
fn missing_live_actual_does_not_promote_or_enter_a_declaration() {
    let prepared = prepare(&sampled_client(
        "dead: tensor[seq, f32], x: tensor[seq, f32]",
        "sum(x, seq)",
    ));
    let result = prepared
        .eval_root(
            BTreeMap::from([("dead".into(), tensor(&[-1], "f32", &[]))]),
            "main",
        )
        .unwrap();
    assert!(result.manifest.entries.is_empty(), "{result:?}");
    assert!(result.roots.is_empty(), "{result:?}");
    assert!(result.transcript.is_empty(), "{result:?}");
}

#[test]
fn selected_host_ignores_unrelated_inputs_and_unselected_sibling() {
    let source = format!(
        "{}\ndef sibling() -> i32 = {{ _ = print(\"sibling\")\n floor_div(1i32, 0i32) }}\n",
        sampled_client("x: tensor[seq, f32]", "sum(x, seq)")
    );
    let prepared = prepare(&source);
    let absent = prepared.eval_root(BTreeMap::new(), "main").unwrap();
    assert!(absent.manifest.entries.is_empty(), "{absent:?}");
    assert!(absent.roots.is_empty(), "{absent:?}");
    assert!(absent.transcript.is_empty(), "{absent:?}");
    let mut bindings = input(&["40e00000", "41300000"]);
    bindings.insert("unrelated".into(), tensor(&[-1], "f32", &[]));
    let result = prepared.eval_root(bindings, "main").unwrap();
    assert_sample(&result, "41900000");
}

#[test]
fn invalid_live_shape_is_rejected_before_entering_the_host_body() {
    let prepared = prepare(&sampled_client("x: tensor[seq, f32]", "sum(x, seq)"));
    let error = prepared
        .eval_root(
            BTreeMap::from([("x".into(), tensor(&[-1], "f32", &[]))]),
            "main",
        )
        .unwrap_err();
    assert_eq!(error.stage, "eval");
    assert!(error.transcript.is_empty(), "{error:?}");
    assert!(
        error
            .errors
            .iter()
            .any(|diagnostic| diagnostic.message.contains("negative")),
        "{error:?}"
    );
}

const INTEGER_HOST: &str =
    "def main[n](x: tensor[n, i32]) -> tensor[n, n, i32] = insert(x, 1, shape(x, 0))\n";

#[test]
fn shape_reading_host_preserves_declared_integer_values() {
    let prepared = prepare(INTEGER_HOST);
    let input = TensorValue {
        shape: vec![2],
        data: serde_json::from_value(json!({"dtype":"int32","values":[1,2]})).unwrap(),
    };
    let result = prepared
        .eval_root(BTreeMap::from([("x".into(), input)]), "main")
        .unwrap();
    assert_host(
        &result,
        &[(
            "main",
            json!({"type":"tensor","value":{
                "shape":[2,2],"data":{"dtype":"int32","values":[1,1,2,2]}
            }}),
        )],
        &[],
    );
}

#[test]
fn nonintegral_actual_reaches_the_declared_integer_ingress_trap() {
    let prepared = prepare(INTEGER_HOST);
    // Host ingress finalizes at the declared dtype; mismatched tags alone are
    // not an invalidity. A nonintegral f32 cannot become an i32 parameter.
    let error = prepared
        .eval_root(input(&["3fc00000", "40000000"]), "main")
        .unwrap_err();
    assert_eq!(error.stage, "eval");
    assert!(error.transcript.is_empty(), "{error:?}");
    assert_eq!(error.errors.len(), 1, "{error:?}");
    assert_eq!(
        error.errors[0].message,
        "numeric trap: domain in param at i32"
    );
}

#[test]
fn fixed_control_host_keeps_its_existing_supplied_input_route() {
    // A negated literal rate keeps its supplied-input route: the keyed draw at
    // rate -0 returns its input. This is not an admission extension for
    // runtime rates.
    let prepared = prepare(
        "def main(x: tensor[2, f32]) -> tensor[2, f32] = dropout(key_from_seed(42i64), x, -0.0f32)\n",
    );
    for bits in [
        ["40e00000", "41300000"],
        ["40400000", "40a00000"],
        ["40e00000", "41300000"],
    ] {
        let result = prepared.eval_root(input(&bits), "main").unwrap();
        assert_host(&result, &[("main", tagged(&[2], "f32", &bits))], &[]);
    }
}

#[test]
fn prepared_library_shape_demand_preserves_repeated_and_changed_actuals() {
    let prepared = prepare_with_library(
        "def total[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, m, f32] = insert(x, 1, shape(x, 0))\n",
        "import Probe.Values (total)\ndef main[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, m, f32] = total(x, y)\n",
    );
    for bits in [
        ["3f800000", "40000000"],
        ["40400000", "40800000"],
        ["3f800000", "40000000"],
    ] {
        let mut bindings = input(&bits);
        bindings.insert("y".into(), tensor(&[2], "f32", &["40400000", "40800000"]));
        bindings.insert("unrelated".into(), tensor(&[-1], "f32", &[]));
        let result = prepared.eval_root(bindings, "main").unwrap();
        assert_host_with_inputs(
            &result,
            &[(
                "main",
                tagged(&[2, 2], "f32", &[bits[0], bits[0], bits[1], bits[1]]),
            )],
            &[],
            &["x", "y"],
        );
    }
}
