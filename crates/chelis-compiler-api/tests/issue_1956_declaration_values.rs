//! #1956 prerequisite: declaration memoization is independent of lexical frames.
use chelis_compiler_api::compiler::{eval, eval_selected};
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use serde_json::{Value, json};

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: Default::default(),
    }
}

fn assert_out(source: &str, expected: Value) {
    let result = eval(request(source)).unwrap();
    let out = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("out"))
        .unwrap();
    assert_eq!(serde_json::to_value(&out.value).unwrap(), expected);
}

#[test]
fn ordinary_named_tensor_uses_global() {
    assert_out(
        "w = to_tensor([3.0f32, 5.0f32])\ndef weighted_input(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, w)\nout = { w = to_tensor([7.0f32, 11.0f32])\n weighted_input(to_tensor([1.0f32, 2.0f32])) }\n",
        json!({"type":"tensor","value":{"shape":[2],"data":{"dtype":"f32","bits":["40400000","41200000"]}}}),
    );
}

#[test]
fn anonymous_uses_lexical_shadow() {
    assert_out(
        "a = 3.0f32\nout = { a = 7.0f32\n f = fn (x: f32) -> add(x, a)\n f(1.0f32) }\n",
        json!({"type":"scalar","value":{"dtype":"f32","bits":"41000000"}}),
    );
}

#[test]
fn named_scalar_and_helper_keep_declaration_scope() {
    for source in [
        "a = 3.0f32\ndef read(x: f32) -> f32 = add(x, a)\nout = { a = 7.0f32\n read(1.0f32) }\n",
        "def twice(x: f32) -> f32 = add(x, x)\ndef read(x: f32) -> f32 = twice(x)\nout = { twice = fn (x: f32) -> mul(x, 3.0f32)\n read(2.0f32) }\n",
    ] {
        assert_out(
            source,
            json!({"type":"scalar","value":{"dtype":"f32","bits":"40800000"}}),
        );
    }
}

#[test]
fn initializer_runs_once_eager_or_lazy_through_nested_calls() {
    let source = "a = { _ = print(\"init\")\n 3.0f32 }\ndef read(x: f32) -> f32 = add(x, a)\ndef outer(x: f32) -> f32 = read(x)\nout = add(outer(1.0f32), outer(2.0f32))\n";
    let results = [
        eval(request(source)).unwrap(),
        eval_selected(request(source), &["out".into()]).unwrap(),
    ];
    for result in &results {
        assert_eq!(
            serde_json::to_value(&result.roots.last().unwrap().value).unwrap(),
            json!({"type":"scalar","value":{"dtype":"f32","bits":"41100000"}})
        );
    }
    assert_eq!(
        results.each_ref().map(|result| result.transcript.clone()),
        [vec!["init".to_string()], vec!["init".to_string()]]
    );
}

#[test]
fn nullary_calls_aliases_and_observation_roots_remain_distinct() {
    let source = "def value() -> i32 = 5\nalias = value\ndef noisy() -> i32 ! { IO } = { _ = print(\"call\")\n 7 }\nout = add(noisy(), noisy())\nother = alias()\n";
    let result = eval(request(source)).unwrap();
    assert_eq!(result.transcript, ["call", "call"]);
    for (name, expected) in [("value", 5), ("out", 14), ("other", 5)] {
        let root = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some(name))
            .unwrap();
        assert_eq!(
            serde_json::to_value(&root.value).unwrap(),
            json!({"type":"scalar","value":{"dtype":"int32","value":expected}})
        );
    }
    assert!(
        !result
            .roots
            .iter()
            .any(|root| root.name.as_deref() == Some("alias")
                || root.name.as_deref() == Some("noisy"))
    );
}

#[test]
fn selected_named_axis_formal_does_not_force_shadowed_global_initializer() {
    let source = "w = { _ = print(\"unselected\")\n to_tensor([3.0f32, 5.0f32]) }\ndef total(w: tensor[seq, f32]) -> tensor[f32] = sum(w, seq)\nout = { w = to_tensor([7.0f32, 11.0f32])\n total(w) }\n";
    let result = eval_selected(request(source), &["out".into()]).unwrap();
    assert_eq!(result.transcript, Vec::<String>::new());
    assert_eq!(
        serde_json::to_value(&result.roots[0].value).unwrap(),
        json!({"type":"tensor","value":{"shape":[],"data":{"dtype":"f32","bits":["41900000"]}}})
    );
}

#[test]
fn successful_declaration_values_do_not_cross_requests() {
    for value in [3, 5, 3] {
        let source = format!("a = {{ _ = print(\"init\")\n {value} }}\nout = a\n");
        let result = eval(request(&source)).unwrap();
        assert_eq!(result.transcript, ["init"]);
        let out = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("out"))
            .unwrap();
        assert_eq!(
            serde_json::to_value(&out.value).unwrap(),
            json!({"type":"scalar","value":{"dtype":"int32","value":value}})
        );
    }
}

fn linked_context(files: &[(&str, &str)]) -> chelis_compiler_api::CompiledContext {
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(
        directory.path().join("reef.toml"),
        format!("[package]\nname = \"declaration-frames\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Probe\"\n", chelis_compiler_api::COMPILER_VERSION),
    ).unwrap();
    for (name, source) in files {
        std::fs::write(directory.path().join("src").join(name), source).unwrap();
    }
    chelis_compiler_api::compile_reef_context(
        directory.path(),
        directory.path(),
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .unwrap()
}

fn selected_context_out(
    context: &chelis_compiler_api::CompiledContext,
    source: &str,
) -> chelis_compiler_api::schema::EvalResult {
    chelis_compiler_api::compiler::prepare_eval_in_context(context, source)
        .expect("source passes full contextual admission")
        .eval_root(Default::default(), "out")
        .expect("selected host value executes")
}

#[test]
fn qualified_private_declarations_with_the_same_terminal_stay_distinct() {
    let context = linked_context(&[
        (
            "left.ch",
            "module Probe.Left\nexport (left)\nvalue = { _ = print(\"left\")\n 3 }\ndef left(x: i32) -> i32 = add(value, x)\n",
        ),
        (
            "right.ch",
            "module Probe.Right\nexport (right)\nvalue = { _ = print(\"right\")\n 5 }\ndef right(x: i32) -> i32 = add(value, x)\n",
        ),
    ]);
    let result = selected_context_out(
        &context,
        "module Probe.Client\nimport Probe.Left (left)\nimport Probe.Right (right)\nout = add(add(left(1), right(1)), left(2))\n",
    );
    let out = result
        .roots
        .iter()
        .find(|root| root.name.as_deref() == Some("out"))
        .unwrap();
    assert_eq!(
        serde_json::to_value(&out.value).unwrap(),
        json!({"type":"scalar","value":{"dtype":"int32","value":15}})
    );
    assert_eq!(result.transcript, ["left", "right"]);
}

#[test]
fn new_source_replaces_library_value_and_callable_without_cross_request_reuse() {
    let context = linked_context(&[(
        "values.ch",
        "module Probe.Values\nexport (value, read)\nvalue = { _ = print(\"library\")\n 3 }\ndef read(x: i32) -> i32 = add(value, x)\n",
    )]);
    for replacement in [7, 11, 7] {
        let source = format!(
            "module Probe.Values\nvalue = {{ _ = print(\"source\")\n {replacement} }}\ndef read(x: i32) -> i32 = add(value, mul(x, 2))\nout = read(1)\n"
        );
        let result = selected_context_out(&context, &source);
        let out = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("out"))
            .unwrap();
        assert_eq!(
            serde_json::to_value(&out.value).unwrap(),
            json!({"type":"scalar","value":{"dtype":"int32","value":replacement + 2}})
        );
        assert_eq!(result.transcript, ["source"]);
    }
}

#[test]
fn reused_prepared_request_reinitializes_successful_library_values() {
    let context = linked_context(&[(
        "values.ch",
        "module Probe.Values\nexport (read)\nvalue = { _ = print(\"initialize\")\n 3 }\ndef read(x: i32) -> i32 = add(value, x)\n",
    )]);
    let prepared = chelis_compiler_api::compiler::prepare_eval_in_context(
        &context,
        "module Probe.Client\nimport Probe.Values (read)\nout = read(1)\n",
    )
    .unwrap();
    for _ in 0..2 {
        let result = prepared.eval_root(Default::default(), "out").unwrap();
        let out = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("out"))
            .unwrap();
        assert_eq!(
            serde_json::to_value(&out.value).unwrap(),
            json!({"type":"scalar","value":{"dtype":"int32","value":4}})
        );
        assert_eq!(result.transcript, ["initialize"]);
    }
}
