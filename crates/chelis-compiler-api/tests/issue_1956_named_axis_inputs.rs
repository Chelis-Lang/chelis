//! #1956: selected named-axis inputs, without declaration-environment migration.
use chelis_compiler_api::compiler::eval;
use chelis_compiler_api::schema::{EvalRequest, SourceKind};
use serde_json::json;

fn request(source: &str) -> EvalRequest {
    EvalRequest {
        source_kind: SourceKind::Surf,
        source: source.into(),
        bindings: Default::default(),
    }
}

fn prepare(library: &str) -> chelis_compiler_api::compiler::PreparedEvalInContext {
    use chelis_compiler_api::compiler::prepare_eval_in_context;
    use chelis_compiler_api::{COMPILER_VERSION, compile_reef_context};
    let directory = tempfile::tempdir().unwrap();
    std::fs::create_dir(directory.path().join("src")).unwrap();
    std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"declaration_ownership\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
    std::fs::write(
        directory.path().join("src/values.ch"),
        format!("module Probe.Values\nexport (total)\n{library}"),
    )
    .unwrap();
    let context = compile_reef_context(directory.path(), directory.path()).unwrap();
    let draws = "_ = uniform_like(copy(x), 0.0f32, 1.0f32)\n".repeat(5);
    let source = format!(
        "module Probe.Client\nimport Probe.Values (total)\ndef entry(x: tensor[seq, f32]) = with seed(42i64) {{ _ = print(\"entry\")\n {draws} result = total(x)\n (result, uniform_like(x, 0.0f32, 1.0f32)) }}\nalias = entry\ndef main() = alias(to_tensor([7.0f32, 11.0f32]))\nout = main()\n"
    );
    prepare_eval_in_context(&context, &source).unwrap()
}

#[test]
fn spread_rank_library_formal_does_not_initialize_same_named_declaration() {
    let prepared = prepare(
        "weights = with seed(17i64) { _ = print(\"initialize\")\n to_tensor([3.0f32, 5.0f32]) }\ndef total(weights: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(weights, seq)\n",
    );
    let results = (0..2)
        .map(|_| {
            prepared
                .eval_root(std::collections::BTreeMap::new(), "out")
                .unwrap()
        })
        .collect::<Vec<_>>();
    for result in &results {
        let out = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("out.0"))
            .unwrap();
        assert_eq!(
            serde_json::to_value(&out.value).unwrap(),
            json!({"type":"tensor","value":{"shape":[],"data":{"dtype":"f32","bits":["41900000"]}}})
        );
        let next = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("out.1"))
            .unwrap();
        let seed = 42u64 ^ 5u64.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let expected = chelis_types::tensor_from_scalars(
            chelis_types::types::Prim::F32,
            &(0..2)
                .map(|index| {
                    chelis_types::uniform_sample(
                        chelis_types::types::Prim::F32,
                        0.0,
                        1.0,
                        seed,
                        index,
                    )
                    .unwrap()
                })
                .collect::<Vec<_>>(),
        );
        assert_eq!(
            serde_json::to_value(&next.value).unwrap(),
            json!({"type":"tensor","value":{"shape":[2],"data":expected}})
        );
    }
    assert_eq!(
        results
            .into_iter()
            .map(|result| result.transcript)
            .collect::<Vec<_>>(),
        vec![vec!["entry"], vec!["entry"]]
    );
}

fn capture_library(initializer: &str, body: &str) -> String {
    format!(
        "baseline = to_tensor([7.0f32, 11.0f32])\nweights = with seed(17i64) {{ _ = print(\"initialize\")\n {initializer} }}\ndef total(x: &tensor[..pre, seq, ..post, f32]) -> (tensor[..pre, ..post, f32], tensor[f32]) = {body}\n"
    )
}

fn assert_capture(body: &str, second: &str, transcript: &[&str]) {
    let prepared = prepare(&capture_library("to_tensor([3.0f32, 5.0f32])", body));
    for _ in 0..2 {
        let result = prepared.eval_root(Default::default(), "out").unwrap();
        for (name, expected) in [("out.0.0", "41900000"), ("out.0.1", second)] {
            let root = result
                .roots
                .iter()
                .find(|root| root.name.as_deref() == Some(name))
                .unwrap();
            assert_eq!(
                serde_json::to_value(&root.value).unwrap(),
                json!({"type":"tensor","value":{"shape":[],"data":{"dtype":"f32","bits":[expected]}}})
            );
        }
        let next = result
            .roots
            .iter()
            .find(|root| root.name.as_deref() == Some("out.1"))
            .unwrap();
        assert_eq!(
            serde_json::to_value(&next.value).unwrap(),
            json!({"type":"tensor","value":{"shape":[2],"data":{"dtype":"f32","bits":["3e68de41","3f38fdad"]}}})
        );
        assert_eq!(result.transcript, transcript);
    }
}

#[test]
fn spread_rank_library_selected_capture_runs_initializer() {
    assert_capture(
        "(sum(x, seq), sum(weights, 0i32))",
        "41000000",
        &["entry", "initialize"],
    );
}

#[test]
fn spread_rank_library_dead_capture_does_not_initialize() {
    assert_capture(
        "(sum(x, seq), if true then sum(baseline, 0i32) else sum(weights, 0i32))",
        "41900000",
        &["entry"],
    );
}

#[test]
fn spread_rank_library_capture_preserves_initializer_error() {
    let prepared = prepare(&capture_library(
        "_ = uniform_like(to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32)\n to_tensor([cast(floor_div(1i32, 0i32), f32), 0.0f32])",
        "(sum(x, seq), sum(weights, 0i32))",
    ));
    for _ in 0..2 {
        let error = prepared.eval_root(Default::default(), "out").unwrap_err();
        assert_eq!(error.stage, "eval");
        assert_eq!(error.errors.len(), 1);
        assert_eq!(
            error.errors[0].message,
            "numeric trap: division by zero in floor_div at int32"
        );
        assert_eq!(error.transcript, ["entry", "initialize"]);
    }
}

#[test]
fn spread_rank_call_without_a_named_axis_remains_a_check_error() {
    let error = eval(request("def total(x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\nout = total(to_tensor([7.0f32, 11.0f32]))\n")).unwrap_err();
    assert_eq!(error.stage, "check");
    assert!(error.transcript.is_empty());
    assert!(error.errors.iter().any(|error| error.kind() == chelis_vocab::DiagnosticKind::DimensionMismatch && error.message == "rank-spread operand carries no named `seq` axis; a fully-literal or differently-named operand cannot locate the axis (spec/04-type-system.md §4.5.3)"));
}
