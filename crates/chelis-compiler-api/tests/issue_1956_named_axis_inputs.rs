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
    std::fs::write(directory.path().join("reef.toml"), format!("[package]\nname = \"declaration-ownership\"\nversion = \"0.1.0\"\ncompiler = \"={COMPILER_VERSION}\"\nmodule_prefix = \"Probe\"\n")).unwrap();
    std::fs::write(
        directory.path().join("src/values.ch"),
        format!("module Probe.Values\nexport (total)\n{library}"),
    )
    .unwrap();
    let context = compile_reef_context(
        directory.path(),
        directory.path(),
        &chelis_std_bundle::EMBEDDED_RUNTIME,
    )
    .unwrap();
    // Five discarded draws on their own keys; under explicit keys they do not
    // move the returned draw, which is keyed by `key_from_seed(42)` alone.
    let draws = (0..5)
        .map(|n| {
            format!(
                "_ = uniform_like(fold_in(key_from_seed(7i64), {n}i64), copy(x), 0.0f32, 1.0f32)\n"
            )
        })
        .collect::<String>();
    let source = format!(
        "module Probe.Client\nimport Probe.Values (total)\ndef entry(x: tensor[seq, f32]) = {{ _ = print(\"entry\")\n {draws} result = total(x)\n (result, uniform_like(key_from_seed(42i64), x, 0.0f32, 1.0f32)) }}\nalias = entry\ndef main() = alias(to_tensor([7.0f32, 11.0f32]))\nout = main()\n"
    );
    prepare_eval_in_context(&context, &source).unwrap()
}

#[test]
fn spread_rank_library_formal_does_not_initialize_same_named_declaration() {
    let prepared = prepare(
        "weights = { _ = print(\"initialize\")\n to_tensor([3.0f32, 5.0f32]) }\ndef total[pre, post](weights: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(weights, seq)\n",
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
        // [05-OP-8] keyed by `key_from_seed(42)`, [0, 1), f32: the f32
        // rounding of key_ref.py's `unit(key_from_seed(42), i)`, i = 0, 1.
        let expected = json!({"dtype":"f32","bits":["3efa06fe","3e762d86"]});
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

/// The initializer is a helper's body rather than a bare block: a captured
/// block-bodied declaration whose block binds `_ = print(...)` reads as rank
/// zero inside the capturing definition, a defect that predates the key
/// switch (the retired `with seed` wrapper masked it here).
fn capture_library(initializer: &str, body: &str) -> String {
    format!(
        "baseline = to_tensor([7.0f32, 11.0f32])\ndef initialize() -> tensor[2, f32] = {{ _ = print(\"initialize\")\n {initializer} }}\nweights = initialize()\ndef total[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> (tensor[..pre, ..post, f32], tensor[f32]) = {body}\n"
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
            json!({"type":"tensor","value":{"shape":[2],"data":{"dtype":"f32","bits":["3efa06fe","3e762d86"]}}})
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
        "_ = uniform_like(key_from_seed(17i64), to_tensor([0.0f32, 0.0f32]), 0.0f32, 1.0f32)\n to_tensor([cast(floor_div(1i32, 0i32), f32), 0.0f32])",
        "(sum(x, seq), sum(weights, 0i32))",
    ));
    for _ in 0..2 {
        let error = prepared.eval_root(Default::default(), "out").unwrap_err();
        assert_eq!(error.stage, "eval");
        assert_eq!(error.errors.len(), 1);
        assert_eq!(
            error.errors[0].message,
            "numeric trap: division by zero in floor_div at i32"
        );
        assert_eq!(error.transcript, ["entry", "initialize"]);
    }
}

#[test]
fn spread_rank_call_without_a_named_axis_remains_a_check_error() {
    let source = "def total[pre, post](x: &tensor[..pre, seq, ..post, f32]) -> tensor[..pre, ..post, f32] = sum(x, seq)\nout = total(to_tensor([7.0f32, 11.0f32]))\n";
    let error = eval(request(source)).unwrap_err();
    assert_eq!(error.stage, "check");
    assert!(error.transcript.is_empty());
    assert!(
        error.errors.iter().any(|diagnostic| {
            diagnostic.kind() == chelis_vocab::DiagnosticKind::DimensionMismatch
                && diagnostic.message.contains("total")
                && diagnostic.message.contains("argument 1")
                && diagnostic.message.contains("no named `seq` axis")
                && diagnostic.expected.is_none()
                && diagnostic.got.is_none()
                && diagnostic.span.map(|span| span.offset())
                    == source.rfind("total(").map(|at| at as u64)
        }),
        "{:?}",
        error.errors
    );
}
