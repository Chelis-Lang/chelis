use std::fs;

use assert_cmd::Command;
use axum::{Router, body::Body, http::Request};
use chelis_tide::http::router;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tempfile::tempdir;
use tower::ServiceExt;

mod replace_fixtures;

const HELLO_TENSOR: &str = include_str!("../../../examples/hello_tensor.ch");
const MATMUL_PROGRAM: &str = r#"a = (a : tensor[2, 3, f32])
b = (b : tensor[3, 4, f32])
out = (matmul(a, b) : tensor[2, 4, f32])
"#;
const LOSS_PROGRAM: &str = r#"x = (x : tensor[4, f32])
loss = (mean(x, 0) : tensor[f32])
"#;
const NON_SCALAR_PROGRAM: &str = r#"x = (x : tensor[4, f32])
out = (add(copy(x), x) : tensor[4, f32])
"#;
const DEEP_AUTHORING_RENAME_MODULE: &str = r#"(module {}
  tide.rename
  (export {} first second)
  (defsig {} first (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {} first (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))
  (defsig {} second (t-fn {eff: (effects {})} (t-prim {} f32) (t-prim {} f32)))
  (def {} second (fn {} (params {} (x {type: (t-prim {} f32)})) (app {} (var {} first) (var {} x)))))
"#;

async fn post_json(app: Router, path: &str, value: Value) -> (u16, Value) {
    post_raw_json(app, path, value.to_string()).await
}

async fn post_raw_json(app: Router, path: &str, body: impl Into<String>) -> (u16, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(body.into()))
                .expect("request"),
        )
        .await
        .expect("response");
    let status = response.status().as_u16();
    let body = response
        .into_body()
        .collect()
        .await
        .expect("body")
        .to_bytes();
    let json = serde_json::from_slice(&body).expect("json body");
    (status, json)
}

#[tokio::test]
async fn parse_endpoint_returns_ast_and_parse_errors() {
    let (status, ok) = post_json(
        router(),
        "/parse",
        json!({"source_kind":"surf","source":HELLO_TENSOR}),
    )
    .await;
    assert_eq!(status, 200);
    assert!(ok["ok"].as_bool().unwrap());
    assert!(!ok["result"]["surf_ast"].as_array().unwrap().is_empty());

    let (_, bad) = post_json(
        router(),
        "/parse",
        json!({"source_kind":"deep","source":"(def {}"}),
    )
    .await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "parse");
}

#[tokio::test]
async fn desugar_endpoint_returns_deep_and_rejects_bad_surf() {
    let (_, ok) = post_json(router(), "/desugar", json!({"source":HELLO_TENSOR})).await;
    assert!(ok["ok"].as_bool().unwrap());
    let deep_text = ok["result"]["deep_text"].as_str().unwrap();
    assert!(deep_text.contains("(module {surf_path:"), "{deep_text}");

    let (_, bad) = post_json(
        router(),
        "/desugar",
        json!({"source":"def f(x) = a == b == c"}),
    )
    .await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "parse");
}

#[tokio::test]
async fn check_endpoint_matches_cli_and_surfaces_errors() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("hello_tensor.ch");
    fs::write(&path, HELLO_TENSOR).expect("write source");

    let cli_output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let cli_json: Value = serde_json::from_slice(&cli_output).expect("cli json");

    let (_, api) = post_json(
        router(),
        "/check",
        json!({"source_kind":"surf","source":HELLO_TENSOR}),
    )
    .await;
    assert!(api["ok"].as_bool().unwrap());
    assert_eq!(
        api["result"]["score"].as_f64().unwrap(),
        cli_json["score"].as_f64().unwrap()
    );
    assert_eq!(api["result"]["typed_nodes"], cli_json["typed_nodes"]);
    assert_eq!(
        api["result"]["unresolved_names"],
        cli_json["unresolved_names"]
    );

    let (_, bad) = post_json(
        router(),
        "/check",
        json!({"source_kind":"surf","source":"def f() = 1 + true\n"}),
    )
    .await;
    assert!(bad["ok"].as_bool().unwrap());
    assert!(bad["result"]["score"].as_f64().unwrap() < 1.0);
    assert!(!bad["result"]["errors"].as_array().unwrap().is_empty());
}

#[tokio::test]
async fn lower_endpoint_returns_wire_dag_and_check_failures() {
    let (_, ok) = post_json(
        router(),
        "/lower",
        json!({"source_kind":"surf","source":MATMUL_PROGRAM}),
    )
    .await;
    assert!(ok["ok"].as_bool().unwrap());
    assert!(!ok["result"]["dag"]["nodes"].as_array().unwrap().is_empty());
    assert!(ok["result"]["named_roots"]["out"].as_u64().is_some());

    let (_, bad) = post_json(
        router(),
        "/lower",
        json!({"source_kind":"surf","source":"def f() = 1 + true\n"}),
    )
    .await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "check");
}

#[tokio::test]
async fn lower_and_grad_responses_carry_validated_schema_version() {
    // WI-2 validate-on-consume (WS-5 Part A): the WireDag the server hands
    // back across the process edge is validated at the boundary, so the DAG
    // the client receives carries the supported `schema_version` and the
    // response is never a `schema`-stage failure on the happy path.
    let supported = chelis_tide::schema::WIRE_DAG_SCHEMA_VERSION as u64;

    let (_, lowered) = post_json(
        router(),
        "/lower",
        json!({"source_kind":"surf","source":MATMUL_PROGRAM}),
    )
    .await;
    assert!(lowered["ok"].as_bool().unwrap());
    assert_eq!(
        lowered["result"]["dag"]["schema_version"].as_u64(),
        Some(supported),
        "lowered DAG crosses the boundary stamped at the supported version"
    );
    assert_ne!(
        lowered["stage"].as_str(),
        Some("schema"),
        "the happy path never surfaces a schema-stage failure"
    );

    let (_, graded) = post_json(
        router(),
        "/grad",
        json!({
            "source_kind":"surf",
            "source":LOSS_PROGRAM,
            "output_name":"loss",
            "wrt_names":["x"]
        }),
    )
    .await;
    assert!(graded["ok"].as_bool().unwrap());
    assert_eq!(
        graded["result"]["dag"]["schema_version"].as_u64(),
        Some(supported),
        "gradient DAG crosses the boundary stamped at the supported version"
    );
    assert_ne!(graded["stage"].as_str(), Some("schema"));
}

#[tokio::test]
async fn compile_endpoint_returns_generated_files_and_backend_errors() {
    let (_, c_ok) = post_json(
        router(),
        "/compile",
        json!({"source_kind":"surf","source":MATMUL_PROGRAM,"target":"c"}),
    )
    .await;
    assert!(c_ok["ok"].as_bool().unwrap());
    assert!(
        c_ok["result"]["files"]
            .as_array()
            .unwrap()
            .iter()
            .any(|file| file["path"] == "chelis_main.c")
    );

    let (_, hip_ok) = post_json(
        router(),
        "/compile",
        json!({"source_kind":"surf","source":MATMUL_PROGRAM,"target":"hip"}),
    )
    .await;
    assert!(hip_ok["ok"].as_bool().unwrap());
    assert!(
        hip_ok["result"]["peak_device_bytes_estimate"]
            .as_u64()
            .is_some()
    );

    let (_, bad) = post_json(
        router(),
        "/compile",
        json!({
            "source_kind":"surf",
            "source":"def f(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] = xs\n",
            "target":"hip"
        }),
    )
    .await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "compile");
}

#[tokio::test]
async fn eval_endpoint_uses_named_bindings_and_rejects_missing_inputs() {
    let (_, ok) = post_json(
        router(),
        "/eval",
        json!({
            "source_kind":"surf",
            "source":LOSS_PROGRAM,
            "bindings":{"x":{"shape":[4],"data":{"dtype":"f32","bits":["3f800000","40000000","40400000","40800000"]}}}
        }),
    )
    .await;
    assert!(ok["ok"].as_bool().unwrap());
    let loss_root = ok["result"]["roots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|root| root["name"] == "loss")
        .expect("loss root");
    assert_eq!(loss_root["value"]["type"], "tensor");
    assert_eq!(loss_root["value"]["value"]["data"]["bits"][0], "40200000");

    let (_, bad) = post_json(
        router(),
        "/eval",
        json!({"source_kind":"surf","source":LOSS_PROGRAM,"bindings":{}}),
    )
    .await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "eval");
}

#[tokio::test]
async fn eval_endpoint_returns_host_values_and_transcript() {
    let (_, ok) = post_json(
        router(),
        "/eval",
        json!({
            "source_kind":"surf",
            "source":"value = debug(string_concat(\"ok-\", to_string(string_len(\"hé\"))))\n",
            "bindings":{}
        }),
    )
    .await;
    assert!(ok["ok"].as_bool().unwrap());
    assert_eq!(ok["result"]["transcript"][0], "ok-2");
    let value_root = ok["result"]["roots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|root| root["name"] == "value")
        .expect("value root");
    assert_eq!(value_root["value"]["type"], "string");
    assert_eq!(value_root["value"]["value"], "ok-2");
}

#[tokio::test]
async fn eval_endpoint_returns_dict_and_tuple_collection_values() {
    let source = r#"
keys: List[string] = ["alpha", "beta"]
ids: List[int64] = [cast(1, int64), cast(2, int64)]
pairs = zip(keys, ids)
enumerated = enumerate(keys)
vocab: Dict[string, int64] = dict_of(pairs)
entries = dict_entries(vocab)
"#;
    let (_, ok) = post_json(
        router(),
        "/eval",
        json!({"source_kind":"surf","source":source,"bindings":{}}),
    )
    .await;
    assert!(ok["ok"].as_bool().unwrap());
    let roots = ok["result"]["roots"].as_array().expect("roots");
    let pairs = roots
        .iter()
        .find(|root| root["name"] == "pairs")
        .expect("pairs root");
    assert_eq!(pairs["value"]["type"], "list");
    assert_eq!(pairs["value"]["value"][0]["type"], "tuple");
    let enumerated = roots
        .iter()
        .find(|root| root["name"] == "enumerated")
        .expect("enumerated root");
    assert_eq!(enumerated["value"]["type"], "list");
    assert_eq!(enumerated["value"]["value"][0]["type"], "tuple");
    let vocab = roots
        .iter()
        .find(|root| root["name"] == "vocab")
        .expect("vocab root");
    assert_eq!(vocab["value"]["type"], "dict");
    assert_eq!(vocab["value"]["entries"][0]["key"]["type"], "string");
    assert_eq!(vocab["value"]["entries"][0]["value"]["type"], "scalar");
    assert_eq!(
        vocab["value"]["entries"][0]["value"]["value"]["dtype"],
        "int64"
    );
    let entries = roots
        .iter()
        .find(|root| root["name"] == "entries")
        .expect("entries root");
    assert_eq!(entries["value"]["type"], "list");
    assert_eq!(entries["value"]["value"][0]["type"], "tuple");
    assert_eq!(entries["value"]["value"][0]["value"][0]["type"], "string");
    assert_eq!(entries["value"]["value"][0]["value"][1]["type"], "scalar");
    assert_eq!(
        entries["value"]["value"][0]["value"][1]["value"]["dtype"],
        "int64"
    );
}

#[tokio::test]
async fn eval_endpoint_returns_list_from_tensor_bridge() {
    let source = r#"
x = (x : tensor[4, f32])
items = to_list(x)
"#;
    let (_, ok) = post_json(
        router(),
        "/eval",
        json!({
            "source_kind":"surf",
            "source":source,
            "bindings":{"x":{"shape":[4],"data":{"dtype":"f32","bits":["3f800000","40000000","40400000","40800000"]}}}
        }),
    )
    .await;
    assert!(ok["ok"].as_bool().unwrap());
    let items = ok["result"]["roots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|root| root["name"] == "items")
        .expect("items root");
    assert_eq!(items["value"]["type"], "list");
    let values = items["value"]["value"].as_array().expect("list values");
    assert_eq!(values.len(), 4);
    assert!(
        values
            .iter()
            .all(|value| value["type"] == "scalar" && value["value"]["dtype"] == "f32")
    );
    assert_eq!(
        values
            .iter()
            .map(|value| value["value"]["bits"].as_str().expect("f32 bits"))
            .collect::<Vec<_>>(),
        vec!["3f800000", "40000000", "40400000", "40800000"]
    );
}

#[tokio::test]
async fn eval_endpoint_returns_sequence_and_dict_helper_values() {
    let source = r#"
xs: List[int64] = [cast(1, int64), cast(2, int64), cast(3, int64)]
prefix = take(xs, cast(2, int64))
suffix = drop(xs, cast(1, int64))
groups = chunk(xs, cast(2, int64))
scanned = scan(fn (acc: int64, x: int64) -> add(acc, x), cast(0, int64), xs)
buckets = partition(fn (x: int64) -> gt(x, cast(1, int64)), xs)
exploded = flat_map(fn (x: int64) -> [x, add(x, cast(10, int64))], take(xs, cast(2, int64)))
flattened = flatten([[cast(1, int64)], [cast(2, int64), cast(3, int64)]])
base: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
key_count = len(base)
extended = dict_insert(base, "beta", cast(2, int64))
merged = dict_merge(extended, dict_of([("beta", cast(20, int64)), ("gamma", cast(3, int64))]))
trimmed = dict_remove(merged, "gamma")
"#;
    let (_, ok) = post_json(
        router(),
        "/eval",
        json!({"source_kind":"surf","source":source,"bindings":{}}),
    )
    .await;
    assert!(ok["ok"].as_bool().unwrap());
    let roots = ok["result"]["roots"].as_array().expect("roots");
    let prefix = roots
        .iter()
        .find(|root| root["name"] == "prefix")
        .expect("prefix root");
    assert_eq!(prefix["value"]["type"], "list");
    assert_eq!(prefix["value"]["value"].as_array().unwrap().len(), 2);
    assert_eq!(prefix["value"]["value"][1]["value"]["value"], 2);
    let suffix = roots
        .iter()
        .find(|root| root["name"] == "suffix")
        .expect("suffix root");
    assert_eq!(suffix["value"]["type"], "list");
    assert_eq!(suffix["value"]["value"][0]["value"]["value"], 2);
    let groups = roots
        .iter()
        .find(|root| root["name"] == "groups")
        .expect("groups root");
    assert_eq!(groups["value"]["type"], "list");
    assert_eq!(groups["value"]["value"][0]["type"], "list");
    assert_eq!(groups["value"]["value"][1]["value"][0]["value"]["value"], 3);
    let scanned = roots
        .iter()
        .find(|root| root["name"] == "scanned")
        .expect("scanned root");
    assert_eq!(scanned["value"]["type"], "list");
    assert_eq!(scanned["value"]["value"][2]["value"]["value"], 6);
    // Top-level tuple-typed bindings are expanded into per-field roots
    // (`<name>.0`, `<name>.1`, …) by `extend_root_names_from_value`, so
    // the `buckets` tuple-typed binding surfaces as two list roots.
    let buckets_pass = roots
        .iter()
        .find(|root| root["name"] == "buckets.0")
        .expect("buckets.0 root");
    assert_eq!(buckets_pass["value"]["type"], "list");
    assert_eq!(buckets_pass["value"]["value"][0]["value"]["value"], 2);
    let buckets_fail = roots
        .iter()
        .find(|root| root["name"] == "buckets.1")
        .expect("buckets.1 root");
    assert_eq!(buckets_fail["value"]["type"], "list");
    assert_eq!(buckets_fail["value"]["value"][0]["value"]["value"], 1);
    let exploded = roots
        .iter()
        .find(|root| root["name"] == "exploded")
        .expect("exploded root");
    assert_eq!(exploded["value"]["type"], "list");
    assert_eq!(exploded["value"]["value"][1]["value"]["value"], 11);
    let flattened = roots
        .iter()
        .find(|root| root["name"] == "flattened")
        .expect("flattened root");
    assert_eq!(flattened["value"]["type"], "list");
    assert_eq!(flattened["value"]["value"][2]["value"]["value"], 3);
    let merged = roots
        .iter()
        .find(|root| root["name"] == "merged")
        .expect("merged root");
    assert_eq!(merged["value"]["type"], "dict");
    assert_eq!(merged["value"]["entries"].as_array().unwrap().len(), 3);
    let beta = merged["value"]["entries"]
        .as_array()
        .unwrap()
        .iter()
        .find(|entry| entry["key"]["value"] == "beta")
        .expect("beta entry");
    assert_eq!(beta["value"]["type"], "scalar");
    assert_eq!(beta["value"]["value"]["dtype"], "int64");
    assert_eq!(beta["value"]["value"]["value"], 20);
    let trimmed = roots
        .iter()
        .find(|root| root["name"] == "trimmed")
        .expect("trimmed root");
    assert_eq!(trimmed["value"]["type"], "dict");
    assert_eq!(trimmed["value"]["entries"].as_array().unwrap().len(), 2);
    let key_count = roots
        .iter()
        .find(|root| root["name"] == "key_count")
        .expect("key_count root");
    assert_eq!(key_count["value"]["value"]["value"], 1);
}

#[tokio::test]
async fn eval_endpoint_rejects_mapped_file_roots_on_machine_surface() {
    let dir = tempdir().expect("tempdir");
    let data = dir.path().join("dataset.txt");
    fs::write(&data, "alpha\nbeta\n").expect("write dataset");
    let source = format!(
        "mapped = mmap_file(\"{}\")\n",
        data.to_str()
            .expect("utf8 path")
            .replace('\\', "\\\\")
            .replace('"', "\\\"")
    );

    let (_, bad) = post_json(
        router(),
        "/eval",
        json!({"source_kind":"surf","source":source,"bindings":{}}),
    )
    .await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "eval");
    assert!(
        bad["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|entry| entry["message"]
                .as_str()
                .is_some_and(|msg| msg.contains("MappedFile values are not serializable"))),
        "expected mapped-file serialization diagnostic, got {bad}"
    );
}

#[tokio::test]
async fn grad_endpoint_returns_gradient_dag_and_rejects_bad_outputs() {
    let (_, ok) = post_json(
        router(),
        "/grad",
        json!({
            "source_kind":"surf",
            "source":LOSS_PROGRAM,
            "output_name":"loss",
            "wrt_names":["x"]
        }),
    )
    .await;
    assert!(ok["ok"].as_bool().unwrap());
    assert!(ok["result"]["grad_nodes_by_name"]["x"].as_u64().is_some());
    assert!(
        ok["result"]["forward_nodes_by_name"]["loss"]
            .as_u64()
            .is_some()
    );

    let (_, bad) = post_json(
        router(),
        "/grad",
        json!({
            "source_kind":"surf",
            "source":NON_SCALAR_PROGRAM,
            "output_name":"out",
            "wrt_names":["x"]
        }),
    )
    .await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "grad");
}

#[tokio::test]
async fn validate_endpoint_reports_validity_and_failures() {
    let (_, ok) = post_json(
        router(),
        "/validate",
        json!({"mode":"surf","source":HELLO_TENSOR}),
    )
    .await;
    assert!(ok["ok"].as_bool().unwrap());
    assert_eq!(ok["result"]["valid"], true);

    let (_, bad) = post_json(
        router(),
        "/validate",
        json!({"mode":"deep","source":"(mystery {} x)\n"}),
    )
    .await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "validate");
}

#[tokio::test]
async fn decompile_endpoint_round_trips_and_rejects_bad_deep() {
    let (_, desugared) = post_json(router(), "/desugar", json!({"source":HELLO_TENSOR})).await;
    let deep_text = desugared["result"]["deep_text"].as_str().unwrap();

    let (_, ok) = post_json(router(), "/decompile", json!({"source":deep_text})).await;
    assert!(ok["ok"].as_bool().unwrap());
    let surf_text = ok["result"]["surf_text"].as_str().unwrap();
    assert!(surf_text.contains("def main"));
    let reparsed = chelis_surf::parser::parse_str(surf_text)
        .expect("decompile endpoint returns parseable Surf");
    assert_eq!(
        chelis_surf::format::format_program(&reparsed),
        surf_text,
        "decompile endpoint returns formatter-canonical Surf"
    );

    let (_, bad) = post_json(router(), "/decompile", json!({"source":"(def {}"})).await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "parse");
}

#[tokio::test]
async fn replace_function_body_endpoint_accepts_well_typed_replacement() {
    let (status, ok) = post_json(
        router(),
        "/replace_function_body",
        json!({
            "module": replace_fixtures::TENSOR_DEEP,
            "function_name": "passthrough",
            "new_body": replace_fixtures::TENSOR_WELL_TYPED_BODY,
        }),
    )
    .await;
    assert_eq!(status, 200);
    assert!(ok["ok"].as_bool().unwrap(), "replacement should pass: {ok}");
    let module_deep = ok["result"]["module_deep"]
        .as_str()
        .expect("module_deep string");
    let reparsed =
        chelis_deep::parser::parse_str_strict(module_deep).expect("module_deep reparses");
    assert_eq!(
        chelis_deep::printer::print_canonical(&reparsed),
        module_deep
    );
    let check = chelis_compiler_api::compiler::check(chelis_compiler_api::schema::CheckRequest {
        source_kind: chelis_compiler_api::schema::SourceKind::Deep,
        source: module_deep.to_string(),
    })
    .expect("rewritten module checks");
    assert!(check.errors.is_empty(), "rewritten module is clean");
}

async fn assert_http_replace_rejection(new_body: &str, expected_stage: &str, expected_kind: &str) {
    let (status, bad) = post_json(
        router(),
        "/replace_function_body",
        json!({
            "module": replace_fixtures::TENSOR_DEEP,
            "function_name": "passthrough",
            "new_body": new_body,
        }),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(bad["ok"], false, "replacement should reject: {bad}");
    assert_eq!(bad["stage"], expected_stage);
    assert!(
        bad.get("result").is_none(),
        "failed replacement must not include result: {bad}"
    );
    let errors = bad["errors"].as_array().expect("errors array");
    assert!(!errors.is_empty());
    assert_eq!(errors[0]["kind"], expected_kind);
}

async fn assert_http_add_rejection(new_decls: &str, expected_stage: &str, expected_kind: &str) {
    let (status, bad) = post_json(
        router(),
        "/add_function",
        json!({
            "module": replace_fixtures::TENSOR_DEEP,
            "new_decls": new_decls,
        }),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(bad["ok"], false, "add_function should reject: {bad}");
    assert_eq!(bad["stage"], expected_stage);
    assert!(
        bad.get("result").is_none(),
        "failed add_function must not include result: {bad}"
    );
    let errors = bad["errors"].as_array().expect("errors array");
    assert!(!errors.is_empty());
    assert_eq!(errors[0]["kind"], expected_kind);
}

fn assert_invalid_http_request_envelope(bad: &Value) {
    assert_eq!(bad["ok"], false, "request should reject: {bad}");
    assert_eq!(bad["stage"], "http");
    assert!(
        bad.get("result").is_none(),
        "invalid HTTP authoring request must not include result: {bad}"
    );
    let errors = bad["errors"].as_array().expect("errors array");
    assert!(!errors.is_empty());
    assert_eq!(errors[0]["kind"], "invalid_request");
}

#[tokio::test]
async fn replace_function_body_endpoint_rejects_malformed_cast_body() {
    assert_http_replace_rejection(
        replace_fixtures::LIVE_MALFORMED_CAST_BODY,
        "replace",
        "deep_parse_error",
    )
    .await;
}

#[tokio::test]
async fn replace_function_body_endpoint_locks_effect_and_linearity_envelopes() {
    assert_http_replace_rejection(
        replace_fixtures::TENSOR_EFFECTING_BODY,
        "effects",
        "effect_error",
    )
    .await;
    assert_http_replace_rejection(
        replace_fixtures::TENSOR_LINEARITY_BODY,
        "linearity",
        "linearity_error",
    )
    .await;
}

#[tokio::test]
async fn add_function_endpoint_accepts_well_typed_function_bundle() {
    let (status, ok) = post_json(
        router(),
        "/add_function",
        json!({
            "module": replace_fixtures::TENSOR_DEEP,
            "new_decls": replace_fixtures::ADD_TENSOR_IDENTITY,
            "insert_after_function": "passthrough",
        }),
    )
    .await;
    assert_eq!(status, 200);
    assert!(
        ok["ok"].as_bool().unwrap(),
        "add_function should pass: {ok}"
    );
    assert!(
        ok["result"]["added_def_deep"]
            .as_str()
            .expect("added_def_deep")
            .contains("added_passthrough")
    );
    let module_deep = ok["result"]["module_deep"]
        .as_str()
        .expect("module_deep string");
    let reparsed =
        chelis_deep::parser::parse_str_strict(module_deep).expect("module_deep reparses");
    assert_eq!(
        chelis_deep::printer::print_canonical(&reparsed),
        module_deep
    );
}

#[tokio::test]
async fn add_function_endpoint_rejects_structured_failure_without_result() {
    assert_http_add_rejection(
        replace_fixtures::ADD_DECL_SHAPE_ERROR,
        "add-function",
        "deep_decl_error",
    )
    .await;
}

#[tokio::test]
async fn add_function_endpoint_locks_effect_and_linearity_envelopes() {
    assert_http_add_rejection(
        replace_fixtures::ADD_TENSOR_EFFECTING,
        "effects",
        "effect_error",
    )
    .await;
    assert_http_add_rejection(
        replace_fixtures::ADD_TENSOR_LINEARITY,
        "linearity",
        "linearity_error",
    )
    .await;
}

#[tokio::test]
async fn deep_query_and_rename_http_endpoints_lock_preimage_contract() {
    let (_, outline) = post_json(
        router(),
        "/deep_outline",
        json!({"module": DEEP_AUTHORING_RENAME_MODULE}),
    )
    .await;
    assert_eq!(outline["ok"], true);
    let functions = outline["result"]["functions"].as_array().unwrap();
    let first = functions
        .iter()
        .find(|function| function["name"] == "first")
        .expect("first outline");
    let preimage = first["preimage_sha256"].as_str().unwrap();
    assert_eq!(preimage.len(), 64);

    let (_, graph) = post_json(
        router(),
        "/deep_call_graph",
        json!({"module": DEEP_AUTHORING_RENAME_MODULE}),
    )
    .await;
    assert_eq!(graph["ok"], true);
    let edges = graph["result"]["edges"].as_array().unwrap();
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0]["caller"], "tide.rename.second");
    assert_eq!(edges[0]["callee"], "tide.rename.first");

    let (_, stale) = post_json(
        router(),
        "/rename",
        json!({
            "module": DEEP_AUTHORING_RENAME_MODULE,
            "function_name": "first",
            "new_name": "renamed",
            "preimage_sha256": "0".repeat(64),
        }),
    )
    .await;
    assert_eq!(stale["ok"], false);
    assert_eq!(stale["stage"], "preimage");
    assert!(stale.get("result").is_none());

    let (_, renamed) = post_json(
        router(),
        "/rename",
        json!({
            "module": DEEP_AUTHORING_RENAME_MODULE,
            "function_name": "first",
            "new_name": "renamed",
            "preimage_sha256": preimage,
        }),
    )
    .await;
    assert_eq!(renamed["ok"], true);
    assert_eq!(renamed["result"]["renamed_references"], 1);
    assert!(
        renamed["result"]["module_deep"]
            .as_str()
            .unwrap()
            .contains("(app {} (var {} renamed) (var {} x))")
    );
}

#[tokio::test]
async fn authoring_endpoints_surface_wire_shape_failures_as_structured_json() {
    let (status, missing_body) = post_json(
        router(),
        "/replace_function_body",
        json!({
            "module": replace_fixtures::TENSOR_DEEP,
            "function_name": "passthrough",
        }),
    )
    .await;
    assert_eq!(status, 200);
    assert_invalid_http_request_envelope(&missing_body);

    let (status, missing_decls) = post_json(
        router(),
        "/add_function",
        json!({
            "module": replace_fixtures::TENSOR_DEEP,
        }),
    )
    .await;
    assert_eq!(status, 200);
    assert_invalid_http_request_envelope(&missing_decls);

    let (status, malformed_json) = post_raw_json(router(), "/add_function", "{").await;
    assert_eq!(status, 200);
    assert_invalid_http_request_envelope(&malformed_json);
}

#[tokio::test]
async fn batch_endpoint_preserves_order_for_mixed_results() {
    let (_, batch) = post_json(
        router(),
        "/batch",
        json!({
            "requests":[
                {"kind":"check","source_kind":"surf","source":HELLO_TENSOR},
                {"kind":"grad","source_kind":"surf","source":NON_SCALAR_PROGRAM,"output_name":"out","wrt_names":["x"]}
            ]
        }),
    )
    .await;
    let results = batch["results"].as_array().unwrap();
    assert_eq!(results.len(), 2);
    assert_eq!(results[0]["kind"], "check");
    assert!(results[0]["ok"].as_bool().unwrap());
    assert_eq!(results[1]["kind"], "grad");
    assert!(!results[1]["ok"].as_bool().unwrap());
}

#[tokio::test]
async fn router_handles_concurrent_requests() {
    let app = router();
    let first = tokio::spawn(post_json(
        app.clone(),
        "/check",
        json!({"source_kind":"surf","source":HELLO_TENSOR}),
    ));
    let second = tokio::spawn(post_json(
        app,
        "/eval",
        json!({
            "source_kind":"surf",
            "source":LOSS_PROGRAM,
            "bindings":{"x":{"shape":[4],"data":{"dtype":"f32","bits":["3f800000","40000000","40400000","40800000"]}}}
        }),
    ));
    let (first, second) = tokio::join!(first, second);
    assert!(first.unwrap().1["ok"].as_bool().unwrap());
    assert!(second.unwrap().1["ok"].as_bool().unwrap());
}

#[tokio::test]
async fn malformed_json_returns_bad_request() {
    let response = router()
        .oneshot(
            Request::builder()
                .method("POST")
                .uri("/check")
                .header("content-type", "application/json")
                .body(Body::from("{"))
                .expect("request"),
        )
        .await
        .expect("response");
    assert_eq!(response.status().as_u16(), 400);
}

#[tokio::test]
async fn eval_endpoint_rejects_legacy_and_malformed_storage_before_execution() {
    for data in [
        json!({"dtype":"f32","values":[1.0,2.0,3.0,4.0]}),
        json!({"dtype":"f32","bits":["3f80000","40000000","40400000","40800000"]}),
    ] {
        let response = router()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/eval")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        json!({
                            "source_kind":"surf", "source":LOSS_PROGRAM,
                            "bindings":{"x":{"shape":[4],"data":data}}
                        })
                        .to_string(),
                    ))
                    .expect("request"),
            )
            .await
            .expect("response");
        assert_eq!(response.status(), 422);
        let body = response
            .into_body()
            .collect()
            .await
            .expect("body")
            .to_bytes();
        let message = String::from_utf8(body.to_vec()).expect("error text");
        assert!(message.contains("bindings"), "{message}");
    }
}

#[tokio::test]
async fn eval_endpoint_rejects_valid_storage_with_wrong_binding_dtype() {
    let (status, error) = post_json(
        router(),
        "/eval",
        json!({
            "source_kind":"surf", "source":LOSS_PROGRAM,
            "bindings":{"x":{"shape":[4],"data":{
                "dtype":"f64",
                "bits":["3ff0000000000000","4000000000000000","4008000000000000","4010000000000000"]
            }}}
        }),
    )
    .await;
    assert_eq!(status, 200);
    assert_eq!(error["ok"], false, "{error}");
    let diagnostics = error["errors"].to_string();
    assert!(
        diagnostics.contains("f32") && diagnostics.contains("f64"),
        "{error}"
    );
}
