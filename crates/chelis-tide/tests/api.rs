use std::fs;

use assert_cmd::Command;
use axum::{Router, body::Body, http::Request};
use chelis_tide::http::router;
use http_body_util::BodyExt;
use serde_json::{Value, json};
use tempfile::tempdir;
use tower::ServiceExt;

const HELLO_TENSOR: &str = include_str!("../../../examples/hello_tensor.ch");
const MATMUL_PROGRAM: &str = r#"let a = (a : tensor[2, 3, f32])
let b = (b : tensor[3, 4, f32])
let out = (matmul(a, b) : tensor[2, 4, f32])
"#;
const LOSS_PROGRAM: &str = r#"let x = (x : tensor[4, f32])
let loss = (mean(x, 0) : tensor[f32])
"#;
const NON_SCALAR_PROGRAM: &str = r#"let x = (x : tensor[4, f32])
let out = (add(x, x) : tensor[4, f32])
"#;

async fn post_json(app: Router, path: &str, value: Value) -> (u16, Value) {
    let response = app
        .oneshot(
            Request::builder()
                .method("POST")
                .uri(path)
                .header("content-type", "application/json")
                .body(Body::from(value.to_string()))
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
    assert!(
        ok["result"]["deep_text"]
            .as_str()
            .unwrap()
            .contains("(module {}")
    );

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
            "source":"def f(xs: tensor[batch, features, f32]): tensor[batch, features, f32] = xs\n",
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
            "bindings":{"x":{"shape":[4],"data":[1.0,2.0,3.0,4.0]}}
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
    assert_eq!(loss_root["value"]["value"]["data"][0], 2.5);

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
            "source":"let value = debug(string_concat(\"ok-\", to_string(string_len(\"hé\"))))\n",
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
let keys: List[string] = ["alpha", "beta"]
let ids: List[int64] = [cast(1, int64), cast(2, int64)]
let pairs = zip(keys, ids)
let enumerated = enumerate(keys)
let vocab: Dict[string, int64] = dict_of(pairs)
let entries = dict_entries(vocab)
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
    assert_eq!(vocab["value"]["entries"][0]["value"]["type"], "int64");
    let entries = roots
        .iter()
        .find(|root| root["name"] == "entries")
        .expect("entries root");
    assert_eq!(entries["value"]["type"], "list");
    assert_eq!(entries["value"]["value"][0]["type"], "tuple");
    assert_eq!(entries["value"]["value"][0]["value"][0]["type"], "string");
    assert_eq!(entries["value"]["value"][0]["value"][1]["type"], "int64");
}

#[tokio::test]
async fn eval_endpoint_returns_list_from_tensor_bridge() {
    let source = r#"
let x = (x : tensor[4, f32])
let items = to_list(x)
"#;
    let (_, ok) = post_json(
        router(),
        "/eval",
        json!({
            "source_kind":"surf",
            "source":source,
            "bindings":{"x":{"shape":[4],"data":[1.0,2.0,3.0,4.0]}}
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
    assert_eq!(items["value"]["value"][0]["type"], "float64");
    assert_eq!(items["value"]["value"][3]["value"], 4.0);
}

#[tokio::test]
async fn eval_endpoint_returns_sequence_and_dict_helper_values() {
    let source = r#"
let xs: List[int64] = [cast(1, int64), cast(2, int64), cast(3, int64)]
let prefix = take(xs, cast(2, int64))
let suffix = drop(xs, cast(1, int64))
let groups = chunk(xs, cast(2, int64))
let scanned = scan(fn (acc: int64, x: int64) -> add(acc, x), cast(0, int64), xs)
let buckets = partition(fn (x: int64) -> gt(x, cast(1, int64)), xs)
let exploded = flat_map(fn (x: int64) -> [x, add(x, cast(10, int64))], take(xs, cast(2, int64)))
let flattened = flatten([[cast(1, int64)], [cast(2, int64), cast(3, int64)]])
let base: Dict[string, int64] = dict_of([("alpha", cast(1, int64))])
let key_count = len(base)
let extended = dict_insert(base, "beta", cast(2, int64))
let merged = dict_merge(extended, dict_of([("beta", cast(20, int64)), ("gamma", cast(3, int64))]))
let trimmed = dict_remove(merged, "gamma")
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
    assert_eq!(prefix["value"]["value"][1]["value"], 2);
    let suffix = roots
        .iter()
        .find(|root| root["name"] == "suffix")
        .expect("suffix root");
    assert_eq!(suffix["value"]["type"], "list");
    assert_eq!(suffix["value"]["value"][0]["value"], 2);
    let groups = roots
        .iter()
        .find(|root| root["name"] == "groups")
        .expect("groups root");
    assert_eq!(groups["value"]["type"], "list");
    assert_eq!(groups["value"]["value"][0]["type"], "list");
    assert_eq!(groups["value"]["value"][1]["value"][0]["value"], 3);
    let scanned = roots
        .iter()
        .find(|root| root["name"] == "scanned")
        .expect("scanned root");
    assert_eq!(scanned["value"]["type"], "list");
    assert_eq!(scanned["value"]["value"][2]["value"], 6);
    let buckets = roots
        .iter()
        .find(|root| root["name"] == "buckets")
        .expect("buckets root");
    assert_eq!(buckets["value"]["type"], "tuple");
    assert_eq!(buckets["value"]["value"][0]["type"], "list");
    assert_eq!(buckets["value"]["value"][0]["value"][0]["value"], 2);
    assert_eq!(buckets["value"]["value"][1]["type"], "list");
    assert_eq!(buckets["value"]["value"][1]["value"][0]["value"], 1);
    let exploded = roots
        .iter()
        .find(|root| root["name"] == "exploded")
        .expect("exploded root");
    assert_eq!(exploded["value"]["type"], "list");
    assert_eq!(exploded["value"]["value"][1]["value"], 11);
    let flattened = roots
        .iter()
        .find(|root| root["name"] == "flattened")
        .expect("flattened root");
    assert_eq!(flattened["value"]["type"], "list");
    assert_eq!(flattened["value"]["value"][2]["value"], 3);
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
    assert_eq!(beta["value"]["type"], "int64");
    assert_eq!(beta["value"]["value"], 20);
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
    assert_eq!(key_count["value"]["value"], 1);
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
    assert!(
        ok["result"]["surf_text"]
            .as_str()
            .unwrap()
            .contains("def main")
    );

    let (_, bad) = post_json(router(), "/decompile", json!({"source":"(def {}"})).await;
    assert!(!bad["ok"].as_bool().unwrap());
    assert_eq!(bad["stage"], "parse");
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
            "bindings":{"x":{"shape":[4],"data":[1.0,2.0,3.0,4.0]}}
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
