use chelis_tide::mcp::handle_message;
use serde_json::json;

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
const SIMPLE_DEEP: &str = r#"(def {} x (var {type: (t-tensor {} (d-lit {} 4) (t-prim {} f32))} x))
"#;

#[test]
fn initialize_and_tool_discovery_work() {
    let init = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":1,
        "method":"initialize",
        "params":{}
    }))
    .expect("initialize response");
    assert_eq!(init["result"]["serverInfo"]["name"], "chelis-tide");

    let tools = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":2,
        "method":"tools/list",
        "params":{}
    }))
    .expect("tools/list response");
    let names = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|tool| tool["name"].as_str().unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        names,
        vec![
            "chelis_check",
            "chelis_compile",
            "chelis_desugar",
            "chelis_decompile",
            "chelis_eval",
            "chelis_grad",
            "chelis_validate",
            "chelis_prove",
            "chelis_verify_spec",
            "chelis_explain_failure",
            "chelis_proof_artifact",
        ]
    );

    let check_tool = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "chelis_check")
        .expect("check tool");
    assert_eq!(
        check_tool["inputSchema"]["properties"]["source_kind"]["$ref"],
        "#/definitions/SourceKind"
    );
    assert_eq!(
        check_tool["inputSchema"]["definitions"]["SourceKind"]["enum"],
        json!(["surf", "deep"])
    );
    assert_eq!(
        check_tool["inputSchema"]["properties"]["source"]["type"],
        "string"
    );
    assert!(
        check_tool["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "source_kind")
    );
    assert!(
        check_tool["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "source")
    );

    let grad_tool = tools["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .find(|tool| tool["name"] == "chelis_grad")
        .expect("grad tool");
    assert_eq!(
        grad_tool["inputSchema"]["properties"]["source_kind"]["$ref"],
        "#/definitions/SourceKind"
    );
    assert_eq!(
        grad_tool["inputSchema"]["properties"]["output_name"]["type"],
        "string"
    );
    assert_eq!(
        grad_tool["inputSchema"]["properties"]["wrt_names"]["type"],
        "array"
    );
    assert!(
        grad_tool["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "output_name")
    );
    assert!(
        grad_tool["inputSchema"]["required"]
            .as_array()
            .unwrap()
            .iter()
            .any(|value| value == "wrt_names")
    );
}

#[test]
fn each_tool_dispatches_successfully() {
    let cases = vec![
        (
            "chelis_check",
            json!({"source_kind":"surf","source":HELLO_TENSOR}),
        ),
        (
            "chelis_compile",
            json!({"source_kind":"surf","source":MATMUL_PROGRAM,"target":"c"}),
        ),
        ("chelis_desugar", json!({"source":HELLO_TENSOR})),
        ("chelis_decompile", json!({"source":SIMPLE_DEEP})),
        (
            "chelis_eval",
            json!({
                "source_kind":"surf",
                "source":LOSS_PROGRAM,
                "bindings":{"x":{"shape":[4],"data":[1.0,2.0,3.0,4.0]}}
            }),
        ),
        (
            "chelis_grad",
            json!({
                "source_kind":"surf",
                "source":LOSS_PROGRAM,
                "output_name":"loss",
                "wrt_names":["x"]
            }),
        ),
        (
            "chelis_validate",
            json!({"mode":"surf","source":HELLO_TENSOR}),
        ),
    ];

    for (idx, (name, arguments)) in cases.into_iter().enumerate() {
        let response = handle_message(&json!({
            "jsonrpc":"2.0",
            "id":idx,
            "method":"tools/call",
            "params":{"name":name,"arguments":arguments}
        }))
        .expect("tool response");
        assert_eq!(response["result"]["isError"], false, "tool {name}");
    }
}

#[test]
fn tool_failures_preserve_structured_errors() {
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":7,
        "method":"tools/call",
        "params":{
            "name":"chelis_grad",
            "arguments":{
                "source_kind":"surf",
                "source":NON_SCALAR_PROGRAM,
                "output_name":"out",
                "wrt_names":["x"]
            }
        }
    }))
    .expect("tool response");
    assert_eq!(response["result"]["isError"], true);
    let structured = &response["result"]["structuredContent"];
    assert_eq!(structured["ok"], false);
    assert_eq!(structured["stage"], "grad");
    assert!(!structured["errors"].as_array().unwrap().is_empty());
}

#[test]
fn invalid_tool_arguments_return_mcp_error_payload() {
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":8,
        "method":"tools/call",
        "params":{"name":"chelis_eval","arguments":{"source_kind":"surf"}}
    }))
    .expect("tool response");
    assert_eq!(response["result"]["isError"], true);
    assert_eq!(response["result"]["structuredContent"]["stage"], "mcp");
}
