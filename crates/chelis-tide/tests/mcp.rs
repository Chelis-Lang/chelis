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

const OPAQUE_MODULE: &str = "module Stats.Prob
export (probability)
@opaque
@invariant(p) p.value >= 0.0 && p.value <= 1.0
type Probability =
  | Probability { value: f32 }
def probability(x: f32) -> Option[Probability] =
  if x >= 0.0 && x <= 1.0 then Some(Probability { value: x }) else None
";

/// RFC D-PARITY: a prove invoked through the tide MCP tool runs the SAME
/// derived obligations as the CLI on the same module. We assert tide's
/// obligation records match the shared chelis-prove engine the CLI also
/// drives (same obligation set AND same proof_tier per obligation).
#[test]
fn tide_runs_obligations_via_shared_engine() {
    // Tide path.
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":9,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": OPAQUE_MODULE, "seed": 0
        }}
    }))
    .expect("prove response");
    let tide_obs = response["result"]["structuredContent"]["obligations"]
        .as_array()
        .expect("obligations array")
        .clone();

    // Shared-engine path (what the CLI also calls).
    let engine_out = match chelis_prove::obligation_engine::run_surf_source_obligations(
        OPAQUE_MODULE,
        &chelis_prove::obligation_engine::ObligationRunOptions::default(),
    )
    .expect("engine run")
    {
        chelis_prove::obligation_engine::ObligationRunResult::Ran(o) => o,
        other => panic!("expected a clean module to run, got {other:?}"),
    };

    // Same obligation set: same names.
    let tide_names: Vec<&str> = tide_obs
        .iter()
        .map(|o| o["name"].as_str().unwrap())
        .collect();
    let engine_names: Vec<&str> = engine_out.iter().map(|o| o.name.as_str()).collect();
    assert_eq!(
        tide_names, engine_names,
        "same obligation set across surfaces"
    );

    // Same proof_tier per obligation.
    for (tide_ob, engine_ob) in tide_obs.iter().zip(&engine_out) {
        assert_eq!(
            tide_ob["proof_tier"].as_str().unwrap(),
            engine_ob.proof_tier.as_str(),
            "same proof_tier per obligation across surfaces"
        );
        assert_eq!(
            tide_ob["status"].as_str().unwrap(),
            match engine_ob.status {
                chelis_prove::obligation_engine::ObligationStatus::Passed => "passed",
                chelis_prove::obligation_engine::ObligationStatus::Failed => "failed",
                chelis_prove::obligation_engine::ObligationStatus::Unsupported => "unsupported",
                chelis_prove::obligation_engine::ObligationStatus::Error => "error",
            }
        );
    }

    assert_eq!(tide_obs.len(), 1, "the flagship has exactly one obligation");
    assert_eq!(tide_obs[0]["name"], "invariant:Probability:probability");
}

/// Under the `smt` feature, the flagship obligation proves at proof_tier
/// "smt" through tide (the same tier the CLI gets).
#[cfg(feature = "smt")]
#[test]
fn tide_flagship_obligation_proves_at_smt_tier() {
    let response = handle_message(&json!({
        "jsonrpc":"2.0",
        "id":10,
        "method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{
            "source_kind":"surf","source": OPAQUE_MODULE
        }}
    }))
    .expect("prove response");
    let obs = response["result"]["structuredContent"]["obligations"]
        .as_array()
        .unwrap();
    assert_eq!(obs[0]["status"], "passed");
    assert_eq!(obs[0]["proof_tier"], "smt");
    assert_eq!(obs[0]["arith_model"], "real");
}
