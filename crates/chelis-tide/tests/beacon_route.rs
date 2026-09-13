use chelis_tide::mcp::handle_message;
use serde_json::json;

#[test]
fn beacon_tier_is_exposed_and_reaches_shared_fail_closed_validation() {
    let response = handle_message(&json!({"jsonrpc":"2.0","id":1,"method":"tools/list"})).unwrap();
    let tools = response["result"]["tools"].as_array().unwrap();
    let tool = tools.iter().find(|tool| tool["name"] == "chelis_prove").unwrap();
    assert!(tool["inputSchema"]["properties"]["tier"]["enum"].as_array().unwrap().contains(&json!("beacon-only")));
    assert_eq!(tool["inputSchema"]["properties"]["beacon_budget"]["minimum"], 0);
    let response = handle_message(&json!({"jsonrpc":"2.0","id":2,"method":"tools/call",
        "params":{"name":"chelis_prove","arguments":{"source_kind":"surf","tier":"beacon-only",
            "source":"@property bounded forall(x: tensor[f64]):\n  tensor_to_scalar(x) <= 1.0f64\n"}}})).unwrap();
    let property = &response["result"]["structuredContent"]["properties"][0];
    assert_eq!(property["proof_tier"], "beacon", "{response}");
    assert_eq!(property["status"], "unsupported");
    assert_eq!(property["samples"], 0);
}

#[test]
fn malformed_beacon_budget_is_rejected_instead_of_defaulting() {
    for budget in [json!(-1),json!(0.5),json!("1000"),json!(null)] {
        let response = handle_message(&json!({"jsonrpc":"2.0","id":3,"method":"tools/call",
            "params":{"name":"chelis_prove","arguments":{"source_kind":"surf","tier":"beacon-only",
                "beacon_budget":budget,"source":"@property bounded forall(x: tensor[f64]):\n  tensor_to_scalar(x) <= 1.0f64\n"}}})).unwrap();
        let payload = &response["result"]["structuredContent"];
        assert_eq!(payload["errors"][0]["kind"],"invalid_arguments");
        assert!(payload["errors"][0]["message"].as_str().unwrap().contains("beacon_budget"),"{payload}");
    }
}
