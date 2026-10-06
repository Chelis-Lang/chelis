use assert_cmd::Command;
use serde_json::{Value, json};
use std::time::Duration;

fn run(input: &str) -> std::process::Output {
    Command::new(assert_cmd::cargo::cargo_bin!("chelis"))
        .args(["tide", "mcp"])
        .write_stdin(input)
        .timeout(Duration::from_secs(10))
        .output()
        .expect("MCP subprocess")
}

fn lines(messages: &[Value]) -> String {
    messages
        .iter()
        .map(|message| format!("{message}\n"))
        .collect()
}

#[test]
fn stdio_initialize_notifications_discovery_and_check() {
    let output = run(&lines(&[
        json!({"jsonrpc":"2.0","id":1,"method":"initialize","params":{
            "protocolVersion":"2024-11-05","capabilities":{},
            "clientInfo":{"name":"stdio-test","version":"1"}}}),
        json!({"jsonrpc":"2.0","method":"notifications/initialized"}),
        json!({"jsonrpc":"2.0","method":"notifications/unknown"}),
        json!({"jsonrpc":"2.0","method":"tools/call","params":{}}),
        json!({"jsonrpc":"2.0","id":2,"method":"tools/list"}),
        json!({"jsonrpc":"2.0","id":"píng","method":"ping"}),
        json!({"jsonrpc":"2.0","id":3,"method":"tools/call","params":{
            "name":"chelis_check","arguments":{"source_kind":"surf",
            "source":"-- café\nx = 2.0f32 + 3.0f32\n"}}}),
    ]));
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stderr.is_empty(), "{:?}", output);
    let stdout = String::from_utf8(output.stdout).unwrap();
    assert!(stdout.ends_with('\n'));
    let replies: Vec<Value> = stdout
        .lines()
        .map(|line| serde_json::from_str(line).expect("only JSON on stdout"))
        .collect();
    assert_eq!(replies.len(), 4, "notifications must not get replies");
    assert_eq!(replies[0]["id"], 1);
    assert_eq!(replies[0]["result"]["protocolVersion"], "2024-11-05");
    assert_eq!(replies[1]["id"], 2);
    assert!(
        replies[1]["result"]["tools"]
            .as_array()
            .unwrap()
            .iter()
            .any(|tool| tool["name"] == "chelis_check")
    );
    assert_eq!(replies[2]["id"], "píng");
    assert_eq!(replies[2]["result"], json!({}));
    assert_eq!(replies[3]["id"], 3);
    assert_eq!(replies[3]["result"]["isError"], false);
    assert_eq!(replies[3]["result"]["structuredContent"]["ok"], true);
}

#[test]
fn stdio_tool_failures_remain_json_rpc_responses() {
    let output = run(&lines(&[
        json!({"jsonrpc":"2.0","id":0,"method":"tools/call","params":{
            "name":"not_a_tool"}}),
        json!({"jsonrpc":"2.0","id":1,"method":"tools/call","params":{
            "name":"chelis_check","arguments":{}}}),
    ]));
    assert!(output.status.success());
    let replies: Vec<Value> = String::from_utf8(output.stdout)
        .unwrap()
        .lines()
        .map(|line| serde_json::from_str(line).unwrap())
        .collect();
    assert_eq!(replies.len(), 2);
    assert_eq!(replies[0]["id"], 0);
    assert_eq!(replies[0]["error"]["code"], -32601);
    assert_eq!(replies[1]["id"], 1);
    assert_eq!(replies[1]["result"]["isError"], true);
}

#[test]
fn stdio_tool_notifications_do_not_execute() {
    let dir = tempfile::tempdir().expect("notification witness directory");
    let witness = dir.path().join("dispatch.txt");
    std::fs::write(&witness, "untouched").expect("seed dispatch witness");
    let notification = json!({"jsonrpc":"2.0","method":"tools/call","params":{
        "name":"chelis_eval","arguments":{"source_kind":"surf",
        "source":format!("written = write_file({witness:?}, \"dispatched\")\n")}}});
    let ping = json!({"jsonrpc":"2.0","id":"after-notification","method":"ping"});

    let output = run(&lines(&[notification.clone(), ping]));
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stderr.is_empty(), "{:?}", output);
    assert_eq!(
        std::fs::read_to_string(&witness).unwrap(),
        "untouched",
        "a notification must not execute its tool, even if its reply is discarded"
    );
    assert_eq!(
        serde_json::from_slice::<Value>(&output.stdout).unwrap(),
        json!({"jsonrpc":"2.0","id":"after-notification","result":{}})
    );

    // The same tool arguments must actually write when sent as a request;
    // an invalid or refused tool call cannot prove notification non-dispatch.
    let mut request = notification;
    request["id"] = json!("dispatch-control");
    let output = run(&lines(&[request]));
    assert!(output.status.success(), "{:?}", output);
    assert!(output.stderr.is_empty(), "{:?}", output);
    let reply: Value = serde_json::from_slice(&output.stdout).unwrap();
    assert_eq!(reply["id"], "dispatch-control");
    assert_eq!(reply["result"]["isError"], false);
    assert_eq!(reply["result"]["structuredContent"]["ok"], true);
    assert_eq!(std::fs::read_to_string(&witness).unwrap(), "dispatched");
}

#[test]
fn stdio_rejects_lsp_headers_invalid_json_and_unterminated_messages() {
    for input in ["Content-Length: 2\r\n\r\n{}", "not JSON\n", "{\n}\n", "{}"] {
        let output = run(input);
        assert!(!output.status.success(), "accepted {input:?}");
        assert!(output.stdout.is_empty(), "non-MCP stdout for {input:?}");
    }
}

#[test]
fn stdio_empty_input_closes_cleanly() {
    let output = run("");
    assert!(output.status.success());
    assert!(output.stdout.is_empty());
    assert!(output.stderr.is_empty());
}
