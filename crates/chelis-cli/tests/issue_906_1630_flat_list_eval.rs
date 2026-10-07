//! Large flat list literals evaluate without exhausting the native stack.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

fn eval(source: &str) -> serde_json::Value {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("flat.ch");
    fs::write(&file, source).expect("write Surf source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args([
            "eval",
            "--file",
            file.to_str().expect("utf8 path"),
            "--json",
        ])
        .output()
        .expect("run eval");
    assert!(
        output.status.success(),
        "eval: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("eval JSON")
}

#[test]
fn a_flat_list_beyond_the_reported_abort_threshold_evaluates() {
    let length = 4_096;
    let source = format!(
        "module Probe.Flat\nresult = [{}]\n",
        (0..length)
            .map(|index| format!("{index}.0f64"))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let result = eval(&source);
    let items = result["roots"][0]["value"]["value"]
        .as_array()
        .expect("list values");
    assert_eq!(items.len(), length);
    assert_eq!(items[0]["value"]["dtype"], "f64");
    assert_eq!(items[length - 1]["value"]["dtype"], "f64");
}

#[test]
fn a_flat_tensor_input_beyond_the_reported_abort_threshold_evaluates() {
    let length = 5_000;
    let source = format!(
        "module Probe.Tensor\nresult = to_tensor([{}], f64)\n",
        (0..length)
            .map(|index| format!("{}.0f64", index % 97))
            .collect::<Vec<_>>()
            .join(", ")
    );
    let result = eval(&source);
    let tensor = &result["roots"][0]["value"]["value"];
    assert_eq!(tensor["shape"], serde_json::json!([length]));
    assert_eq!(tensor["data"]["dtype"], "f64");
    assert_eq!(
        tensor["data"]["bits"]
            .as_array()
            .expect("tensor data")
            .len(),
        length
    );
}

#[test]
fn a_cons_with_a_non_list_tail_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let file = dir.path().join("bad-tail.ch");
    fs::write(&file, "module Probe.BadTail\nresult = Cons(1i64, 2i64)\n")
        .expect("write Surf source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["check", file.to_str().expect("utf8 path")])
        .output()
        .expect("run check");
    assert!(!output.status.success(), "improper tail was accepted");
    let report: serde_json::Value = serde_json::from_slice(&output.stdout).expect("check JSON");
    assert!(!report["errors"].as_array().expect("errors").is_empty());
}
