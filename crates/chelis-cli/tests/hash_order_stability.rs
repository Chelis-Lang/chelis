//! Phase A acceptance surface for hash-order-independent positional `expand` settlement.
//!
//! Every command invocation below is a fresh process with a freshly seeded
//! standard-library `HashMap`. One successful in-process check would not lock
//! chelis#1338: the original reproducer accepted roughly two runs in three.

use assert_cmd::Command;
use serde_json::Value;
use std::io::Write;

const FRESH_PROCESS_RUNS: usize = 24;

const ACCEPTING_ROWS: &[(&str, &str)] = &[
    (
        "original",
        "result = sub(reshape(expand(to_tensor([0.0f32]), 0, 6i64), [3i64, 2i64]), expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64))\n",
    ),
    (
        "mirrored",
        "result = sub(expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64), reshape(expand(to_tensor([0.0f32]), 0, 6i64), [3i64, 2i64]))\n",
    ),
    (
        "aliased_pair",
        "result = add(expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64), expand(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]), 0, 3i64))\n",
    ),
    (
        "three_way_original",
        "result = sub(reshape(expand(to_tensor([0.0f32]), 0, 6i64), [3i64, 2i64]), add(expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64), expand(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]), 0, 3i64)))\n",
    ),
    (
        "three_way_mirrored",
        "result = sub(add(expand(to_tensor([1.0f32, 2.0f32]), 0, 3i64), expand(to_tensor([[1.0f32, 2.0f32], [3.0f32, 4.0f32], [5.0f32, 6.0f32]]), 0, 3i64)), reshape(expand(to_tensor([0.0f32]), 0, 6i64), [3i64, 2i64]))\n",
    ),
];

const REJECTING_SHAPE_READ: &str = "expanded = expand(to_tensor([0.0f32]), 0, 6i64)\nresult = reshape(expanded, [shape(expanded, 1), 6i64])\n";

fn fixture(label: &str, source: &str) -> tempfile::NamedTempFile {
    let mut file = tempfile::Builder::new()
        .prefix(&format!("hash-order-{label}-"))
        .suffix(".ch")
        .tempfile()
        .expect("create hash-order fixture");
    file.write_all(source.as_bytes())
        .expect("write hash-order fixture");
    file.flush().expect("flush hash-order fixture");
    file
}

fn check(path: &std::path::Path) -> (Option<i32>, Value) {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("UTF-8 fixture path")])
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8(output.stdout).expect("UTF-8 check report");
    let report = serde_json::from_str(&stdout)
        .unwrap_or_else(|error| panic!("check report must be JSON: {error}\n{stdout}"));
    (output.status.code(), report)
}

#[test]
fn accepting_rows_are_stable_across_24_fresh_processes() {
    for &(label, source) in ACCEPTING_ROWS {
        let file = fixture(label, source);
        for run in 0..FRESH_PROCESS_RUNS {
            let (status, report) = check(file.path());
            assert_eq!(status, Some(0), "{label} run {run}: {report}");
            assert_eq!(
                report["score"].as_f64(),
                Some(1.0),
                "{label} run {run}: {report}"
            );
            assert!(
                report["errors"]
                    .as_array()
                    .expect("errors array")
                    .is_empty(),
                "{label} run {run}: {report}"
            );
        }
    }
}

#[test]
fn replacement_shape_read_rejects_identically_across_24_fresh_processes() {
    let file = fixture("replacement-shape-read", REJECTING_SHAPE_READ);
    let mut expected_errors = None;
    for run in 0..FRESH_PROCESS_RUNS {
        let (status, report) = check(file.path());
        assert_eq!(status, Some(2), "negative run {run}: {report}");
        assert!(
            report["score"].as_f64().is_some_and(|score| score < 1.0),
            "negative run {run}: {report}"
        );
        let errors = report["errors"].as_array().expect("errors array");
        assert!(
            errors.iter().any(|error| {
                error["kind"] == "DimensionMismatch"
                    && error["message"]
                        .as_str()
                        .is_some_and(|message| message.contains("axis 1"))
            }),
            "negative run {run}: {report}"
        );
        if let Some(expected) = &expected_errors {
            assert_eq!(errors, expected, "negative run {run}: {report}");
        } else {
            expected_errors = Some(errors.clone());
        }
    }
}
