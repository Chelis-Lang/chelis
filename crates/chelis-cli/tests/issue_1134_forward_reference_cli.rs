//! chelis#1134 / [04-INF-4]: the public checker accepts legal top-level
//! forward references and remains honest for the sequential/cyclic controls.

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run, gcc_available, write_file};

fn check_report(name: &str, extension: &str, source: &str) -> (bool, serde_json::Value) {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join(format!("{name}.{extension}"));
    write_file(&path, source);
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("UTF-8 fixture path")])
        .output()
        .expect("chelis check must run");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "chelis check must emit JSON: {error}\nstdout: {}\nstderr: {}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status.success(), report)
}

fn assert_clean_report(name: &str, extension: &str, source: &str) {
    let (success, report) = check_report(name, extension, source);
    assert!(success, "{name}: expected check success: {report:#}");
    assert_eq!(report["score"].as_f64(), Some(1.0), "{name}: {report:#}");
    assert_eq!(report["errors"].as_array().map(Vec::len), Some(0));
    assert_eq!(report["unresolved_names"].as_array().map(Vec::len), Some(0));
}

fn assert_failed_report(name: &str, source: &str, expected_kind: &str) {
    let (success, report) = check_report(name, "dp", source);
    assert!(!success, "{name}: invalid program exited successfully");
    assert!(
        report["score"].as_f64().is_some_and(|score| score < 1.0),
        "{name}: invalid program reported perfect fitness: {report:#}"
    );
    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        errors.iter().any(|error| error["kind"] == expected_kind),
        "{name}: expected {expected_kind}: {report:#}"
    );
}

#[test]
fn check_accepts_forward_values_on_deep_and_surf_surfaces() {
    assert_clean_report(
        "forward_value",
        "dp",
        "(def {} use_base (var {} base))\n\n\
         (def {} base (lit {type: (t-prim {} int32)} 7))\n",
    );
    assert_clean_report(
        "forward_helper_bare",
        "ch",
        "def use(x: f32) -> f32 = identity(x)\n\
         def identity(x) = x\n",
    );
    assert_clean_report(
        "forward_helper_module",
        "ch",
        "module ForwardCli\n\
         def use(x: f32) -> f32 = identity(x)\n\
         def identity(x) = x\n",
    );
}

#[test]
fn check_rejects_missing_local_forward_and_cycle_controls_honestly() {
    assert_failed_report(
        "missing_name",
        "(def {} use_missing (var {} missing))\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "local_forward",
        "(def {} local_forward\n\
           (let {}\n\
             (bind {} x (var {} y) y (lit {type: (t-prim {} int32)} 1))\n\
             (var {} x)))\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "value_cycle",
        "(def {} first (var {} second))\n\n\
         (def {} second (var {} first))\n",
        "CycleDetected",
    );
}

#[test]
fn eval_observes_a_later_generic_helper() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("forward_eval.ch");
    write_file(
        &path,
        "def use(x: f32) -> f32 = identity(x)\n\
         def identity(x) = x\n\
         result = use(7.0)\n",
    );
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 fixture path")])
        .assert()
        .success()
        .stdout("result = 7.0\n");
}

#[test]
fn generated_c_observes_a_later_concrete_helper() {
    if !gcc_available() {
        return;
    }
    let stdout = build_and_run(
        "def use(x: f32) -> f32 = identity(x)\n\
         def identity(x: f32) -> f32 = x\n\
         result = use(7.0)\n",
        "forward_reference_c",
    );
    assert_eq!(stdout, "result = 7.0\n");
}
