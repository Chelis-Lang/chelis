//! chelis#1134 / [04-INF-4]: public commands reject eager top-level value
//! forward references before evaluation or lowering.

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

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

fn assert_failed_report(name: &str, extension: &str, source: &str, expected_kind: &str) {
    let (success, report) = check_report(name, extension, source);
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
fn check_rejects_forward_values_on_deep_and_surf_surfaces() {
    assert_failed_report(
        "forward_value",
        "dp",
        "(def {} use_base (var {} base))\n\n\
         (def {} base (lit {type: (t-prim {} int32)} 7))\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "forward_value_surf",
        "ch",
        "use_base = base\n\
         base = 7\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "later_external_input",
        "ch",
        "module ExternalCli\n\
         def capture() -> int32 = x\n\
         x = (x : int32)\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "later_declared_external_input",
        "ch",
        "module DeclaredExternalCli\n\
         def capture() -> int32 = x\n\
         x: int32 = x\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "later_declared_value",
        "ch",
        "module DeclaredValueCli\n\
         def capture() -> int32 = value\n\
         value: int32 = 7\n",
        "UnboundVariable",
    );
}

#[test]
fn check_accepts_backward_values_module_helpers_and_external_inputs() {
    assert_clean_report(
        "backward_value",
        "dp",
        "(def {} base (lit {type: (t-prim {} int32)} 7))\n\n\
         (def {} use_base (var {} base))\n",
    );
    assert_clean_report(
        "forward_helper_module",
        "ch",
        "module ForwardCli\n\
         def use(x: f32) -> f32 = identity(x)\n\
         def identity(x) = x\n",
    );
    assert_clean_report("external_input", "ch", "x = (x : tensor[4, f32])\n");
    assert_clean_report("declared_external_input", "ch", "x: tensor[4, f32] = x\n");
}

#[test]
fn check_rejects_missing_local_forward_and_cycle_controls_honestly() {
    assert_failed_report(
        "missing_name",
        "dp",
        "(def {} use_missing (var {} missing))\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "local_forward",
        "dp",
        "(def {} local_forward\n\
           (let {}\n\
             (bind {} x (var {} y) y (lit {type: (t-prim {} int32)} 1))\n\
             (var {} x)))\n",
        "UnboundVariable",
    );
    assert_failed_report(
        "value_cycle",
        "dp",
        "(def {} first (var {} second))\n\n\
         (def {} second (var {} first))\n",
        "CycleDetected",
    );
}

#[test]
fn eval_and_build_reject_forward_values_before_execution_or_lowering() {
    let directory = tempdir().expect("tempdir");
    let path = directory.path().join("forward_value.ch");
    write_file(
        &path,
        "def capture() -> List[int64] = later\n\
         later = [1i64, 2i64]\n",
    );
    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 fixture path")])
        .output()
        .expect("chelis eval must run");
    let eval_stderr = String::from_utf8_lossy(&eval.stderr);
    assert!(
        !eval.status.success() && eval_stderr.contains("unbound variable: later"),
        "eval must reject the forward value as unbound; stdout={} stderr={eval_stderr}",
        String::from_utf8_lossy(&eval.stdout)
    );

    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().expect("UTF-8 fixture path"),
            "--target",
            "c",
        ])
        .output()
        .expect("chelis build must run");
    let build_stderr = String::from_utf8_lossy(&build.stderr);
    assert!(
        !build.status.success() && build_stderr.contains("unbound variable: later"),
        "C build must reject the forward value as unbound; stdout={} stderr={build_stderr}",
        String::from_utf8_lossy(&build.stdout)
    );
}

#[test]
fn eval_and_build_reject_a_later_external_input_before_lowering() {
    let directory = tempdir().expect("tempdir");
    for (name, source) in [
        (
            "later_ascribed_external_input",
            "module ExternalCli\n\
             def capture() -> int32 = x\n\
             x = (x : int32)\n",
        ),
        (
            "later_declared_external_input",
            "module DeclaredExternalCli\n\
             def capture() -> int32 = x\n\
             x: int32 = x\n",
        ),
    ] {
        let path = directory.path().join(format!("{name}.ch"));
        write_file(&path, source);
        for command in [
            vec!["eval", "--file", path.to_str().expect("UTF-8 fixture path")],
            vec![
                "build",
                path.to_str().expect("UTF-8 fixture path"),
                "--target",
                "c",
            ],
        ] {
            let output = Command::cargo_bin("chelis")
                .expect("chelis binary")
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args(command)
                .output()
                .expect("chelis command must run");
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                !output.status.success() && stderr.contains("unbound variable: x"),
                "command must reject the later external input; stdout={} stderr={stderr}",
                String::from_utf8_lossy(&output.stdout)
            );
        }
    }
}
