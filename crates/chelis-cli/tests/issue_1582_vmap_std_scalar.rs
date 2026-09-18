//! chelis#1582: imported `Std.Scalar.min` and `Std.Scalar.max` remain
//! ordinary lexical callables under `vmap`.
//!
//! The original failure accepted these programs in the checker, then host
//! evaluation treated the callable's own name as a symbolic tensor input and
//! failed with `missing required input min` (or `max`). The closing oracle
//! checks the package import, evaluator, and compiled-C lanes with exact
//! results. Its negative mirrors prove the names are not ambient builtins:
//! without the explicit import, the checker reports `UnboundVariable`.

use assert_cmd::Command;
use serde_json::Value;

#[path = "common/mod.rs"]
mod common;

use common::{build_and_run_app, make_app, write_file};

const IMPORTED_PROBE: &str = r#"module Demo.Main

import Std.Scalar (max, min)

def clamp_min(x: f64) -> f64 = min(x, cast(0.5, f64))
def clamp_max(x: f64) -> f64 = max(x, cast(0.5, f64))

def rowwise_min(t: tensor[3, f64]) -> tensor[3, f64] = {
  s = tensor_to_scalar(sum(t, cast(0, i32)))
  insert(scalar_to_tensor(clamp_min(s)), cast(0, i32), cast(3, i64))
}

def rowwise_max(t: tensor[3, f64]) -> tensor[3, f64] = {
  s = tensor_to_scalar(sum(t, cast(0, i32)))
  insert(scalar_to_tensor(clamp_max(s)), cast(0, i32), cast(3, i64))
}

def batched_min(b: tensor[2, 3, f64]) -> tensor[2, 3, f64] = vmap(rowwise_min)(b)
def batched_max(b: tensor[2, 3, f64]) -> tensor[2, 3, f64] = vmap(rowwise_max)(b)

inbatch = to_tensor([
  [cast(1.0, f64), cast(0.0, f64), cast(0.0, f64)],
  [cast(0.1, f64), cast(0.0, f64), cast(0.0, f64)]
])
min_probe = to_list(sum(batched_min(inbatch), cast(1, i32)))
max_probe = to_list(sum(batched_max(inbatch), cast(1, i32)))

outside_min = min(cast(9.0, f64), cast(0.5, f64))
outside_max = max(cast(0.1, f64), cast(0.5, f64))
"#;

fn assert_probe_output(stdout: &str, lane: &str) {
    assert!(
        stdout
            .lines()
            .any(|line| line == "min_probe = [1.5, 0.30000000000000004]"),
        "{lane}: imported min under vmap returned the wrong exact value:\n{stdout}"
    );
    assert!(
        stdout.lines().any(|line| line == "max_probe = [3.0, 1.5]"),
        "{lane}: imported max under vmap returned the wrong exact value:\n{stdout}"
    );
    assert!(
        stdout.lines().any(|line| line == "outside_min = 0.5"),
        "{lane}: outside-transform min control failed:\n{stdout}"
    );
    assert!(
        stdout.lines().any(|line| line == "outside_max = 0.5"),
        "{lane}: outside-transform max control failed:\n{stdout}"
    );
}

#[test]
fn imported_min_and_max_work_under_vmap_in_eval_and_native_c() {
    let (_dir, reef_home, app_pkg) = make_app("issue-1582-vmap-std-scalar");
    write_file(&app_pkg.join("src/main.ch"), IMPORTED_PROBE);

    let eval = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .output()
        .expect("run chelis eval");
    assert!(
        eval.status.success(),
        "eval failed: stdout={} stderr={}",
        String::from_utf8_lossy(&eval.stdout),
        String::from_utf8_lossy(&eval.stderr)
    );
    let eval_stdout = String::from_utf8(eval.stdout).expect("eval stdout is utf-8");
    assert_probe_output(&eval_stdout, "eval");

    let native_stdout = build_and_run_app(&reef_home, &app_pkg, "main");
    assert_probe_output(&native_stdout, "native C");
    for binding in ["min_probe", "max_probe", "outside_min", "outside_max"] {
        let prefix = format!("{binding} = ");
        let eval_line = eval_stdout
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("eval omitted {binding}: {eval_stdout}"));
        let native_line = native_stdout
            .lines()
            .find(|line| line.starts_with(&prefix))
            .unwrap_or_else(|| panic!("native C omitted {binding}: {native_stdout}"));
        assert_eq!(
            native_line, eval_line,
            "eval/native disagreement for {binding}"
        );
    }
}

#[test]
fn unimported_min_and_max_are_unbound_variables() {
    for name in ["min", "max"] {
        let (_dir, reef_home, app_pkg) = make_app(&format!("issue-1582-unbound-{name}"));
        write_file(
            &app_pkg.join("src/main.ch"),
            &format!("module Demo.Main\n\ndef clamp(x: f64) -> f64 = {name}(x, cast(0.5, f64))\n"),
        );

        let check = Command::cargo_bin("chelis")
            .expect("chelis binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef_home)
            .current_dir(&app_pkg)
            .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
            .output()
            .expect("run chelis check");
        assert!(
            !check.status.success(),
            "unimported {name} unexpectedly checked"
        );
        let report: Value = serde_json::from_slice(&check.stdout)
            .unwrap_or_else(|error| panic!("invalid check JSON for {name}: {error}"));
        let errors = report["errors"]
            .as_array()
            .unwrap_or_else(|| panic!("missing errors array for {name}: {report:#}"));
        assert!(
            errors.iter().any(|error| {
                error["kind"] == "UnboundVariable"
                    && error["message"]
                        .as_str()
                        .is_some_and(|message| message.contains(name))
            }),
            "unimported {name} must report UnboundVariable naming {name}: {report:#}"
        );
    }
}
