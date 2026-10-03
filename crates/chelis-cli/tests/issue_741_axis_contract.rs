//! [05-AXIS-2]: shape-changing gather is static; shape-preserving sort is runtime.
#[path = "common/mod.rs"]
mod common;

use assert_cmd::Command;
use serde_json::Value;
use std::{fs, path::Path};
use tempfile::tempdir;

fn cli() -> Command {
    let mut command = Command::cargo_bin("chelis").unwrap();
    command.env("CHELIS_STYLE_GATE_DISABLE", "1");
    command
}

fn check(path: &Path) -> Value {
    let output = cli().arg("check").arg(path).output().unwrap();
    serde_json::from_slice(&output.stdout).expect("check JSON")
}

fn gather(axis: &str) -> String {
    format!("def axis0() -> i32 = cast(0, i32)\ndef pick(x: tensor[2, 3, f32], ids: tensor[2, i64], axis: i32) = gather(x, ids, {axis})\n")
}

#[test]
fn unresolved_gather_axes_fail_at_surf_and_deep_checking() {
    let dir = tempdir().unwrap();
    for axis in ["axis0()", "axis", "add(0i32, 0i32)"] {
        let surf = dir.path().join("probe.ch");
        fs::write(&surf, gather(axis)).unwrap();
        let output = cli().arg("deep").arg(&surf).output().unwrap();
        assert!(output.status.success(), "{output:?}");
        let deep = dir.path().join("probe.dp");
        fs::write(&deep, output.stdout).unwrap();
        for path in [&surf, &deep] {
            let report = check(path);
            assert!(report["score"].as_f64().unwrap() < 1.0, "{axis}: {report}");
            assert!(report["errors"].as_array().unwrap().iter().any(|e|
                e["kind"] == "DimensionMismatch"
                && e["message"].as_str().unwrap().contains("[05-AXIS-2]")
                && e["message"].as_str().unwrap().contains("integer constant")
            ), "{axis}: {report}");
        }
        let out = dir.path().join("out");
        let built = cli().args(["build", "--target", "c", "--emit-c", "--output"])
            .arg(&out).arg(&surf).output().unwrap();
        assert!(!built.status.success(), "{axis}: {built:?}");
        assert!(String::from_utf8_lossy(&built.stderr).contains("[05-AXIS-2]"), "{built:?}");
        assert!(!out.join("probe.c").exists());
        let evaluated = cli().args(["eval", "--file"]).arg(&surf).output().unwrap();
        assert!(!evaluated.status.success(), "{axis}: {evaluated:?}");
    }
}

#[test]
fn literal_gather_axes_determine_the_actual_result_geometry() {
    for axis in ["1i32", "cast(1, i32)", "-1i32", "cast(-1, i32)"] {
        let source = format!("def pick(x: tensor[2, 3, f32], ids: tensor[2, i64]) -> tensor[2, 2, f32] = gather(x, ids, {axis})\nxs = to_tensor([[1.0f32, 2.0f32, 3.0f32], [4.0f32, 5.0f32, 6.0f32]])\nids = to_tensor([2i64, 0i64])\nselected = pick(xs, ids)\n");
        let dir = tempdir().unwrap();
        let path = dir.path().join("gather.ch");
        fs::write(&path, &source).unwrap();
        let report = check(&path);
        assert_eq!(report["score"], 1.0, "{axis}: {report}");
        assert!(report["errors"].as_array().unwrap().is_empty(), "{report}");
        let eval = cli().args(["eval", "--file"]).arg(&path).output().unwrap();
        assert!(eval.status.success(), "{eval:?}");
        let stdout = String::from_utf8(eval.stdout).unwrap();
        assert_eq!(common::parse_tensor_data(&stdout, "selected"), vec![3.0, 1.0, 6.0, 4.0]);
        assert_eq!(common::build_and_run(&source, "gather"), stdout);
    }
}

#[test]
fn invalid_constant_axes_and_wrong_dtypes_are_checker_errors() {
    let dir = tempdir().unwrap();
    for (axis, reason) in [("2i32", "bounds"), ("-3i32", "bounds"), ("0i64", "i32")] {
        let path = dir.path().join("invalid.ch");
        fs::write(&path, gather(axis)).unwrap();
        let report = check(&path);
        assert!(report["score"].as_f64().unwrap() < 1.0, "{report}");
        assert!(report["errors"].as_array().unwrap().iter().any(|e| e["message"].as_str().unwrap().contains(reason)), "{axis}: {report}");
    }
}

const SORT: &str = "def axis0() -> i32 = cast(0, i32)\ndef ordered(x: tensor[3, f32], axis: i32) -> tensor[3, f32] = sort(x, axis).0\ndef main() -> tensor[3, f32] = ordered(to_tensor([3.0f32, 1.0f32, 2.0f32]), axis0())\n";

#[test]
fn computed_sort_axis_agrees_in_bare_and_reef_lanes() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("sort.ch");
    fs::write(&path, SORT).unwrap();
    let report = check(&path);
    assert_eq!(report["score"], 1.0, "{report}");
    let eval = cli().args(["eval", "--file"]).arg(&path).output().unwrap();
    assert!(eval.status.success(), "{eval:?}");
    let stdout = String::from_utf8(eval.stdout).unwrap();
    assert_eq!(common::parse_tensor_data(&stdout, "main"), vec![1.0, 2.0, 3.0]);
    assert_eq!(common::build_and_run(SORT, "sort"), stdout);
    let (_dir, reef_home, app) = common::make_app("axis-contract");
    let reef_source = format!("module Demo.Main\n{SORT}");
    fs::write(app.join("src/main.ch"), reef_source).unwrap();
    let eval = cli().env("CHELIS_REEF_HOME", &reef_home).current_dir(&app)
        .args(["eval", "--file", "src/main.ch"]).output().unwrap();
    assert!(eval.status.success(), "{eval:?}");
    assert_eq!(common::build_and_run_app(&reef_home, &app, "main"), String::from_utf8(eval.stdout).unwrap());
    fs::write(app.join("src/main.ch"), format!("module Demo.Main\n{}", gather("axis0()"))).unwrap();
    let out = app.join("gather-out");
    let failed = cli().env("CHELIS_REEF_HOME", &reef_home).current_dir(&app)
        .args(["build", "--target", "c", "--output"]).arg(&out)
        .arg("src/main.ch").output().unwrap();
    assert!(!failed.status.success(), "{failed:?}");
    assert!(String::from_utf8_lossy(&failed.stderr).contains("[05-AXIS-2]"), "{failed:?}");
    assert!(!out.join("main.c").exists());
}

#[test]
fn invalid_computed_sort_axis_traps_in_both_lanes() {
    let source = SORT.replace("cast(0, i32)", "cast(1, i32)");
    let dir = tempdir().unwrap();
    let path = dir.path().join("sort.ch");
    fs::write(&path, source).unwrap();
    let report = check(&path);
    assert_eq!(report["score"], 1.0, "{report}");
    let eval = cli().args(["eval", "--file"]).arg(&path).output().unwrap();
    assert!(!eval.status.success(), "{eval:?}");
    assert!(String::from_utf8_lossy(&eval.stderr).contains("axis"), "{eval:?}");
    let out = dir.path().join("out");
    cli().args(["build", "--target", "c", "--output"]).arg(&out).arg(&path).assert().success();
    let run = std::process::Command::new(out.join("sort")).output().unwrap();
    assert!(!run.status.success(), "{run:?}");
    assert!(String::from_utf8_lossy(&run.stderr).contains("axis"), "{run:?}");
}
