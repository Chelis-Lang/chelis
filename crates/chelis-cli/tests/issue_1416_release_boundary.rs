//! #1416 release boundary, not the future implementation acceptance oracle.
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use serde_json::Value;

#[test]
fn unsupported_construct_exports_reject_original_concrete_calls_without_artifacts() {
    let (_dir, reef, app) = common::make_app("issue-1416-boundary");
    for (operation, definition) in [
        (
            "stack",
            "def run(xs: List[tensor[3, f32]]) = stack(xs, cast(0, i32))",
        ),
        (
            "squeeze",
            "def run(x: &tensor[1, 3, f32]) = squeeze(x, cast(0, i32))",
        ),
        (
            "unsqueeze",
            "def run(x: &tensor[3, f32]) = unsqueeze(x, cast(0, i32))",
        ),
    ] {
        common::write_file(
            &app.join("src/main.ch"),
            &format!("module Demo.Main\nimport Std.Tensor.Construct ({operation})\n{definition}\n"),
        );
        let checked = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef)
            .current_dir(&app)
            .args(["check", "src/main.ch"])
            .output()
            .unwrap();
        assert!(!checked.status.success(), "{operation}: {checked:?}");
        let report: Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert!(
            !report["errors"].as_array().unwrap().is_empty(),
            "{operation}: {report}"
        );
        let out = app.join(format!("refused-{operation}"));
        let built = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef)
            .current_dir(&app)
            .args(["build", "src/main.ch", "--target", "c", "--output"])
            .arg(&out)
            .output()
            .unwrap();
        assert!(!built.status.success(), "{operation}: {built:?}");
        assert!(!out.exists(), "refused export must not leave an artifact");
    }
}

#[test]
fn concrete_primitive_alternatives_execute_with_exact_native_eval_values() {
    let (_dir, reef, app) = common::make_app("issue-1416-alternatives");
    let source = include_str!("../../../examples/construct_alternatives.ch")
        .replace("Examples.ConstructAlternatives", "Demo.Main");
    common::write_file(&app.join("src/main.ch"), &source);
    let evaluated = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_REEF_HOME", &reef)
        .current_dir(&app)
        .args(["eval", "--file", "src/main.ch"])
        .output()
        .unwrap();
    assert!(evaluated.status.success(), "{evaluated:?}");
    let stdout = String::from_utf8(evaluated.stdout).unwrap();
    assert!(
        stdout.contains("squeezed = tensor(shape=[2], data=[1.0, 2.0])"),
        "{stdout}"
    );
    assert!(
        stdout.contains("unsqueezed = tensor(shape=[1, 2], data=[1.0, 2.0])"),
        "{stdout}"
    );
    assert!(
        stdout.contains("stacked = tensor(shape=[2, 2], data=[1.0, 2.0, 3.0, 4.0])"),
        "{stdout}"
    );
    assert_eq!(common::build_and_run_app(&reef, &app, "main"), stdout);
    common::write_file(
        &app.join("src/main.ch"),
        "module Demo.Main\nresult = insert(to_tensor([1.0f32, 2.0f32]), 2i32, 1i64)\n",
    );
    let invalid = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_REEF_HOME", &reef)
        .current_dir(&app)
        .args(["check", "src/main.ch"])
        .output()
        .unwrap();
    assert!(!invalid.status.success(), "{invalid:?}");
    let report: Value = serde_json::from_slice(&invalid.stdout).unwrap();
    assert!(
        report["errors"]
            .as_array()
            .unwrap()
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch"),
        "{report}"
    );
}
