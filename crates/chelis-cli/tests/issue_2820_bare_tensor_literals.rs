//! Observable binding kind/default dtype is shared by both execution lanes: a
//! bare bracket literal is a List, and `to_tensor` or a declared tensor type
//! makes a tensor (spec/02-surf-syntax.md §P10b).
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use serde_json::{Value, json};
fn cli(reef: &std::path::Path, app: &std::path::Path) -> Command {
    let mut cmd = Command::cargo_bin("chelis").unwrap();
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", reef)
        .current_dir(app);
    cmd
}
#[test]
fn explicit_and_declared_tensor_roots_agree_in_json_and_native_execution() {
    let (_dir, reef, app) = common::make_app("issue-2820");
    for (literal, ty, shape) in [
        ("[1, 2, 3]", "tensor[3, i32]", json!([3])),
        ("[1.0, 2.0, 3.0]", "tensor[3, f32]", json!([3])),
        ("[-1i64, 9007199254740993i64]", "tensor[2, i64]", json!([2])),
        ("[1.0f16, 2.0f16]", "tensor[2, f16]", json!([2])),
        ("[1.0bf16, 2.0bf16]", "tensor[2, bf16]", json!([2])),
        (
            "[[1.0f64, 2.0f64], [3.0f64, 4.0f64]]",
            "tensor[2, 2, f64]",
            json!([2, 2]),
        ),
    ] {
        let dtype = ty.trim_end_matches(']').rsplit(", ").next().unwrap();
        common::write_file(
            &app.join("src/main.ch"),
            &format!(
                "module Demo.Main\nbare = {literal}\nexplicit = to_tensor({literal}, {dtype})\ndeclared: {ty} = {literal}\n"
            ),
        );
        let checked = cli(&reef, &app)
            .args(["check", "src/main.ch"])
            .output()
            .unwrap();
        let checked_json: Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert!(checked.status.success(), "{checked:?}");
        assert_eq!(checked_json["errors"], json!([]));
        assert_eq!(checked_json["score"], 1.0);
        let eval = cli(&reef, &app)
            .args(["eval", "--json", "--file", "src/main.ch"])
            .output()
            .unwrap();
        assert!(eval.status.success(), "{eval:?}");
        let report: Value = serde_json::from_slice(&eval.stdout).unwrap();
        let roots = report["roots"].as_array().unwrap();
        let bare = roots.iter().find(|r| r["name"] == "bare").unwrap();
        let explicit = roots.iter().find(|r| r["name"] == "explicit").unwrap();
        let declared = roots.iter().find(|r| r["name"] == "declared").unwrap();
        assert_eq!(bare["value"]["type"], "list");
        assert_eq!(explicit["value"]["type"], "tensor");
        assert_eq!(explicit["value"]["value"]["shape"], shape);
        assert_eq!(explicit["value"], declared["value"]);
        let text = cli(&reef, &app)
            .args(["eval", "--file", "src/main.ch"])
            .output()
            .unwrap();
        assert!(text.status.success(), "{text:?}");
        let evaluated = String::from_utf8(text.stdout).unwrap();
        let native = common::build_and_run_app(&reef, &app, "main");
        assert_eq!(native, evaluated);
        assert!(native.contains("explicit = tensor("), "{native}");
        assert!(!native.contains("bare = tensor("), "{native}");
    }
}
#[test]
fn explicit_list_roots_retain_list_kind_and_mixed_tensor_dtypes_reject() {
    let (_dir, reef, app) = common::make_app("issue-2820-negative");
    for declaration in [
        "result: List[f32] = [1.0, 2.0]",
        "sig result: List[i32]\nresult = [1, 2]",
    ] {
        common::write_file(
            &app.join("src/main.ch"),
            &format!("module Demo.Main\n{declaration}\n"),
        );
        let output = cli(&reef, &app)
            .args(["eval", "--json", "--file", "src/main.ch"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["roots"][0]["value"]["type"], "list");
        let evaluated = cli(&reef, &app)
            .args(["eval", "--file", "src/main.ch"])
            .output()
            .unwrap();
        assert!(evaluated.status.success(), "{evaluated:?}");
        assert_eq!(
            common::build_and_run_app(&reef, &app, "main"),
            String::from_utf8(evaluated.stdout).unwrap()
        );
    }
    for literal in [
        "[1.0, 2.0f64]",
        "[1i32, 2i64]",
        "to_tensor([1.0, 2.0f64], f32)",
        "to_tensor([[1.0], [2.0, 3.0]], f32)",
    ] {
        common::write_file(
            &app.join("src/main.ch"),
            &format!("module Demo.Main\nresult = {literal}\n"),
        );
        let output = cli(&reef, &app)
            .args(["check", "src/main.ch"])
            .output()
            .unwrap();
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(!output.status.success(), "{literal}: {report}");
        assert!(
            !report["errors"].as_array().unwrap().is_empty(),
            "{literal}: {report}"
        );
    }
}

#[test]
fn captured_constructor_refuses_and_ragged_rejection_has_a_source_location() {
    let (_dir, reef, app) = common::make_app("issue-2820-capture");
    let source = "module Demo.Main\ndef sample() = {\n to_tensor = fn (xs: List[i32]) -> xs\n xs: tensor[2, i32] = [1, 2]\n xs\n}\nresult = sample()\n";
    common::write_file(&app.join("src/main.ch"), source);
    for command in [
        ["check", "src/main.ch"].as_slice(),
        ["eval", "--file", "src/main.ch"].as_slice(),
    ] {
        let output = cli(&reef, &app).args(command).output().unwrap();
        assert!(!output.status.success(), "{output:?}");
        let reported = format!(
            "{}{}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        );
        assert!(
            reported.contains("`to_tensor` is reserved and cannot be bound"),
            "{output:?}"
        );
    }
    cli(&reef, &app)
        .args([
            "build",
            "src/main.ch",
            "--target",
            "c",
            "--output",
            "capture-output",
        ])
        .assert()
        .failure();
    assert!(!app.join("capture-output/main").exists());
    common::write_file(
        &app.join("src/main.ch"),
        "module Demo.Main\nresult = to_tensor([[1i32], [2i32, 3i32]])\n",
    );
    let output = cli(&reef, &app)
        .args(["check", "src/main.ch"])
        .output()
        .unwrap();
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    let error = report["errors"]
        .as_array()
        .unwrap()
        .iter()
        .find(|e| {
            e["message"]
                .as_str()
                .is_some_and(|m| m.contains("[05-OP-57]"))
        })
        .unwrap();
    assert!(
        error["span_id"].is_string() || error["span"].is_object(),
        "{error}"
    );
}
