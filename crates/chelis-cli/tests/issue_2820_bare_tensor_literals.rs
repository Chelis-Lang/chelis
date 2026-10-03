//! Observable binding kind/default dtype is shared by both execution lanes.
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use serde_json::{Value, json};
fn cli(reef: &std::path::Path, app: &std::path::Path) -> Command {
    let mut cmd = Command::cargo_bin("chelis").unwrap();
    cmd.env("CHELIS_STYLE_GATE_DISABLE", "1").env("CHELIS_REEF_HOME", reef).current_dir(app);
    cmd
}
#[test]
fn bare_and_declared_tensor_roots_agree_in_json_and_native_execution() {
    let (_dir, reef, app) = common::make_app("issue-2820");
    for (literal, ty, shape) in [
        ("[1, 2, 3]", "tensor[3, i32]", json!([3])),
        ("[1.0, 2.0, 3.0]", "tensor[3, f32]", json!([3])),
        ("[-1i64, 9007199254740993i64]", "tensor[2, i64]", json!([2])),
        ("[1.0f16, 2.0f16]", "tensor[2, f16]", json!([2])),
        ("[1.0bf16, 2.0bf16]", "tensor[2, bf16]", json!([2])),
        ("[[1.0f64, 2.0f64], [3.0f64, 4.0f64]]", "tensor[2, 2, f64]", json!([2, 2])),
    ] {
        common::write_file(&app.join("src/main.ch"), &format!("module Demo.Main\nbare = {literal}\ndeclared: {ty} = {literal}\n"));
        let checked = cli(&reef,&app).args(["check","src/main.ch"]).output().unwrap();
        let checked_json: Value = serde_json::from_slice(&checked.stdout).unwrap();
        assert!(checked.status.success(), "{checked:?}");
        assert_eq!(checked_json["errors"], json!([]));
        assert_eq!(checked_json["score"], 1.0);
        let eval = cli(&reef,&app).args(["eval","--json","--file","src/main.ch"]).output().unwrap();
        assert!(eval.status.success(), "{eval:?}");
        let report: Value = serde_json::from_slice(&eval.stdout).unwrap();
        let roots = report["roots"].as_array().unwrap();
        let bare = roots.iter().find(|r|r["name"]=="bare").unwrap();
        let declared = roots.iter().find(|r|r["name"]=="declared").unwrap();
        assert_eq!(bare["value"]["type"], "tensor");
        assert_eq!(bare["value"]["value"]["shape"], shape);
        assert_eq!(bare["value"], declared["value"]);
        let text = cli(&reef,&app).args(["eval","--file","src/main.ch"]).output().unwrap();
        assert!(text.status.success(), "{text:?}");
        let evaluated = String::from_utf8(text.stdout).unwrap();
        let native = common::build_and_run_app(&reef,&app,"main");
        assert_eq!(native,evaluated);
        assert!(native.contains("bare = tensor("), "{native}");
    }
}
#[test]
fn explicit_list_roots_retain_list_kind_and_mixed_tensor_dtypes_reject() {
    let (_dir,reef,app) = common::make_app("issue-2820-negative");
    for declaration in ["result: List[f32] = [1.0, 2.0]", "sig result: List[i32]\nresult = [1, 2]"] {
        common::write_file(&app.join("src/main.ch"), &format!("module Demo.Main\n{declaration}\n"));
        let output = cli(&reef,&app).args(["eval","--json","--file","src/main.ch"]).output().unwrap();
        assert!(output.status.success(), "{output:?}");
        let report: Value=serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(report["roots"][0]["value"]["type"], "list");
        let evaluated = cli(&reef,&app).args(["eval","--file","src/main.ch"]).output().unwrap();
        assert!(evaluated.status.success(), "{evaluated:?}");
        assert_eq!(common::build_and_run_app(&reef,&app,"main"),String::from_utf8(evaluated.stdout).unwrap());
    }
    for literal in ["[1.0, 2.0f64]", "[1i32, 2i64]", "[[1.0], [2.0, 3.0]]"] {
        common::write_file(&app.join("src/main.ch"), &format!("module Demo.Main\nresult = {literal}\n"));
        let output = cli(&reef,&app).args(["check","src/main.ch"]).output().unwrap();
        let report: Value=serde_json::from_slice(&output.stdout).unwrap();
        assert!(!output.status.success(), "{literal}: {report}");
        assert!(!report["errors"].as_array().unwrap().is_empty(), "{literal}: {report}");
    }
}
