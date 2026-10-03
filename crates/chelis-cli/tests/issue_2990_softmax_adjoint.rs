//! [05-OP-48]: preserve softmax identity until its declared adjoint is applied.
#[path = "common/mod.rs"]
mod common;
use assert_cmd::Command;
use serde_json::Value;

#[test]
fn softmax_adjoint_matches_its_lane_forward_formula_at_every_float_width() {
    for dtype in ["f16", "bf16", "f32", "f64"] {
        for (shape, axis, values, weights, reduce) in [
            (
                "4",
                "0i32",
                "[0.1P, 0.7P, 1.3P, 2.9P]",
                "[1.0P, 2.0P, 3.0P, 4.0P]",
                "sum(mul(y, w), 0i32)",
            ),
            (
                "4",
                "-1i32",
                "[2.0P, 2.0P, -1.0P, 0.0P]",
                "[1.0P, 2.0P, 3.0P, 4.0P]",
                "sum(mul(y, w), 0i32)",
            ),
            (
                "2, 4",
                "0i32",
                "[[0.1P, 0.7P, 1.3P, 2.9P], [2.0P, 2.0P, -1.0P, 0.0P]]",
                "[[1.0P, 2.0P, 3.0P, 4.0P], [4.0P, 3.0P, 2.0P, 1.0P]]",
                "sum(sum(mul(y, w), 1i32), 0i32)",
            ),
            (
                "2, 4",
                "-1i32",
                "[[0.1P, 0.7P, 1.3P, 2.9P], [2.0P, 2.0P, -1.0P, 0.0P]]",
                "[[1.0P, 2.0P, 3.0P, 4.0P], [4.0P, 3.0P, 2.0P, 1.0P]]",
                "sum(sum(mul(y, w), 1i32), 0i32)",
            ),
        ] {
            let (_dir, reef, app) = common::make_app("issue-2990");
            let extent = if axis == "0i32" && shape == "2, 4" {
                2
            } else {
                4
            };
            let program = format!("module Demo.Main\ndef loss(x: tensor[{shape}, P]) -> tensor[P] = {{\n    y = softmax(x, {axis})\n    w = to_tensor({weights})\n    {reduce}\n}}\nx = to_tensor({values})\nw = to_tensor({weights})\ny = softmax(x, {axis})\nactual = grad(loss)(x)\nexpected = mul(y, sub(w, insert(cast(sum(mul(w, y), {axis}), P), {axis}, {extent}i64)))\n").replace('P', dtype);
            common::write_file(&app.join("src/main.ch"), &program);
            let output = Command::cargo_bin("chelis")
                .unwrap()
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .env("CHELIS_REEF_HOME", &reef)
                .current_dir(&app)
                .args(["eval", "--json", "--file", "src/main.ch"])
                .output()
                .unwrap();
            assert!(
                output.status.success(),
                "{dtype}/{shape}/{axis}: {output:?}"
            );
            let report: Value = serde_json::from_slice(&output.stdout).unwrap();
            let roots = report["roots"].as_array().unwrap();
            let actual = roots.iter().find(|r| r["name"] == "actual").unwrap();
            let expected = roots.iter().find(|r| r["name"] == "expected").unwrap();
            assert_eq!(actual["value"], expected["value"], "{dtype}/{shape}/{axis}");
            let native = common::build_and_run_app(&reef, &app, "main");
            let actual = native
                .lines()
                .find_map(|line| line.strip_prefix("actual = "))
                .unwrap();
            let expected = native
                .lines()
                .find_map(|line| line.strip_prefix("expected = "))
                .unwrap();
            assert_eq!(actual, expected, "native {dtype}/{shape}/{axis}");
        }
    }
}

#[test]
fn softmax_still_rejects_nonfloat_operands_and_invalid_axes() {
    let (_dir, reef, app) = common::make_app("issue-2990-negative");
    for call in [
        "softmax(to_tensor([1i32, 2i32]), 0i32)",
        "softmax(to_tensor([1.0f32, 2.0f32]), 1i32)",
    ] {
        common::write_file(
            &app.join("src/main.ch"),
            &format!("module Demo.Main\nresult = {call}\n"),
        );
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef)
            .current_dir(&app)
            .args(["check", "src/main.ch"])
            .output()
            .unwrap();
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        assert!(
            !output.status.success() && !report["errors"].as_array().unwrap().is_empty(),
            "{call}: {report}"
        );
    }
}
