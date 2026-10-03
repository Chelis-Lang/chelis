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
            let restored_axis = if axis.starts_with('-') {
                if shape == "2, 4" { "1i32" } else { "0i32" }
            } else {
                axis
            };
            let program = format!("module Demo.Main\ndef loss(x: tensor[{shape}, P]) -> tensor[P] = {{\n    y = softmax(x, {axis})\n    w = to_tensor({weights})\n    {reduce}\n}}\nx = to_tensor({values})\nw = to_tensor({weights})\ny = softmax(x, {axis})\nactual = grad(loss)(x)\nexpected = mul(y, sub(w, insert(cast(sum(mul(w, y), {axis}), P), {restored_axis}, {extent}i64)))\n").replace('P', dtype);
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

#[test]
fn named_batched_softmax_grad_uses_the_same_formula_on_each_row() {
    for dtype in ["f16", "f64"] {
        let (_dir, reef, app) = common::make_app("issue-2990-batched");
        let source = "module Demo.Main\ndef loss(x: tensor[4, P]) -> tensor[P] = sum(mul(softmax(x, -1i32), to_tensor([1.0P, 2.0P, 3.0P, 4.0P])), 0i32)\nx = to_tensor([[0.1P, 0.7P, 1.3P, 2.9P], [2.0P, 2.0P, -1.0P, 0.0P]])\ny = softmax(x, 1i32)\nw = to_tensor([[1.0P, 2.0P, 3.0P, 4.0P], [1.0P, 2.0P, 3.0P, 4.0P]])\nactual = vmap(grad(loss))(x)\nexpected = mul(y, sub(w, insert(cast(sum(mul(w, y), 1i32), P), 1i32, 4i64)))\n".replace('P', dtype);
        common::write_file(&app.join("src/main.ch"), &source);
        let output = Command::cargo_bin("chelis")
            .unwrap()
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_REEF_HOME", &reef)
            .current_dir(&app)
            .args(["eval", "--json", "--file", "src/main.ch"])
            .output()
            .unwrap();
        assert!(output.status.success(), "{dtype}: {output:?}");
        let report: Value = serde_json::from_slice(&output.stdout).unwrap();
        let roots = report["roots"].as_array().unwrap();
        let actual = roots.iter().find(|r| r["name"] == "actual").unwrap();
        let expected = roots.iter().find(|r| r["name"] == "expected").unwrap();
        assert_eq!(actual["value"], expected["value"], "batched {dtype}");
        let native = common::build_and_run_app(&reef, &app, "main");
        let actual = native
            .lines()
            .find_map(|l| l.strip_prefix("actual = "))
            .unwrap();
        let expected = native
            .lines()
            .find_map(|l| l.strip_prefix("expected = "))
            .unwrap();
        assert_eq!(actual, expected, "native batched {dtype}");
    }
}

#[test]
fn softmax_f64_adjoint_agrees_with_independent_finite_differences() {
    let (_dir, reef, app) = common::make_app("issue-2990-finite-difference");
    common::write_file(
        &app.join("src/main.ch"),
        "module Demo.Main\ndef loss(x: tensor[4, f64]) -> tensor[f64] = sum(mul(softmax(x, 0i32), to_tensor([1.0f64, 2.0f64, 3.0f64, 4.0f64])), 0i32)\nactual = grad(loss)(to_tensor([0.1f64, 0.7f64, 1.3f64, 2.9f64]))\n",
    );
    let output = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef)
        .current_dir(&app)
        .args(["eval", "--json", "--file", "src/main.ch"])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    let report: Value = serde_json::from_slice(&output.stdout).unwrap();
    let root = report["roots"]
        .as_array()
        .unwrap()
        .iter()
        .find(|r| r["name"] == "actual")
        .unwrap();
    let bits = root["value"]["value"]["data"]["bits"].as_array().unwrap();
    assert_eq!(bits.len(), 4);
    let loss = |x: [f64; 4]| {
        let exponentials = x.map(f64::exp);
        let denominator: f64 = exponentials.iter().sum();
        exponentials
            .iter()
            .enumerate()
            .map(|(i, e)| e / denominator * (i + 1) as f64)
            .sum::<f64>()
    };
    for (i, bits) in bits.iter().enumerate() {
        let actual = f64::from_bits(u64::from_str_radix(bits.as_str().unwrap(), 16).unwrap());
        let mut plus = [0.1, 0.7, 1.3, 2.9];
        let mut minus = plus;
        let epsilon = 1e-5;
        plus[i] += epsilon;
        minus[i] -= epsilon;
        let finite_difference = (loss(plus) - loss(minus)) / (2.0 * epsilon);
        assert!(
            (actual - finite_difference).abs() < 1e-9,
            "component {i}: {actual} vs {finite_difference}"
        );
    }
}
