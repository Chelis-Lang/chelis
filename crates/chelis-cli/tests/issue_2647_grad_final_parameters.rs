//! Checker, evaluator and native C agree on the final inferred Grad payload.
use std::process::Command;

#[path = "common/mod.rs"]
mod common;

fn eval(source: &str) -> String {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("grad.ch");
    std::fs::write(&path, source).unwrap();
    let output = Command::new(assert_cmd::cargo_bin!("chelis"))
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .unwrap();
    assert!(output.status.success(), "{output:?}");
    String::from_utf8(output.stdout).unwrap()
}

#[test]
fn final_parameter_types_determine_evaluator_payloads() {
    for (name, source, expected) in [
        (
            "recursive_result_equality",
            "def first(p, flag: bool) = second(p, flag)\ndef second(p, flag: bool) = if flag then p else first(1.0f32, true)\nout = first(2.0f32, false)\n",
            "out = 1.0",
        ),
        (
            "grad_inferred_float",
            "def result(ignored: unit) -> (f32, f32) = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, 1.0f32)\nout = result(())\n",
            "out.0 = 6.0\nout.1 = 0.0",
        ),
        (
            "grad_inferred_discrete",
            "def result(ignored: unit) -> f32 = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, true)\nout = result(())\n",
            "out = 6.0",
        ),
        (
            "grad_inferred_selected",
            "def result(ignored: unit) -> f32 = grad(fn (z: f32, w) -> mul(z, z), wrt=w)(3.0f32, 1.0f32)\nout = result(())\n",
            "out = 0.0",
        ),
        (
            "grad_inferred_nested",
            "def result(ignored: unit) -> (f32, (f32, unit)) = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, (1.0f32, true))\nout = result(())\n",
            "out.0 = 6.0\nout.1.0 = 0.0\nout.1.1 = ()",
        ),
        (
            "grad_inferred_let",
            "def result(ignored: unit) -> (f32, f32) = {\n g = grad(fn (z: f32, w) -> mul(z, z))\n g(3.0f32, 1.0f32)\n}\nout = result(())\n",
            "out.0 = 6.0\nout.1 = 0.0",
        ),
    ] {
        assert_eq!(eval(source).trim(), expected, "{name}");
    }
}

#[test]
fn incorrect_gradient_payloads_fail_at_checking() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("grad.ch");
    for source in [
        "def main() -> bool = {\n op = fold\n op(fn (acc: f32, item: i64) -> acc, 1.0f32, [1i64])\n}",
        "def main() -> bool = {\n op = scan\n op(fn (acc: f32, item: i64) -> acc, 1.0f32, [1i64])\n}",
        "def main() = dict_insert(dict_of([(\"a\", grad(fn (w: f32) -> 1.0f32))]), \"b\", grad(fn (w) -> 1.0f32))",
        "def main() = dict_merge(dict_of([(\"a\", grad(fn (w) -> 1.0f32))]), dict_of([(\"b\", grad(fn (w: f32) -> 1.0f32))]))",
        "def pick(p) = index(dict_values(dict_insert(dict_of([(\"a\", grad(fn (w: f32) -> 1.0f32))]), \"b\", p)), 0i64)\ndef main() = pick(grad(fn (w) -> 1.0f32))",
        "def pick(p) = index(dict_values(dict_merge(dict_of([(\"a\", p)]), dict_of([(\"b\", grad(fn (w: f32) -> 1.0f32))]))), 0i64)\ndef main() = pick(grad(fn (w) -> 1.0f32))",
        "def main() = fold(fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), grad(fn (w) -> 1.0f32), [1i64])",
        "def main() = scan(fn (acc, item: i64) -> grad(fn (w: f32) -> 1.0f32), grad(fn (w) -> 1.0f32), [1i64])",
        "def main() = concat([grad(fn (w) -> 1.0f32)], [grad(fn (w: f32) -> 1.0f32)])",
        "def main() = append([grad(fn (w) -> 1.0f32)], grad(fn (w: f32) -> 1.0f32))",
        "def first(p, flag: bool) = second(p, flag)\ndef second(p, flag: bool) = if flag then p else first(1.0f32, true)\ndef main() -> bool = first(true, false)\nout = main()",
        "def first(p, flag: bool) = second(p, flag)\ndef second(p, flag: bool) = if flag then p else first(grad(fn (w: f32) -> 1.0f32), true)\nout = (first(grad(fn (w) -> 1.0f32), false))(true)",
        "def main() -> f32 = grad(fn (z: f32, w) -> mul(z, z))(3.0f32, 1.0f32)",
        "def main() -> f32 = grad(fn (z: f32, w) -> mul(z, z), wrt=w)(3.0f32, true)",
        "def main() = grad(fn (z: f32, w) -> mul(z, z))",
        "def make_grad() -> (f32 -> f32) = grad(fn (w) -> 1.0f32)",
        "def make_grad() = {\n g: (f32 -> f32) = grad(fn (w) -> 1.0f32)\n g\n}",
        "def main() = {\n make = fn (f) -> {\n d: (f32 -> f32) = grad(f)\n d\n }\n make(fn (w) -> 1.0f32)\n}",
        "def main(flag: bool) = if flag then grad(fn (w) -> 1.0f32) else grad(fn (w: f32) -> 1.0f32)",
        "def main(flag: bool) -> (f32 -> f32) = if flag then grad(fn (w) -> 1.0f32) else main(flag)",
    ] {
        std::fs::write(&path, source).unwrap();
        let output = Command::new(assert_cmd::cargo_bin!("chelis"))
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{source}\n{output:?}");
    }
}

#[test]
fn inferred_tensor_gradients_execute_in_eval_and_native_c() {
    let source = "out = grad(fn (z: tensor[2, f32], w) -> tensor_to_scalar(sum(mul(z, z), 0i32)))(to_tensor([3.0f32, 4.0f32]), to_tensor([1.0f32, 2.0f32]))\n";
    let expected =
        "out.0 = tensor(shape=[2], data=[6.0, 8.0])\nout.1 = tensor(shape=[2], data=[0.0, 0.0])";
    assert_eq!(eval(source).trim(), expected);
    assert_eq!(
        common::build_and_run(source, "grad_inferred_tensors").trim(),
        expected
    );
}

#[test]
fn native_scalar_lambda_admission_matches_the_annotated_control() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("grad.ch");
    for parameter in ["w", "w: f32"] {
        std::fs::write(
            &path,
            format!("out = grad(fn (z: f32, {parameter}) -> mul(z, z))(3.0f32, 1.0f32)\n"),
        )
        .unwrap();
        let output = Command::new(assert_cmd::cargo_bin!("chelis"))
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                directory.path().join("native").to_str().unwrap(),
            ])
            .output()
            .unwrap();
        assert!(!output.status.success(), "{output:?}");
        assert!(
            String::from_utf8_lossy(&output.stderr).contains("applies/binds `grad`"),
            "{output:?}"
        );
    }
}

#[test]
fn accumulator_aliases_check_at_their_required_result_types() {
    let directory = tempfile::tempdir().unwrap();
    let path = directory.path().join("accumulator_alias.ch");
    for (operation, output) in [("fold", "f32"), ("scan", "List[f32]")] {
        let source = format!(
            "def main() -> {output} = {{\n op = {operation}\n op(fn (acc: f32, item: i64) -> acc, 1.0f32, [1i64])\n}}"
        );
        std::fs::write(&path, source).unwrap();
        let output = Command::new(assert_cmd::cargo_bin!("chelis"))
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", path.to_str().unwrap()])
            .output()
            .unwrap();
        assert!(output.status.success(), "{output:?}");
        let payload: serde_json::Value = serde_json::from_slice(&output.stdout).unwrap();
        assert_eq!(payload["errors"], serde_json::json!([]));
    }
}
