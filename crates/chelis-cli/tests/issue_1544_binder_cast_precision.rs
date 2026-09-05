//! CLI coverage for binder-target literal adoption (#1544/#1553).

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::{Command as StdCommand, Output};
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

const EXACT_F64: &str = "0.30000000000000004";
const ROUNDED_VIA_F32: &str = "0.30000000447034836";

fn make_package(name: &str, source: &str) -> (TempDir, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join(name);
    fs::create_dir_all(root.join("src")).expect("create src");
    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\nmodule_prefix = \"Bind\"\n",
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    )
    .expect("write manifest");
    fs::write(root.join("src/main.ch"), source).expect("write source");
    (dir, root)
}

fn run(root: &Path, args: &[&str]) -> Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(args)
        .output()
        .expect("run chelis")
}

fn diagnostics(output: &Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn assert_success(output: &Output, label: &str) {
    assert!(
        output.status.success(),
        "{label} failed: {}",
        diagnostics(output)
    );
}

fn check_then_eval(root: &Path) -> String {
    let check = run(root, &["check", "src/main.ch"]);
    assert_success(&check, "check");
    assert!(
        root.join("reef.lock").exists(),
        "check must write reef.lock"
    );
    let eval = run(root, &["eval", "--file", "src/main.ch"]);
    assert_success(&eval, "eval");
    String::from_utf8_lossy(&eval.stdout).into_owned()
}

fn assert_all_lanes_reject(root: &Path, citation: &str) {
    for args in [
        vec!["check", "src/main.ch"],
        vec!["eval", "--file", "src/main.ch"],
        vec!["build", "src/main.ch", "--target", "c", "--output", "out"],
    ] {
        let output = run(root, &args);
        let text = diagnostics(&output);
        assert!(
            !output.status.success() && text.contains(citation),
            "{} must reject with {citation}: {text}",
            args[0]
        );
    }
}

const SCALAR_SCALE: &str = "module Bind.Main\n\
     export (main)\n\
     def scale[p: Float](x: p) -> p = mul(x, cast(0.1, p))\n\
     def at_f32() -> f32 = scale(3.0f32)\n\
     def at_f64() -> f64 = scale(3.0f64)\n\
     def main() -> f64 = at_f64()\n";

const TENSOR_SCALE: &str = "module Bind.Main\n\
     export (main)\n\
     def scale[p: Float](x: tensor[1, p]) -> tensor[1, p] = mul(x, cast(0.1, p))\n\
     def at_f32() -> tensor[1, f32] = scale(to_tensor([3.0f32]))\n\
     def at_f64() -> tensor[1, f64] = scale(to_tensor([3.0f64]))\n\
     def main() -> tensor[1, f64] = at_f64()\n";

const TENSOR_BINDER_SCALE: &str = "module Bind.Main\n\
     export (main)\n\
     def scale[p: Float](x: tensor[3, p]) -> tensor[3, p] = mul(x, expand(scalar_to_tensor(cast(0.1, p)), cast(0, int32), cast(3, int64)))\n\
     def at_f64() -> tensor[3, f64] = scale(to_tensor([3.0f64, 3.0f64, 3.0f64]))\n\
     def main() -> tensor[3, f64] = at_f64()\n";

#[test]
fn scalar_binder_cast_computes_at_each_instantiation() {
    for (name, source) in [
        ("float-scale", SCALAR_SCALE.to_owned()),
        ("numeric-scale", SCALAR_SCALE.replace("Float", "Numeric")),
    ] {
        let (_dir, root) = make_package(name, &source);
        let stdout = check_then_eval(&root);
        assert!(stdout.contains("at_f32 = 0.3\n"), "{stdout}");
        assert!(
            stdout.contains(&format!("at_f64 = {EXACT_F64}")),
            "must not return {ROUNDED_VIA_F32}: {stdout}"
        );
    }
}

#[test]
fn tensor_binder_cast_computes_at_each_instantiation() {
    let (_dir, root) = make_package("tensor-scale", TENSOR_SCALE);
    let stdout = check_then_eval(&root);
    assert!(
        stdout.contains("at_f32 = tensor(shape=[1], data=[0.3])"),
        "{stdout}"
    );
    assert!(
        stdout.contains(&format!("at_f64 = tensor(shape=[1], data=[{EXACT_F64}])")),
        "must not return {ROUNDED_VIA_F32}: {stdout}"
    );
}

#[test]
fn compiled_c_executes_the_same_exact_value_as_eval() {
    let (_dir, root) = make_package("scalar-scale-build", SCALAR_SCALE);
    let eval = check_then_eval(&root);
    let build = run(
        &root,
        &["build", "src/main.ch", "--target", "c", "--output", "out"],
    );
    assert_success(&build, "build");
    let emitted = fs::read_to_string(root.join("out/main.c")).expect("emitted C");
    assert!(emitted.contains("0x3fb999999999999a"), "{emitted}");
    let linked = common::link_generated(&root.join("out"), "main.c", "binder-cast");
    assert!(linked.success(), "link failed: {linked}");
    let executed = StdCommand::new(root.join("out/binder-cast"))
        .output()
        .expect("run compiled program");
    assert_success(&executed, "compiled program");
    let expected = format!("at_f64 = {EXACT_F64}");
    let compiled = String::from_utf8_lossy(&executed.stdout);
    assert!(
        eval.contains(&expected) && compiled.contains(&expected),
        "eval={eval} compiled={compiled}"
    );
}

#[test]
fn compiled_c_tensor_binder_cast_executes_the_same_exact_value_as_eval() {
    let (_dir, root) = make_package("tensor-cast-build", TENSOR_BINDER_SCALE);
    let eval = check_then_eval(&root);
    let build = run(
        &root,
        &["build", "src/main.ch", "--target", "c", "--output", "out"],
    );
    assert_success(&build, "build");
    let emitted = fs::read_to_string(root.join("out/main.c")).expect("emitted C");
    assert!(emitted.contains("0x3fb999999999999a"), "{emitted}");
    let linked = common::link_generated(&root.join("out"), "main.c", "tensor-binder-cast");
    assert!(linked.success(), "link failed: {linked}");
    let executed = StdCommand::new(root.join("out/tensor-binder-cast"))
        .output()
        .expect("run compiled program");
    assert_success(&executed, "compiled program");
    let expected =
        format!("at_f64 = tensor(shape=[3], data=[{EXACT_F64}, {EXACT_F64}, {EXACT_F64}])");
    let compiled = String::from_utf8_lossy(&executed.stdout);
    assert!(
        eval.contains(&expected) && compiled.contains(&expected),
        "eval={eval} compiled={compiled}"
    );
}

#[test]
fn undeclared_cast_target_remains_unknown() {
    let (_dir, root) = make_package(
        "typo-target",
        "module Bind.Main\nexport (main)\ndef typo(x: f32) -> f32 = cast(x, flt32)\ndef main() -> f32 = typo(1.0f32)\n",
    );
    assert_all_lanes_reject(
        &root,
        "cast target `flt32` is not a recognized primitive type",
    );
}

#[test]
fn unbounded_binder_is_not_a_dtype_target() {
    for (name, literal) in [
        ("positive-unsuffixed", "0.1"),
        ("negative-unsuffixed", "-0.1"),
        ("positive-suffixed", "0.1f64"),
        ("negative-suffixed", "-0.1f64"),
    ] {
        let source = format!(
            "module Bind.Main\nexport (main)\ndef scale[p](x: p) -> p = mul(x, cast({literal}, p))\ndef main() -> f64 = scale(3.0f64)\n"
        );
        let (_dir, root) = make_package(&format!("unbounded-binder-{name}"), &source);
        assert_all_lanes_reject(&root, "[04-DTYPE-1]");
    }
}

#[test]
fn integer_literal_adopts_an_int_binder() {
    let (_dir, root) = make_package(
        "int-binder",
        "module Bind.Main\nexport (main)\ndef addk[p: Int](x: p) -> p = add(x, cast(1, p))\ndef at_i32() -> int32 = addk(41i32)\ndef at_i64() -> int64 = addk(41i64)\ndef main() -> int64 = at_i64()\n",
    );
    let stdout = check_then_eval(&root);
    assert!(
        stdout.contains("at_i32 = 42") && stdout.contains("at_i64 = 42"),
        "{stdout}"
    );
}

#[test]
fn integer_literal_adopts_float_and_numeric_binders() {
    for family in ["Float", "Numeric"] {
        let source = format!(
            "module Bind.Main\nexport (main)\ndef addk[p: {family}](x: p) -> p = add(x, cast(7, p))\ndef at_f32() -> f32 = addk(1.0f32)\ndef at_f64() -> f64 = addk(1.0f64)\ndef main() -> f64 = at_f64()\n"
        );
        let (_dir, root) = make_package(&format!("int-{family}"), &source);
        let stdout = check_then_eval(&root);
        assert!(
            stdout.contains("at_f32 = 8") && stdout.contains("at_f64 = 8"),
            "{family}: {stdout}"
        );
    }
}

#[test]
fn ascription_is_not_a_literal_adoption_rule() {
    let (_dir, root) = make_package(
        "ascribed",
        "module Bind.Main\nexport (main)\ndef scale[p: Float](x: p) -> p = mul(x, (0.1 : p))\ndef main() -> f64 = scale(3.0f64)\n",
    );
    assert_all_lanes_reject(&root, "[04-INF-6]");
}
