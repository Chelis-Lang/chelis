//! CLI coverage for binder-target literal adoption (#1544/#1553).

use assert_cmd::Command;
use std::{fs, path::Path, process::Command as StdCommand};
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

const EXACT_F64: &str = "0.30000000000000004";
const SCALAR: &str = "module Bind.Main\nexport (main)\n\
 def scale[p: Float](x: p) -> p = mul(x, cast(0.1, p))\n\
 def at_f32() -> f32 = scale(3.0f32)\n\
 def at_f64() -> f64 = scale(3.0f64)\n\
 def main() -> f64 = at_f64()\n";
const TENSOR: &str = "module Bind.Main\nexport (main)\n\
 def scale[p: Float](x: tensor[1, p]) -> tensor[1, p] = mul(x, cast(0.1, p))\n\
 def at_f32() -> tensor[1, f32] = scale(to_tensor([3.0f32]))\n\
 def at_f64() -> tensor[1, f64] = scale(to_tensor([3.0f64]))\n\
 def main() -> tensor[1, f64] = at_f64()\n";
const TENSOR_NATIVE: &str = "module Bind.Main\nexport (main)\n\
 def scale[p: Float](x: tensor[3, p]) -> tensor[3, p] = mul(x, expand(scalar_to_tensor(cast(0.1, p)), cast(0, int32), cast(3, int64)))\n\
 def at_f64() -> tensor[3, f64] = scale(to_tensor([3.0f64, 3.0f64, 3.0f64]))\n\
 def main() -> tensor[3, f64] = at_f64()\n";

fn package(name: &str, source: &str) -> (TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let root = dir.path().join(name);
    fs::create_dir_all(root.join("src")).expect("src");
    fs::write(
        root.join("reef.toml"),
        format!(
            "[package]\nname = \"{name}\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\nmodule_prefix = \"Bind\"\n",
            chelis_compiler_api::COMPILER_VERSION
        ),
    )
    .expect("manifest");
    fs::write(root.join("src/main.ch"), source).expect("source");
    (dir, root)
}

fn run(root: &Path, args: &[&str]) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(root)
        .args(args)
        .output()
        .expect("run")
}

fn text(output: &std::process::Output) -> String {
    format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    )
}

fn success(output: &std::process::Output) {
    assert!(output.status.success(), "{}", text(output));
}

fn eval(root: &Path) -> String {
    let check = run(root, &["check", "src/main.ch"]);
    success(&check);
    assert!(root.join("reef.lock").exists());
    let output = run(root, &["eval", "--file", "src/main.ch"]);
    success(&output);
    String::from_utf8_lossy(&output.stdout).into_owned()
}

fn reject(source: &str, citation: &str) {
    let (_dir, root) = package("rejected", source);
    for args in [
        &["check", "src/main.ch"][..],
        &["eval", "--file", "src/main.ch"],
        &["build", "src/main.ch", "--target", "c", "--output", "out"],
    ] {
        let output = run(&root, args);
        assert!(!output.status.success() && text(&output).contains(citation));
    }
}

fn assert_native(name: &str, source: &str, expected: &str) {
    let (_dir, root) = package(name, source);
    let interpreted = eval(&root);
    let built = run(
        &root,
        &["build", "src/main.ch", "--target", "c", "--output", "out"],
    );
    success(&built);
    let linked = common::link_generated(&root.join("out"), "main.c", name);
    assert!(linked.success(), "{linked}");
    let output = StdCommand::new(root.join("out").join(name))
        .output()
        .expect("native");
    success(&output);
    let native = String::from_utf8_lossy(&output.stdout);
    assert!(interpreted.contains(expected) && native.contains(expected));
}

#[test]
fn scalar_and_tensor_binders_compute_at_each_instantiation() {
    for family in ["Float", "Numeric"] {
        let source = SCALAR.replace("Float", family);
        let (_dir, root) = package(family, &source);
        let output = eval(&root);
        assert!(output.contains("at_f32 = 0.3"));
        assert!(output.contains(&format!("at_f64 = {EXACT_F64}")));
    }
    let (_dir, root) = package("tensor", TENSOR);
    let output = eval(&root);
    assert!(output.contains("at_f32 = tensor(shape=[1], data=[0.3])"));
    assert!(output.contains(&format!("at_f64 = tensor(shape=[1], data=[{EXACT_F64}])")));
}

#[test]
fn native_scalar_and_tensor_match_eval_exactly() {
    assert_native("scalar", SCALAR, &format!("at_f64 = {EXACT_F64}"));
    let expected =
        format!("at_f64 = tensor(shape=[3], data=[{EXACT_F64}, {EXACT_F64}, {EXACT_F64}])");
    assert_native("tensor", TENSOR_NATIVE, &expected);
}

#[test]
fn undeclared_and_unbounded_targets_fail_all_lanes() {
    reject(
        "module Bind.Main\nexport (main)\ndef typo(x: f32) -> f32 = cast(x, flt32)\ndef main() -> f32 = typo(1.0f32)\n",
        "cast target `flt32` is not a recognized primitive type",
    );
    for literal in ["0.1", "-0.1", "0.1f64", "-0.1f64"] {
        reject(
            &format!(
                "module Bind.Main\nexport (main)\ndef scale[p](x: p) -> p = mul(x, cast({literal}, p))\ndef main() -> f64 = scale(3.0f64)\n"
            ),
            "[04-DTYPE-1]",
        );
    }
}

#[test]
fn integer_literals_adopt_every_compatible_family() {
    for family in ["Int", "Float", "Numeric"] {
        let ty = if family == "Int" { "int64" } else { "f64" };
        let source = format!(
            "module Bind.Main\nexport (main)\ndef addk[p: {family}](x: p) -> p = add(x, cast(1, p))\ndef main() -> {ty} = addk(41{suffix})\n",
            suffix = if family == "Int" { "i64" } else { ".0f64" }
        );
        let (_dir, root) = package(family, &source);
        assert!(eval(&root).contains("main = 42"));
    }
}

#[test]
fn adopted_integer_literals_must_fit_every_family_member() {
    for (family, literal) in [
        ("Int", "128"),
        ("Int", "-129"),
        ("Numeric", "128"),
        ("Numeric", "-129"),
    ] {
        reject(
            &format!(
                "module Bind.Main\nexport (main)\ndef value[p: {family}](x: p) -> p = cast({literal}, p)\ndef main() -> int64 = value(0i64)\n"
            ),
            "§5.6",
        );
    }
}

#[test]
fn float_family_literal_finalization_remains_total() {
    let source = "module Bind.Main\nexport (main)\n\
                  def value[p: Float](x: p) -> p = cast(10000000000.0, p)\n\
                  def main() -> f16 = value(0.0f16)\n";
    let (_dir, root) = package("float_range", source);
    success(&run(&root, &["check", "src/main.ch"]));
}

#[test]
fn ascription_is_not_literal_adoption() {
    reject(
        "module Bind.Main\nexport (main)\ndef scale[p: Float](x: p) -> p = mul(x, (0.1 : p))\ndef main() -> f64 = scale(3.0f64)\n",
        "[04-INF-6]",
    );
}
