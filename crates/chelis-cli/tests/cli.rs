use assert_cmd::Command;
use chelis_shell::{ShellSymbol, SymbolKind, read_shell, write_shell};
use predicates::prelude::*;
use serde_json::Value;
use std::fs;
use std::path::{Path, PathBuf};
use tempfile::tempdir;

fn example_path(rel: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join(rel)
        .canonicalize()
        .expect("example path should exist")
}

fn mnist_example() -> PathBuf {
    example_path("../../examples/mnist.ch")
}

fn hello_tensor_example() -> PathBuf {
    example_path("../../examples/hello_tensor.ch")
}

fn vmap_example() -> PathBuf {
    example_path("../../examples/vmap_relu.ch")
}

fn scalar_string_foundation_example() -> PathBuf {
    example_path("../../examples/scalar_string_foundation.ch")
}

fn linreg_example() -> PathBuf {
    example_path("../../examples/linreg.ch")
}

fn transformer_block_example() -> PathBuf {
    example_path("../../examples/transformer_block.ch")
}

fn executable_examples() -> [PathBuf; 6] {
    [
        hello_tensor_example(),
        linreg_example(),
        mnist_example(),
        scalar_string_foundation_example(),
        transformer_block_example(),
        vmap_example(),
    ]
}

fn illustrative_example(name: &str) -> PathBuf {
    example_path(&format!("../../examples/illustrative/{name}"))
}

fn package_std() -> PathBuf {
    example_path("../../packages/chelis-std")
}

fn editor_file(rel: &str) -> PathBuf {
    example_path(&format!("../../editors/vscode/{rel}"))
}

fn write_matmul_program(path: &Path) {
    fs::write(
        path,
        r#"let a = (a : tensor[2, 3, f32])
let b = (b : tensor[3, 4, f32])
let out = (matmul(a, b) : tensor[2, 4, f32])
"#,
    )
    .expect("write matmul program");
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dst dir");
    for entry in fs::read_dir(src).expect("read dir") {
        let entry = entry.expect("dir entry");
        let path = entry.path();
        let target = dst.join(entry.file_name());
        if path.is_dir() {
            copy_dir_recursive(&path, &target);
        } else {
            fs::copy(&path, &target).expect("copy file");
        }
    }
}

fn write_symbolic_matmul_program(path: &Path) {
    write_file(
        path,
        "def f(a: tensor[batch, in_dim, f32], b: tensor[in_dim, out_dim, f32]): tensor[batch, out_dim, f32] = (matmul(a, b) : tensor[batch, out_dim, f32])\n",
    );
}

fn write_symbolic_softmax_program(path: &Path) {
    write_file(
        path,
        "def f(x: tensor[batch, seq, f32]): tensor[batch, seq, f32] = (softmax(x, 1) : tensor[batch, seq, f32])\n",
    );
}

fn write_symbolic_row_sum_program(path: &Path) {
    write_file(
        path,
        "def f(x: tensor[batch, seq, f32]): tensor[batch, f32] = (sum(x, 1) : tensor[batch, f32])\n",
    );
}

fn write_symbolic_layer_norm_program(path: &Path) {
    write_file(
        path,
        "def f(x: tensor[batch, 128, f32], gamma: tensor[128, f32], beta: tensor[128, f32]): tensor[batch, 128, f32] = (layer_norm(x, gamma, beta) : tensor[batch, 128, f32])\n",
    );
}

fn write_symbolic_hidden_layer_norm_program(path: &Path) {
    write_file(
        path,
        "def f(x: tensor[batch, hidden, f32], gamma: tensor[hidden, f32], beta: tensor[hidden, f32]): tensor[batch, hidden, f32] = (layer_norm(x, gamma, beta) : tensor[batch, hidden, f32])\n",
    );
}

fn run_json_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("check output should be json")
}

#[test]
fn check_accepts_executable_examples() {
    for path in executable_examples() {
        let json = run_json_check(&path);
        assert_eq!(json["score"].as_f64().unwrap(), 1.0, "{path:?}");
        assert_eq!(json["errors"].as_array().unwrap().len(), 0, "{path:?}");
        assert_eq!(
            json["unresolved_names"].as_array().unwrap().len(),
            0,
            "{path:?}"
        );
    }
}

#[test]
fn fmt_check_accepts_canonical_executable_examples() {
    for path in executable_examples() {
        Command::cargo_bin("chelis")
            .expect("binary")
            .args(["fmt", path.to_str().unwrap(), "--check"])
            .assert()
            .success();
    }
}

#[test]
fn eval_rejects_missing_inputs() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", mnist_example().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("missing required input"));
}

#[test]
fn eval_prints_labeled_tuple_components() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("tuple_eval.ch");
    write_file(
        &path,
        r#"let grads = (1.0, 2.5)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("grads.0 = tensor(shape=[], data=[1.0])"))
        .stdout(predicate::str::contains("grads.1 = tensor(shape=[], data=[2.5])"));
}

#[test]
fn eval_supports_host_scalars_and_strings() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            r#"if 3 > 2 then string_concat("ok-", to_string(string_len("hé"))) else "bad""#,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("ok-2"));
}

#[test]
fn eval_supports_integer_mod_and_bitwise_helpers() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            r#"bitxor(bitand(cast(7, int64), cast(3, int64)), shl(cast(1, int64), cast(2, int64)))"#,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("7"));
}

#[test]
fn eval_surfaces_debug_transcript() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", r#"debug("trace")"#])
        .assert()
        .success()
        .stdout(predicate::str::contains("trace\ntrace"));
}

#[test]
fn phase3c_scalar_string_acceptance_oracle() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            "--file",
            scalar_string_foundation_example().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("progress: ckpt-7.safetensors"))
        .stdout(predicate::str::contains("stop"));
}

#[test]
fn fmt_inplace_preserves_mnist_executability() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mnist.ch");
    fs::copy(mnist_example(), &path).expect("copy");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap(), "--inplace"])
        .assert()
        .success();

    let json = run_json_check(&path);
    assert_eq!(json["score"].as_f64().unwrap(), 1.0);
    assert_eq!(json["errors"].as_array().unwrap().len(), 0);
}

#[test]
fn fmt_inplace_preserves_pipeline_parseability() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pipeline.ch");
    fs::copy(illustrative_example("pipeline.ch"), &path).expect("copy");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap(), "--inplace"])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", path.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn fmt_check_succeeds_for_canonical_deep() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("program.dp");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", hello_tensor_example().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&path, output).expect("write deep");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap(), "--check"])
        .assert()
        .success();
}

#[test]
fn fmt_check_succeeds_for_canonical_surf() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mnist.ch");
    fs::copy(mnist_example(), &path).expect("copy");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap(), "--inplace"])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap(), "--check"])
        .assert()
        .success();
}

#[test]
fn fmt_check_accepts_trailing_newline_terminated_canonical_surf() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("program.ch");
    write_file(
        &path,
        "def f(x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap(), "--check"])
        .assert()
        .success();
}

#[test]
fn fmt_check_fails_for_noncanonical_deep() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("program.dp");
    write_file(
        &path,
        "(app {} (var {} mean) (app {} (var {} neg) (app {} (var {} sum) (var {} very_long_intermediate_name) (lit {type: (t-prim {} i32)} 0))))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap(), "--check"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("is not canonically formatted"));
}

#[test]
fn fmt_rejects_check_and_inplace_together() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("program.dp");
    write_file(&path, "(def {} x (lit {type: (t-prim {} int32)} 1))\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap(), "--check", "--inplace"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "does not allow `--inplace` and `--check` together",
        ));
}

#[test]
fn surf_default_collapses_load_program_into_typed_def() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("forward.ch");
    write_matmul_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "def forward(a: tensor[2, 3, f32], b: tensor[3, 4, f32]) -> tensor[2, 4, f32] =",
        ))
        .stdout(predicate::str::contains("matmul(a, b)"))
        .stdout(predicate::str::contains("let a").not());
}

#[test]
fn surf_verbose_preserves_debug_style_annotations() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("forward.ch");
    write_matmul_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf", path.to_str().unwrap(), "--verbose"])
        .assert()
        .success()
        .stdout(predicate::str::contains("let a = (a : tensor[2, 3, f32])"))
        .stdout(predicate::str::contains(
            "(matmul(a, b) : tensor[2, 4, f32])",
        ));
}

#[test]
fn surf_roundtrip_canonicalizes_def_return_types_to_arrow() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("typed.ch");
    write_file(
        &path,
        "def f(x: tensor[n, f32]): tensor[n, f32] = relu(x)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "def f(x: tensor[n, f32]) -> tensor[n, f32] =",
        ))
        .stdout(predicate::str::contains("sig f").not());
}

#[test]
fn phase3e_pipe_first_acceptance_oracle() {
    let dir = tempdir().expect("tempdir");
    let surf_path = dir.path().join("mnist.ch");
    let deep_path = dir.path().join("mnist.dp");
    fs::copy(mnist_example(), &surf_path).expect("copy");

    let deep = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", surf_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&deep_path, deep).expect("write deep program");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf", deep_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("let h1").not())
        .stdout(predicate::str::contains("let logits").not())
        .stdout(predicate::str::contains(
            "softmax(logits, 1)\n  |> log\n  |> mul(labels)\n  |> sum(1)\n  |> neg\n  |> mean(0)",
        ))
        .stdout(predicate::str::contains("(softmax(logits, 1) :").not())
        .stdout(predicate::str::contains("(matmul(x, w1) :").not());
}

#[test]
fn reef_init_scaffolds_valid_package() {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join("demo");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "init",
            "demo",
            "--module-prefix",
            "Demo",
            "--output",
            pkg.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(pkg.join("reef.toml").exists());
    assert!(pkg.join("src/main.ch").exists());

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn reef_build_emits_shell_and_archive() {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join("chelis-std");
    copy_dir_recursive(&package_std(), &pkg);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "build", pkg.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Built chelis-std 0.1.0"));

    assert!(pkg.join("reef.lock").exists());
    assert!(pkg.join("dist/chelis-std-0.1.0.chb").exists());
    assert!(pkg.join("dist/chelis-std-0.1.0.tar.zst").exists());
}

#[test]
fn phase3a_reef_std_acceptance_oracle() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("demo-app");
    let out_dir = dir.path().join("out");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "demo-app"
version = "0.1.0"
compiler = "=0.1.0"
module_prefix = "Demo"

[dependencies]
chelis-std = { version = "0.1.0" }
"#,
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Linear (forward)
import Std.Loss.CrossEntropy (loss)
import Std.Init.Xavier (sample)

export (main)

def main(
  x: tensor[32, 784, f32],
  w: tensor[784, 10, f32],
  b: tensor[10, f32]
) -> tensor[32, 10, f32] =
  forward(x, w, b)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(app_pkg.join("reef.lock").exists());
    assert!(out_dir.join("main.c").exists());
    assert!(out_dir.join("main.h").exists());
}

#[test]
fn reef_check_accepts_sig_only_shell_imports() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("sig-app");
    copy_dir_recursive(&package_std(), &std_pkg);
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "sig-app"
version = "0.1.0"
compiler = "=0.1.0"
module_prefix = "Demo"

[dependencies]
chelis-std = { version = "0.1.0" }
"#,
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.IO.Safetensors (load_tensors)

export (main)

def main(path: string) -> string =
  load_tensors(path)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));
}

#[test]
fn reef_check_accepts_path_dependencies() {
    let dir = tempdir().expect("tempdir");
    let dep_pkg = dir.path().join("dep");
    let app_pkg = dir.path().join("app");
    fs::create_dir_all(dep_pkg.join("src")).expect("mkdir dep src");
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    write_file(
        &dep_pkg.join("reef.toml"),
        r#"[package]
name = "dep"
version = "0.1.0"
compiler = "=0.1.0"
module_prefix = "Common"
"#,
    );
    write_file(
        &dep_pkg.join("src/helper.ch"),
        r#"module Common.Helper

export (shared)

def shared(x: f32) -> f32 = x
"#,
    );

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "app"
version = "0.1.0"
compiler = "=0.1.0"
module_prefix = "Demo"

[dependencies]
dep = { path = "../dep" }
"#,
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Common.Helper (shared)

export (main)

def main(x: f32) -> f32 = shared(x)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));
}

#[test]
fn reef_check_rejects_tampered_registry_shell_exports() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let dep_pkg = dir.path().join("dep");
    let app_pkg = dir.path().join("app");
    fs::create_dir_all(dep_pkg.join("src")).expect("mkdir dep src");
    fs::create_dir_all(app_pkg.join("src")).expect("mkdir app src");

    write_file(
        &dep_pkg.join("reef.toml"),
        r#"[package]
name = "dep"
version = "0.1.0"
compiler = "=0.1.0"
module_prefix = "Common"
"#,
    );
    write_file(
        &dep_pkg.join("src/api.ch"),
        r#"module Common.Api

export (public)

def public(x: f32) -> f32 = hidden(x)
def hidden(x: f32) -> f32 = x
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", dep_pkg.to_str().unwrap()])
        .assert()
        .success();

    let shell_path = reef_home.join("packages/dep/0.1.0/dep-0.1.0.chb");
    let mut shell = read_shell(&shell_path).expect("read published shell");
    let module = shell
        .modules
        .iter_mut()
        .find(|module| module.module == "Common.Api")
        .expect("Common.Api shell module");
    let public = module
        .exports
        .iter()
        .find(|symbol| symbol.name == "public")
        .expect("public export")
        .clone();
    module.exports.push(ShellSymbol {
        name: "hidden".to_string(),
        kind: SymbolKind::Value,
        type_repr: public.type_repr,
        effects: public.effects,
        has_body: true,
    });
    module.exports.sort_by(|a, b| a.name.cmp(&b.name));
    write_shell(&shell_path, &shell).expect("rewrite shell");

    write_file(
        &app_pkg.join("reef.toml"),
        r#"[package]
name = "app"
version = "0.1.0"
compiler = "=0.1.0"
module_prefix = "Demo"

[dependencies]
dep = { version = "0.1.0" }
"#,
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Common.Api (hidden)

export (main)

def main(x: f32) -> f32 = hidden(x)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("checksum")
                .or(predicate::str::contains("does not export `hidden`")),
        );
}

#[test]
fn check_does_not_report_perfect_score_with_errors() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad_check.ch");
    write_file(&path, "let x = add(1, true)\n");
    let json = run_json_check(&path);
    assert!(json["score"].as_f64().unwrap() < 1.0);
    assert!(!json["errors"].as_array().unwrap().is_empty());
}

#[test]
fn build_creates_missing_output_directory() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("nested/output");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            mnist_example().to_str().unwrap(),
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    assert!(out_dir.join("mnist.c").exists());
    assert!(out_dir.join("mnist.h").exists());
    assert!(out_dir.join("chelis_runtime.c").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
}

#[test]
fn build_hip_accepts_symbolic_dims_and_binds_them_from_input_metadata() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic.ch");
    let out_dir = dir.path().join("hip-out");
    write_file(
        &path,
        "def f(xs: tensor[batch, features, f32]): tensor[batch, features, f32] = xs\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Symbolic dims: batch, features"))
        .stdout(predicate::str::contains("Peak device memory formula:"));

    let source = fs::read_to_string(out_dir.join("symbolic_hip.cpp")).expect("generated source");
    assert!(source.contains("int batch = inputs[0]->shape[0];"));
    assert!(source.contains("int features = inputs[0]->shape[1];"));
}

#[test]
fn build_symbolic_matmul_succeeds_on_c_and_hip_targets() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_matmul.ch");
    let c_out = dir.path().join("c-out");
    let hip_out = dir.path().join("hip-out");
    write_symbolic_matmul_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            c_out.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Symbolic dims: batch, in_dim, out_dim",
        ));

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            hip_out.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "Symbolic dims: batch, in_dim, out_dim",
        ))
        .stdout(predicate::str::contains("Peak device memory formula:"));

    let c_source = fs::read_to_string(c_out.join("symbolic_matmul.c")).expect("generated c");
    assert!(c_source.contains("int batch = inputs[0]->shape[0];"));
    assert!(c_source.contains("int in_dim = inputs[0]->shape[1];"));
    assert!(c_source.contains("inputs[1]->shape[0] != in_dim"));

    let hip_source =
        fs::read_to_string(hip_out.join("symbolic_matmul_hip.cpp")).expect("generated hip");
    assert!(hip_source.contains("int batch = inputs[0]->shape[0];"));
    assert!(hip_source.contains("int in_dim = inputs[0]->shape[1];"));
    assert!(hip_source.contains("inputs[1]->shape[0] != in_dim"));
}

#[test]
fn build_hip_accepts_symbolic_softmax() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_softmax.ch");
    let out_dir = dir.path().join("hip-out");
    write_symbolic_softmax_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Symbolic dims: batch, seq"))
        .stdout(predicate::str::contains("Peak device memory formula:"));

    let source =
        fs::read_to_string(out_dir.join("symbolic_softmax_hip.cpp")).expect("generated source");
    assert!(source.contains("int batch = inputs[0]->shape[0];"));
    assert!(source.contains("int seq = inputs[0]->shape[1];"));
    assert!(source.contains("kernel_maxred_ax1"));
    assert!(source.contains("kernel_sum_ax1"));
}

#[test]
fn build_hip_accepts_symbolic_row_sum() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_sum.ch");
    let out_dir = dir.path().join("hip-out");
    write_symbolic_row_sum_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Symbolic dims: batch, seq"))
        .stdout(predicate::str::contains("Peak device memory formula:"));

    let source =
        fs::read_to_string(out_dir.join("symbolic_sum_hip.cpp")).expect("generated source");
    assert!(source.contains("int batch = inputs[0]->shape[0];"));
    assert!(source.contains("int seq = inputs[0]->shape[1];"));
    assert!(source.contains("kernel_sum_ax1"));
}

#[test]
fn build_hip_accepts_symbolic_leading_dims_for_layer_norm() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_layer_norm.ch");
    let out_dir = dir.path().join("hip-out");
    write_symbolic_layer_norm_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Symbolic dims: batch"))
        .stdout(predicate::str::contains("Peak device memory formula:"));

    let source =
        fs::read_to_string(out_dir.join("symbolic_layer_norm_hip.cpp")).expect("generated source");
    assert!(source.contains("int batch = inputs[0]->shape[0];"));
    assert!(source.contains("kernel_sum_ax1"));
}

#[test]
fn build_hip_rejects_symbolic_normalized_axis_for_layer_norm() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_hidden_layer_norm.ch");
    write_symbolic_hidden_layer_norm_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["build", path.to_str().unwrap(), "--target", "hip"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Phase 0e builtin `layer_norm` requires a concrete normalized axis extent",
        ));
}

#[test]
fn build_hip_rejects_pad_lowering_without_panic() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pad.ch");
    write_file(
        &path,
        "def f(x: tensor[4, f32]): tensor[4, f32] = (pad(x) : tensor[4, f32])\n",
    );

    let json = run_json_check(&path);
    assert_eq!(json["score"].as_f64().unwrap(), 1.0);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["build", path.to_str().unwrap(), "--target", "hip"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("does not yet support `pad`"));
}

#[test]
fn build_hip_creates_missing_output_directory_and_reports_runtime_path() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("nested/hip-output");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            mnist_example().to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            out_dir.join("chelis_runtime.c").display().to_string(),
        ))
        .stdout(predicate::str::contains("Peak device memory formula:"))
        .stdout(predicate::str::contains("Estimated peak device memory:"));

    assert!(out_dir.join("mnist_hip.cpp").exists());
    assert!(out_dir.join("mnist_hip.h").exists());
    assert!(out_dir.join("chelis_runtime.c").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("chelis_hip_runtime.h").exists());
}

#[test]
fn build_hip_mnist_emits_fused_kernels_and_launches() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("hip-output");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            mnist_example().to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let hip_src = fs::read_to_string(out_dir.join("mnist_hip.cpp")).expect("hip source");
    assert!(
        hip_src.contains("kernel_fused_") || hip_src.contains("kernel_fused_sum_"),
        "MNIST HIP build should emit fused kernels on the real CLI path"
    );
    assert!(
        hip_src.contains("chelis_launch_kernel"),
        "MNIST HIP build should emit kernel launches on the real CLI path"
    );
}

#[test]
fn build_hip_matmul_surfaces_hipblas_link_flag_when_specialized() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("hip-output");
    let source = dir.path().join("matmul.ch");
    write_matmul_program(&source);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("-lhipblas"));

    let hip_src = fs::read_to_string(out_dir.join("matmul_hip.cpp")).expect("hip source");
    assert!(
        hip_src.contains("chelis_hipblas_sgemm_row_major"),
        "HIP build should surface hipBLAS specialization for a simple matmul program"
    );
}

#[test]
fn check_reports_unhandled_random_effect() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dropout.ch");
    write_file(
        &path,
        "let x: tensor[32, f32] = x\nlet y: tensor[32, f32] = dropout(x, 0.5)\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().unwrap();
    assert!(errors.iter().any(|error| {
        error["kind"].as_str() == Some("UnhandledEffect")
            && error["message"]
                .as_str()
                .is_some_and(|message| message.contains("Random"))
    }));
    assert!(json["score"].as_f64().unwrap() < 1.0);
}

#[test]
fn check_reports_linearity_errors() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("linearity.ch");
    write_file(
        &path,
        "def bad(x: tensor[4, f32]): tensor[4, f32] = let y: tensor[4, f32] = relu(x) in add(x, y)\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().unwrap();
    assert!(errors.iter().any(|error| {
        error["kind"].as_str() == Some("UseAfterConsume")
            && error["message"]
                .as_str()
                .is_some_and(|message| message.contains("relu"))
    }));
    assert!(json["score"].as_f64().unwrap() < 1.0);
}

#[test]
fn check_reports_macro_provenance_for_type_errors() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("macro_type_error.ch");
    write_file(
        &path,
        r#"
macro bad_bool(x) = and(x, x)
def bad(x: tensor[4, f32]): tensor[4, f32] = bad_bool(x)
"#,
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().unwrap();
    assert!(errors.iter().any(|error| {
        error["message"]
            .as_str()
            .is_some_and(|message| message.contains("in expansion of (bad_bool"))
    }));
}

#[test]
fn check_reports_macro_provenance_for_linearity_errors() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("macro_linearity.ch");
    write_file(
        &path,
        r#"
macro dup_relu(x) = add(relu(x), x)
def bad(x: tensor[4, f32]): tensor[4, f32] = dup_relu(x)
"#,
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().unwrap();
    assert!(errors.iter().any(|error| {
        error["kind"].as_str() == Some("UseAfterConsume")
            && error["message"]
                .as_str()
                .is_some_and(|message| message.contains("in expansion of (dup_relu"))
    }));
}

#[test]
fn check_reports_match_linearity_without_old_phase0e_rejection() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("match_linearity.ch");
    write_file(
        &path,
        r#"def bad(pair: (tensor[4, f32], int32)): int32 =
  let n: int32 = match pair with {
    | (x, _) => 1
  }
  let again: (tensor[4, f32], int32) = pair
  in n
"#,
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().unwrap();
    assert!(errors.iter().any(|error| {
        error["kind"].as_str() == Some("UseAfterConsume")
            && error["message"]
                .as_str()
                .is_some_and(|message| message.contains("pair"))
    }));
    assert!(!errors.iter().any(|error| {
        error["message"].as_str().is_some_and(|message| {
            message.contains("`match` is not supported by Phase 0e lowering")
        })
    }));
    assert!(json["score"].as_f64().unwrap() < 1.0);
}

#[test]
fn deep_expands_macros_and_emits_provenance() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("macro_deep.ch");
    write_file(
        &path,
        r#"
macro relu_ref(x) = max_elem(x, 0.0)
def f(x: tensor[4, f32]): tensor[4, f32] = relu_ref(x)
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let text = String::from_utf8(output).expect("utf8");
    assert!(!text.contains("defmacro"));
    assert!(!text.contains("macro-invoke"));
    assert!(text.contains("source: (relu_ref"));
    assert!(text.contains("max_elem"));
}

#[test]
fn fmt_preserves_macro_syntax() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("macro_fmt.ch");
    write_file(
        &path,
        r#"
macro keep(x)=x
def f(x: tensor[4, f32]): tensor[4, f32] = keep(x)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap(), "--inplace"])
        .assert()
        .success();

    let formatted = fs::read_to_string(&path).expect("formatted");
    assert!(formatted.contains("macro keep(x) = x"));
    let json = run_json_check(&path);
    assert_eq!(json["errors"].as_array().unwrap().len(), 0);
}

#[test]
fn validate_desugar_accepts_macro_program() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("macro_validate.ch");
    write_file(
        &path,
        r#"
macro relu_ref(x) = max_elem(x, 0.0)
def f(x: tensor[4, f32]): tensor[4, f32] = relu_ref(x)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--desugar", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated desugar:"));
}

#[test]
fn surf_rejects_internal_macro_tags_in_deep_input() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("internal_macro.dp");
    write_file(
        &path,
        "(defmacro {} relu_ref (params {} x) (app {} (var {} max_elem) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown tag 'defmacro'"));
}

#[test]
fn fmt_rejects_internal_macro_tags_in_deep_input() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("internal_macro.dp");
    write_file(
        &path,
        "(defmacro {} relu_ref (params {} x) (app {} (var {} max_elem) (var {} x) (lit {type: (t-prim {} f32)} 0.0)))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown tag 'defmacro'"));
}

#[test]
fn build_rejects_gpu_device_region_for_c_target() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("device.ch");
    write_file(&path, "let x: int32 = with device(\"gpu:0\") { 1 }\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["build", path.to_str().unwrap(), "--target", "c"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot satisfy resource region"));
}

#[test]
fn tide_quit_exits_cleanly() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .arg("tide")
        .write_stdin(":quit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("Chelis Tide v0.1"));
}

#[test]
fn tide_serve_help_is_available() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["tide", "serve", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--port"))
        .stdout(predicate::str::contains("--host"));
}

#[test]
fn tide_mcp_help_is_available() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["tide", "mcp", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Start the Tide MCP server"));
}

#[test]
fn tide_lsp_help_is_available() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["tide", "lsp", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Start the Tide LSP server"));
}

#[test]
fn tide_lsp_stdio_flag_is_accepted_and_exits_on_eof() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["tide", "lsp", "--stdio"])
        .assert()
        .success();
}

#[test]
fn cove_help_is_available() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["cove", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("--file"))
        .stdout(predicate::str::contains("Launch the Cove terminal UI"));
}

#[test]
fn cove_reports_missing_file_before_terminal_error() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["cove", "--file", "/definitely/missing/chelis-file.ch"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("read Cove file"));
}

#[test]
fn cove_requires_interactive_terminal_for_valid_launch() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["cove", "--file", hello_tensor_example().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("interactive terminal"));
}

#[test]
fn vscode_extension_manifest_registers_languages_and_command() {
    let manifest = fs::read_to_string(editor_file("package.json")).expect("manifest");
    let json: Value = serde_json::from_str(&manifest).expect("valid json");
    let languages = json["contributes"]["languages"]
        .as_array()
        .expect("languages array");
    assert!(languages.iter().any(|entry| entry["id"] == "chelis"));
    assert!(languages.iter().any(|entry| entry["id"] == "chelis-deep"));

    let commands = json["contributes"]["commands"]
        .as_array()
        .expect("commands array");
    assert!(
        commands
            .iter()
            .any(|entry| entry["command"] == "chelis.showDeep")
    );
}

#[test]
fn vscode_grammars_exist_and_parse_as_json() {
    for path in [
        editor_file("syntaxes/chelis.tmLanguage.json"),
        editor_file("syntaxes/chelis-deep.tmLanguage.json"),
    ] {
        let contents = fs::read_to_string(&path).expect("grammar");
        let json: Value = serde_json::from_str(&contents).expect("valid grammar json");
        assert!(json["scopeName"].is_string(), "{path:?}");
        assert!(json["repository"].is_object(), "{path:?}");
    }
}

#[test]
fn validate_requires_exactly_one_mode() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", hello_tensor_example().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("--surf"))
        .stderr(predicate::str::contains("--deep"))
        .stderr(predicate::str::contains("--desugar"));
}

#[test]
fn validate_surf_accepts_executable_example() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "validate",
            "--surf",
            hello_tensor_example().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated surf:"));
}

#[test]
fn validate_desugar_accepts_executable_example() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "validate",
            "--desugar",
            hello_tensor_example().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated desugar:"));
}

#[test]
fn validate_deep_accepts_canonical_deep_output() {
    let dir = tempdir().expect("tempdir");
    let deep_path = dir.path().join("hello_tensor.dp");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", hello_tensor_example().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&deep_path, output).expect("write deep output");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", deep_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated deep:"));
}

#[test]
fn validate_deep_accepts_flat_canonical_output() {
    let dir = tempdir().expect("tempdir");
    let deep_path = dir.path().join("hello_tensor_flat.dp");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", "--flat", hello_tensor_example().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&deep_path, output).expect("write deep output");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", deep_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated deep:"));
}

#[test]
fn deep_defaults_to_pretty_output_and_flat_flag_preserves_flat_per_form_rendering() {
    let pretty = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", hello_tensor_example().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let pretty = String::from_utf8(pretty).expect("utf8");
    assert!(
        pretty.contains("\n  ("),
        "expected indented pretty Deep: {pretty}"
    );

    let flat = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", "--flat", hello_tensor_example().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let flat = String::from_utf8(flat).expect("utf8");
    assert!(
        !flat.contains("\n  ("),
        "flat output should not indent child lines: {flat}"
    );
}

#[test]
fn deep_flat_keeps_top_level_forms_separated() {
    let dir = tempdir().expect("tempdir");
    let surf_path = dir.path().join("multi.ch");
    write_file(&surf_path, "def a = 1\ndef b = 2\n");

    let flat = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", "--flat", surf_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let flat = String::from_utf8(flat).expect("utf8");
    assert!(
        flat.contains("\n\n"),
        "expected top-level form separation: {flat}"
    );
    assert!(
        !flat.contains("\n  ("),
        "flat output should not indent child lines: {flat}"
    );
}

#[test]
fn validate_deep_accepts_dotted_module_deep_output() {
    let dir = tempdir().expect("tempdir");
    let surf_path = dir.path().join("module_paths.ch");
    let deep_path = dir.path().join("module_paths.dp");
    write_file(
        &surf_path,
        "module Foo.Bar\nimport Baz.Qux(..)\ndef f(x) = x\n",
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", surf_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&deep_path, output).expect("write deep output");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", deep_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated deep:"));
}

#[test]
fn validate_surf_rejects_malformed_operator_chain() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad.ch");
    write_file(&path, "def f(x) = a == b == c\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--surf", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("validation failed"));
}

#[test]
fn validate_surf_accepts_semicolon_block_and_axis_identifier() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("ok.ch");
    write_file(
        &path,
        "def f(axis) = { let y = axis; y }\ndef g() = par { a; b }\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--surf", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated surf:"));
}

#[test]
fn validate_desugar_accepts_dotted_module_paths() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("module_paths.ch");
    write_file(&path, "module Foo.Bar\nimport Baz.Qux(..)\ndef f(x) = x\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--desugar", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated desugar:"));
}

#[test]
fn validate_deep_rejects_unknown_tag() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad.dp");
    write_file(&path, "(mystery {} x)\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown Deep tag"));
}
