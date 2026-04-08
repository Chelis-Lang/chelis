use assert_cmd::Command;
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

fn illustrative_example(name: &str) -> PathBuf {
    example_path(&format!("../../examples/illustrative/{name}"))
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
    for path in [mnist_example(), hello_tensor_example()] {
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
fn eval_rejects_missing_inputs() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", mnist_example().to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("missing required input"));
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
fn check_does_not_report_perfect_score_with_errors() {
    let json = run_json_check(&illustrative_example("pattern_matching.ch"));
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
