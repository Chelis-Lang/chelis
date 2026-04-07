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
fn tide_quit_exits_cleanly() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .arg("tide")
        .write_stdin(":quit\n")
        .assert()
        .success()
        .stdout(predicate::str::contains("Chelis Tide v0.1"));
}
