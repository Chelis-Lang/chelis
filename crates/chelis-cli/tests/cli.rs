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
fn build_hip_rejects_symbolic_dims_without_panic() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic.ch");
    write_file(
        &path,
        "def f(xs: tensor[batch, features, f32]): tensor[batch, features, f32] = xs\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["build", path.to_str().unwrap(), "--target", "hip"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "does not yet support unresolved named dimensions",
        ))
        .stderr(predicate::str::contains("symbolic dimension `batch`"));
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
