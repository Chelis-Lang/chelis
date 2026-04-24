use assert_cmd::Command;
use chelis_shell::{ShellSymbol, SymbolKind, read_shell, write_shell};
use predicates::prelude::*;
use serde_json::Value;
use std::env;
use std::fs;
#[cfg(unix)]
use std::os::unix::process::ExitStatusExt;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
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

fn list_foundation_example() -> PathBuf {
    example_path("../../examples/list_foundation.ch")
}

fn dict_foundation_example() -> PathBuf {
    example_path("../../examples/dict_foundation.ch")
}

fn iter_foundation_example() -> PathBuf {
    example_path("../../examples/iter_foundation.ch")
}

fn tensor_structural_ops_example() -> PathBuf {
    example_path("../../examples/tensor_structural_ops.ch")
}

fn linreg_example() -> PathBuf {
    example_path("../../examples/linreg.ch")
}

fn transformer_block_example() -> PathBuf {
    example_path("../../examples/transformer_block.ch")
}

fn executable_examples() -> [PathBuf; 10] {
    [
        dict_foundation_example(),
        hello_tensor_example(),
        iter_foundation_example(),
        list_foundation_example(),
        linreg_example(),
        mnist_example(),
        scalar_string_foundation_example(),
        tensor_structural_ops_example(),
        transformer_block_example(),
        vmap_example(),
    ]
}

fn illustrative_examples() -> Vec<PathBuf> {
    let root = example_path("../../examples/illustrative");
    let mut paths = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let entry = entry.expect("walk illustrative examples");
        let path = entry.path();
        if entry.file_type().is_file()
            && path.extension().and_then(|ext| ext.to_str()) == Some("ch")
        {
            paths.push(path.to_path_buf());
        }
    }
    paths.sort();
    paths
}

fn stdlib_sources() -> Vec<PathBuf> {
    let root = example_path("../../packages/chelis-std/src");
    let mut paths = Vec::new();
    for entry in walkdir::WalkDir::new(root) {
        let entry = entry.expect("walk stdlib sources");
        let path = entry.path();
        if entry.file_type().is_file()
            && path.extension().and_then(|ext| ext.to_str()) == Some("ch")
        {
            paths.push(path.to_path_buf());
        }
    }
    paths.sort();
    paths
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
        r#"a = (a : tensor[2, 3, f32])
b = (b : tensor[3, 4, f32])
out = (matmul(a, b) : tensor[2, 4, f32])
"#,
    )
    .expect("write matmul program");
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn generated_source_needs_blas(out_dir: &Path, sources: &[&str]) -> bool {
    sources.iter().any(|source| {
        fs::read_to_string(out_dir.join(source))
            .map(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
            .unwrap_or(false)
    })
}

fn cpu_toolchain_for_sources(
    out_dir: &Path,
    sources: &[&str],
) -> chelis_backend_c::toolchain::NativeToolchain {
    chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: generated_source_needs_blas(out_dir, sources),
        },
    )
}

fn gcc_link_generated(out_dir: &Path, source: &str, binary: &str) -> std::process::ExitStatus {
    let toolchain = cpu_toolchain_for_sources(out_dir, &[source]);
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.arg(source);
    cmd.args(["-L.", "-lchelis_runtime"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", binary]);
    cmd.status().expect("gcc should run")
}

fn gcc_link_sources(out_dir: &Path, sources: &[&str], binary: &str) -> std::process::ExitStatus {
    let toolchain = cpu_toolchain_for_sources(out_dir, sources);
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.args(sources);
    cmd.args(["-L.", "-lchelis_runtime"]);
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", binary]);
    cmd.status().expect("gcc should run")
}

fn gcc_compile_generated(out_dir: &Path, source: &str) -> std::process::ExitStatus {
    let toolchain = cpu_toolchain_for_sources(out_dir, &[source]);
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.args(["-I.", "-c", source]);
    cmd.status().expect("gcc should run")
}

fn hipcc_link_generated(out_dir: &Path, source: &str, binary: &str) -> std::process::ExitStatus {
    StdCommand::new("hipcc")
        .current_dir(out_dir)
        .args([
            source,
            "-L.",
            "-lchelis_runtime",
            "-lm",
            "-lpthread",
            "-ldl",
            "-o",
            binary,
        ])
        .status()
        .expect("hipcc should run")
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

fn runtime_library_path() -> PathBuf {
    let manifest_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    for dir in [
        manifest_dir.join("../../target/debug/deps"),
        manifest_dir.join("../../target/release/deps"),
    ] {
        if let Ok(entries) = fs::read_dir(&dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .map(|name| name.starts_with("libchelis_runtime") && name.ends_with(".a"))
                    .unwrap_or(false)
                {
                    return path;
                }
            }
        }
    }
    panic!("could not locate libchelis_runtime.a for cli tests");
}

fn runtime_library_dir() -> PathBuf {
    runtime_library_path()
        .parent()
        .expect("runtime library should have a parent directory")
        .to_path_buf()
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
fn fmt_check_accepts_canonical_illustrative_examples() {
    for path in illustrative_examples() {
        Command::cargo_bin("chelis")
            .expect("binary")
            .args(["fmt", path.to_str().unwrap(), "--check"])
            .assert()
            .success();
    }
}

#[test]
fn fmt_check_accepts_canonical_stdlib_sources() {
    for path in stdlib_sources() {
        Command::cargo_bin("chelis")
            .expect("binary")
            .args(["fmt", path.to_str().unwrap(), "--check"])
            .assert()
            .success();
    }
}

#[test]
fn eval_rejects_unbound_runtime_names() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("unknown_name.ch");
    write_file(&path, "result = add(input, 1.0)\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unbound variable: input"));
}

#[test]
fn eval_prints_labeled_tuple_components() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("tuple_eval.ch");
    write_file(
        &path,
        r#"grads = (1.0, 2.5)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "grads.0 = tensor(shape=[], data=[1.0])",
        ))
        .stdout(predicate::str::contains(
            "grads.1 = tensor(shape=[], data=[2.5])",
        ));
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
fn check_collection_callback_errors_name_the_helper_contract() {
    let dir = tempdir().expect("tempdir");
    let cases = [
        (
            "filter_non_bool.ch",
            r#"
xs: List[int64] = [cast(1, int64)]
bad = filter(fn (x: int64) -> add(x, cast(1, int64)), xs)
"#,
            vec!["filter", "callback", "bool"],
        ),
        (
            "partition_non_bool.ch",
            r#"
xs: List[int64] = [cast(1, int64)]
bad = partition(fn (x: int64) -> add(x, cast(1, int64)), xs)
"#,
            vec!["partition", "callback", "bool"],
        ),
        (
            "fold_acc_mismatch.ch",
            r#"
xs: List[int64] = [cast(1, int64)]
bad = fold(fn (acc: string, x: int64) -> string_concat(acc, to_string(x)), cast(0, int64), xs)
"#,
            vec!["fold", "accumulator", "string", "int64"],
        ),
        (
            "scan_acc_mismatch.ch",
            r#"
xs: List[int64] = [cast(1, int64)]
bad = scan(fn (acc: string, x: int64) -> string_concat(acc, to_string(x)), cast(0, int64), xs)
"#,
            vec!["scan", "accumulator", "string", "int64"],
        ),
    ];

    for (name, source, required_fragments) in cases {
        let path = dir.path().join(name);
        write_file(&path, source);
        let json = run_json_check(&path);
        let errors = json["errors"]
            .as_array()
            .expect("errors should be an array");
        assert!(!errors.is_empty(), "{name} should fail");
        let message = errors[0]["message"]
            .as_str()
            .expect("error should contain a message");
        for fragment in required_fragments {
            assert!(
                message.contains(fragment),
                "{name} message should contain `{fragment}`, got `{message}`"
            );
        }
    }
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
        .stdout(predicate::str::contains("progress: ckpt-7.safetensors"));
}

#[test]
fn eval_supports_lists_and_tensor_bridge() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            "--file",
            list_foundation_example().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("len=4, items=4, shape=2x2"))
        .stdout(predicate::str::contains("\n[1, 2, 3]\n"))
        .stdout(predicate::str::contains("\n[2, 3, 4]\n"))
        .stdout(predicate::str::contains("\n[[1, 2], [3]]\n"))
        .stdout(predicate::str::contains("[1, 2, 3, 4]\n"));
}

#[test]
fn eval_supports_dicts_and_iteration_collections() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            "--file",
            dict_foundation_example().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "[(alpha, 1), (beta, 2), (gamma, 3)]",
        ))
        .stdout(predicate::str::contains(
            "[(0, alpha), (1, beta), (2, gamma)]",
        ))
        .stdout(predicate::str::contains("[alpha, beta, gamma]"))
        .stdout(predicate::str::contains("[1, 2, 3]"))
        .stdout(predicate::str::contains(
            "[(alpha, 1), (beta, 20), (gamma, 3), (delta, 4), (epsilon, 5)]",
        ))
        .stdout(predicate::str::contains(
            "[(alpha, 1), (beta, 20), (delta, 4), (epsilon, 5)]",
        ));
}

#[test]
fn eval_supports_map_filter_fold_collections() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            "--file",
            iter_foundation_example().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("[2, 3, 4, 5]"))
        .stdout(predicate::str::contains("[2, 4]"))
        .stdout(predicate::str::contains("[1, 3, 6, 10]"))
        .stdout(predicate::str::contains("([3, 4], [1, 2])"))
        .stdout(predicate::str::contains("[1, 11, 2, 12]"))
        .stdout(predicate::str::contains("[[1, 11], [2, 12]]"))
        .stdout(predicate::str::contains("total=6"));
}

#[test]
fn eval_supports_phase3h_tensor_structural_ops() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "eval",
            "--file",
            tensor_structural_ops_example().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "contracted = tensor(shape=[2, 2], data=[19.0, 22.0, 43.0, 50.0])",
        ))
        .stdout(predicate::str::contains(
            "sorted_indices = tensor(shape=[2], data=[0.0, 1.0])",
        ));
}

#[test]
fn build_c_runs_list_foundation_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("list-build-out");
    let source = list_foundation_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = gcc_link_generated(&out_dir, "list_foundation.c", "list_foundation");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("list_foundation"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
fn build_c_runs_dict_foundation_and_matches_eval_output() {
    let temp = tempdir().expect("tempdir");
    let out_dir = temp.path().join("build");
    let source = dict_foundation_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = gcc_link_generated(&out_dir, "dict_foundation.c", "dict_foundation");
    assert!(status.success(), "generated C should compile");

    let run_output = StdCommand::new(out_dir.join("dict_foundation"))
        .current_dir(&out_dir)
        .output()
        .expect("run compiled program");
    assert!(run_output.status.success(), "compiled program should run");
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
fn build_c_runs_iter_foundation_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("iter-build-out");
    let source = iter_foundation_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = gcc_link_generated(&out_dir, "iter_foundation.c", "iter_foundation");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("iter_foundation"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
fn build_c_runs_tensor_structural_ops_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("tensor-structural-build-out");
    let source = tensor_structural_ops_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = gcc_link_generated(&out_dir, "tensor_structural_ops.c", "tensor_structural_ops");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("tensor_structural_ops"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
fn build_c_runs_top_level_tensor_add_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("top_level_tensor_add.ch");
    let out_dir = dir.path().join("top-level-tensor-add-build-out");
    write_file(
        &path,
        "result = add((to_tensor([1.0, 2.0, 3.0, 4.0]) : tensor[4, f32]), \
         (to_tensor([10.0, 20.0, 30.0, 40.0]) : tensor[4, f32]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = gcc_link_generated(&out_dir, "top_level_tensor_add.c", "top_level_tensor_add");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("top_level_tensor_add"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    let expected_value = String::from_utf8(eval_stdout).expect("eval stdout utf8");
    let actual = String::from_utf8(run_output.stdout).expect("run stdout utf8");
    assert_eq!(actual, format!("result = {expected_value}"));
}

#[test]
fn build_c_host_tensor_helper_dedups_repeated_inputs_at_callsite() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("gram_bug5.ch");
    let out_dir = dir.path().join("gram-bug5-build-out");
    write_file(
        &path,
        "def gram(a: tensor[2, 3, f32]) -> tensor[3, 3, f32] = matmul(permute(copy(a), 1, 0), copy(a))\n\
         def apply(a: tensor[2, 3, f32]) -> tensor[3, 3, f32] = gram(a)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let generated = fs::read_to_string(out_dir.join("gram_bug5.c")).expect("generated source");
    assert!(
        generated.contains("expected %d inputs, got %d\\n\", 1, n_in"),
        "expected helper ABI to dedup repeated tensor loads, got:\n{generated}"
    );

    write_file(
        &out_dir.join("driver.c"),
        r#"#include "chelis_runtime.h"
#include "gram_bug5.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    int shape[2] = {2, 3};
    chelis_tensor *a = chelis_alloc(2, shape, CHELIS_F32);
    float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    for (int i = 0; i < 6; ++i) {
        a->data[i] = values[i];
    }

    chelis_tensor *out = gram(a);
    float expected[9] = {
        17.0f, 22.0f, 27.0f,
        22.0f, 29.0f, 36.0f,
        27.0f, 36.0f, 45.0f
    };
    for (int i = 0; i < 9; ++i) {
        if (fabsf(out->data[i] - expected[i]) > 1e-4f) {
            fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, out->data[i], expected[i]);
            return 1;
        }
    }

    chelis_free(out);
    chelis_free(a);
    return 0;
}
"#,
    );

    let status = gcc_link_sources(&out_dir, &["driver.c", "gram_bug5.c"], "gram_bug5_driver");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("gram_bug5_driver"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {} stderr:\n{}",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr)
    );
}

#[test]
fn build_c_user_defined_helpers_are_static_inline_when_main_is_emitted() {
    // Regression guard for Nautilus benchmark ask: when `chelis build` produces a
    // self-contained binary (top-level `result = ...` triggers main emission), any
    // user-defined `def` in the same TU should be marked `static inline` so -O2
    // cross-call inlining kicks in without LTO / -Wl,-Bsymbolic on the downstream
    // shell. Object-mode builds (no main) keep external linkage.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("inline_helpers.ch");
    let out_dir = dir.path().join("inline-helpers-build-out");
    write_file(
        &path,
        "def combine(a: tensor[4, f32], b: tensor[4, f32]) -> tensor[4, f32] = add(a, b)\n\
         result = combine((to_tensor([1.0, 2.0, 3.0, 4.0]) : tensor[4, f32]), \
         (to_tensor([10.0, 20.0, 30.0, 40.0]) : tensor[4, f32]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let c_src = fs::read_to_string(out_dir.join("inline_helpers.c")).expect("read generated C");
    assert!(
        c_src.contains("static inline chelis_tensor* combine("),
        "expected combine() to be emitted as `static inline` in self-contained binary, got:\n{c_src}"
    );
    assert!(
        c_src.contains("int main("),
        "expected generated C to emit main(), got:\n{c_src}"
    );

    let status = gcc_link_generated(&out_dir, "inline_helpers.c", "inline_helpers");
    assert!(status.success(), "gcc failed with status {status}");
    let run_output = StdCommand::new(out_dir.join("inline_helpers"))
        .output()
        .expect("compiled binary should run");
    assert!(run_output.status.success());
    let actual = String::from_utf8(run_output.stdout).expect("stdout utf8");
    assert!(
        actual.contains("11.0")
            && actual.contains("22.0")
            && actual.contains("33.0")
            && actual.contains("44.0"),
        "combine should elementwise-add both operands; got: {actual}"
    );
}

#[test]
fn build_c_tuple_return_header_supports_driver_extraction() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("tuple_abi.ch");
    let out_dir = dir.path().join("tuple-abi-build-out");
    write_file(
        &path,
        "def eig_pair() -> (tensor[2, f32], tensor[2, f32]) = (to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let header = fs::read_to_string(out_dir.join("tuple_abi.h")).expect("generated header");
    assert!(
        header.contains("chelis_tuple* eig_pair("),
        "expected tuple-returning C ABI in generated header, got:\n{header}"
    );

    write_file(
        &out_dir.join("driver.c"),
        r#"#include "chelis_runtime.h"
#include "tuple_abi.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    chelis_tuple *out = eig_pair();
    chelis_tensor *lhs = chelis_tuple_get_tensor(out, 0);
    chelis_tensor *rhs = chelis_tuple_get_tensor(out, 1);

    if (lhs->size != 2 || rhs->size != 2) {
        fprintf(stderr, "unexpected tuple tensor sizes\n");
        return 1;
    }
    if (fabsf(lhs->data[0] - 1.0f) > 1e-4f || fabsf(lhs->data[1] - 2.0f) > 1e-4f) {
        fprintf(stderr, "lhs mismatch\n");
        return 1;
    }
    if (fabsf(rhs->data[0] - 3.0f) > 1e-4f || fabsf(rhs->data[1] - 4.0f) > 1e-4f) {
        fprintf(stderr, "rhs mismatch\n");
        return 1;
    }

    chelis_free(lhs);
    chelis_free(rhs);
    chelis_tuple_release(out);
    return 0;
}
"#,
    );

    let status = gcc_link_sources(&out_dir, &["driver.c", "tuple_abi.c"], "tuple_abi_driver");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("tuple_abi_driver"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {} stderr:\n{}",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr)
    );
}

#[test]
fn build_c_multidef_tensor_entry_renames_source_main_for_driver_compatibility() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bug3c.ch");
    let out_dir = dir.path().join("bug3c-build-out");
    write_file(
        &path,
        "def combine(a: tensor[4, f32], b: tensor[4, f32]) -> tensor[4, f32] = add(a, b)\n\
         def main(x: tensor[4, f32], y: tensor[4, f32]) -> tensor[4, f32] = combine(x, y)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let header = fs::read_to_string(out_dir.join("bug3c.h")).expect("generated header");
    let source = fs::read_to_string(out_dir.join("bug3c.c")).expect("generated source");
    assert!(
        header.contains("chelis_tensor* bug3c__main("),
        "expected source-level main to be renamed in generated header, got:\n{header}"
    );
    assert!(
        !header.contains("chelis_tensor* main("),
        "generated header must not export raw C `main`, got:\n{header}"
    );
    assert!(
        source.contains("chelis_tensor* bug3c__main("),
        "expected generated source to rename source-level main, got:\n{source}"
    );
    assert!(
        source.contains("chelis_tensor* combine("),
        "expected helper symbol to remain callable, got:\n{source}"
    );

    write_file(
        &out_dir.join("driver.c"),
        "#include \"chelis_runtime.h\"\n\
         #include \"bug3c.h\"\n\
         int main(void) { return 0; }\n",
    );
    let status = gcc_link_sources(&out_dir, &["driver.c", "bug3c.c"], "bug3c_driver");
    assert!(
        status.success(),
        "downstream C driver with its own main() should link against generated output; status {status}"
    );
}

#[test]
fn check_rejects_literal_dimension_mismatch() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("literal_dim_mismatch.ch");
    write_file(
        &path,
        "def want_2x2(a: tensor[2, 2, f32]) -> f32 = trace(a, 0, 1)\n\
         def main(a: tensor[3, 3, f32]) -> f32 = want_2x2(a)\n",
    );

    let json = run_json_check(&path);
    assert!(
        json["errors"]
            .as_array()
            .expect("errors array")
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch"),
        "expected DimensionMismatch in check output, got {json}"
    );
    assert_eq!(
        json["unresolved_names"]
            .as_array()
            .expect("unresolved_names array")
            .len(),
        0,
        "{json}"
    );
}

#[test]
fn check_rejects_polymorphic_dims_pinned_by_body() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("polymorphic_dim_pinned.ch");
    write_file(
        &path,
        "def want_2x2(a: tensor[2, 2, f32]) -> f32 = trace(a, 0, 1)\n\
         def bad_consumer[m, n](a: tensor[m, n, f32]) -> f32 = want_2x2(a)\n\
         def main() -> f32 = cast(0.0, f32)\n",
    );

    let json = run_json_check(&path);
    assert!(
        json["errors"]
            .as_array()
            .expect("errors array")
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch"),
        "expected DimensionMismatch when polymorphic dims are pinned by the body, got {json}"
    );
    assert_eq!(
        json["unresolved_names"]
            .as_array()
            .expect("unresolved_names array")
            .len(),
        0,
        "{json}"
    );
}

#[test]
fn check_rejects_unresolved_function_names() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("unresolved_name.ch");
    write_file(
        &path,
        "def probe(x: f32) -> f32 = sub(x, frobnicate(x))\n\
         def main() -> f32 = probe(cast(1.0, f32))\n",
    );

    let json = run_json_check(&path);
    assert!(
        json["errors"]
            .as_array()
            .expect("errors array")
            .iter()
            .any(|error| error["kind"] == "UnboundVariable"),
        "expected UnboundVariable in check output, got {json}"
    );
    assert!(
        json["unresolved_names"]
            .as_array()
            .expect("unresolved_names array")
            .iter()
            .any(|name| name == "frobnicate"),
        "expected unresolved_names to include frobnicate, got {json}"
    );
}

#[test]
fn build_c_nested_float_builtins_do_not_emit_int_temps() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested_float_builtins.ch");
    let out_dir = dir.path().join("nested-float-build-out");
    write_file(
        &path,
        "def bad(x: f32) -> f32 = exp(neg(mul(x, x)))\n\
         def main() -> f32 = bad(cast(1.5, f32))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("nested_float_builtins.c")).expect("generated c");
    assert!(
        !source.contains("int __arg"),
        "expected float-valued builtin temps to avoid int declarations:\n{source}"
    );
    assert!(
        source.contains("double __arg"),
        "expected generated host C to use double temps:\n{source}"
    );

    let status = gcc_compile_generated(&out_dir, "nested_float_builtins.c");
    assert!(status.success(), "gcc failed with status {status}");
}

#[test]
fn build_c_fold_tuple_tensor_accumulator_specializes_callback_types() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("fold_tuple_tensor_acc.ch");
    let out_dir = dir.path().join("fold-tuple-build-out");
    write_file(
        &path,
        "xs = to_list(to_tensor([1.0, 2.0]))\n\
         state0 = (to_tensor([0.0, 0.0]), cast(0.0, f32))\n\
         step = fn (state, x) -> {\n\
           l_inner = state.0\n\
           total = state.1\n\
           (l_inner, add(total, x))\n\
         }\n\
         out = fold(step, state0, xs)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("fold_tuple_tensor_acc.c")).expect("generated c");
    assert!(
        source.contains("chelis_tensor* l_inner;"),
        "expected tuple-get binding to lower as a tensor local:\n{source}"
    );
    assert!(
        !source.contains("int l_inner;"),
        "tuple tensor binding must not degrade to int:\n{source}"
    );

    let status = gcc_compile_generated(&out_dir, "fold_tuple_tensor_acc.c");
    assert!(status.success(), "gcc failed with status {status}");
}

#[test]
fn build_c_map_tensor_grad_specializes_callback_item_type() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_rows_map.ch");
    let out_dir = dir.path().join("grad-rows-map-build-out");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32], x: f32) -> f32 =\n\
           add(tensor_to_scalar(sum(mul(copy(theta), copy(theta)), 0)), x)\n\
         grad_loss = grad(loss, wrt=(theta))\n\
         xs: List[f32] = [1.0, 2.0]\n\
         rows = map(fn (x) -> grad_loss(to_tensor([1.0, 2.0]), x), xs)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("grad_rows_map.c")).expect("generated c");
    assert!(
        source.contains("double __map_item_") && source.contains("double x = __map_item_"),
        "expected map callback item to lower as double rather than int:\n{source}"
    );
    assert!(
        !source.contains("int __map_item_") && !source.contains("int x;"),
        "map callback item must not degrade to int:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "grad_rows_map.c", "grad_rows_map");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("grad_rows_map"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    assert!(
        stdout.contains(
            "rows = [tensor(shape=[2], data=[2.0, 4.0]), tensor(shape=[2], data=[2.0, 4.0])]"
        ),
        "unexpected gradient rows output:\n{stdout}"
    );
}

#[test]
fn build_c_tensor_grad_with_host_branching_dependency_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_rows_branching.ch");
    let out_dir = dir.path().join("grad-rows-branching-build-out");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32], x: f32) -> f32 =\n\
           if x < 0.0 then tensor_to_scalar(sum(mul(copy(theta), copy(theta)), 0)) else add(tensor_to_scalar(sum(mul(copy(theta), copy(theta)), 0)), x)\n\
         grad_loss = grad(loss, wrt=(theta))\n\
         xs: List[f32] = [1.0, -2.0]\n\
         rows = map(fn (x) -> grad_loss(to_tensor([1.0, 2.0]), x), xs)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("grad_rows_branching.c")).expect("generated c");
    assert!(
        source.contains("static inline double loss("),
        "expected host-side scalar loss helper to be emitted:\n{source}"
    );
    assert!(
        source.contains("if (__cond"),
        "expected branching loss to stay on the host path rather than panic in DAG lowering:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "grad_rows_branching.c", "grad_rows_branching");
    assert!(status.success(), "gcc failed with status {status}");
}

#[test]
fn build_c_tensor_grad_lm_style_mixed_scalar_tensor_args_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("tensor_grad_lm_canary.ch");
    let out_dir = dir.path().join("tensor-grad-lm-canary-out");
    write_file(
        &path,
        "def residual[n](theta: tensor[n, f32], x: f32, y: f32) -> f32 = {\n\
           y_hat = if lt(x, cast(0.0, f32)) then tensor_to_scalar(sum(copy(theta), 0)) else add(tensor_to_scalar(sum(copy(theta), 0)), x)\n\
           sub(y, y_hat)\n\
         }\n\
         row = grad(residual, wrt=(theta))\n\
         def jac[n, m](theta: tensor[n, f32], xs: tensor[m, f32], ys: tensor[m, f32]) -> List[tensor[n, f32]] = {\n\
           pairs = zip(to_list(xs), to_list(ys))\n\
           map(fn (pair: (f32, f32)) -> row(copy(theta), pair.0, pair.1), pairs)\n\
         }\n\
         out = jac(to_tensor([1.0, 2.0]), to_tensor([1.0, -2.0]), to_tensor([3.0, 4.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("tensor_grad_lm_canary.c")).expect("generated c");
    assert!(
        source.contains("static inline chelis_tensor* row("),
        "expected a host wrapper for the gradient row helper:\n{source}"
    );
    assert!(
        source.contains("__tensor_arg1_")
            && source.contains("chelis_alloc(1, (int[]){1}, CHELIS_F32)"),
        "expected host scalar dependencies to be boxed as rank-1 tensor helper inputs:\n{source}"
    );
    assert!(
        !source.contains("= lt;"),
        "gradient codegen must not leave unresolved host builtins in C:\n{source}"
    );

    let status = gcc_compile_generated(&out_dir, "tensor_grad_lm_canary.c");
    assert!(status.success(), "gcc failed with status {status}");
}

#[test]
fn build_c_tensor_grad_local_wrapper_over_function_param_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("tensor_grad_local_wrapper.ch");
    let out_dir = dir.path().join("tensor-grad-local-wrapper-build-out");
    write_file(
        &path,
        "def jac_row[n](model: tensor[n, f32] -> f32 -> f32 -> f32, theta: tensor[n, f32], x: f32, y: f32) -> tensor[n, f32] = {\n\
           target = fn (theta_local: tensor[n, f32]) -> model(theta_local, x, y)\n\
           grad(target, wrt=(theta_local))(theta)\n\
         }\n\
         def lm_model(theta: tensor[2, f32], x: f32, y: f32) -> f32 = {\n\
           y_hat = if lt(x, cast(0.0, f32)) then tensor_to_scalar(sum(copy(theta), 0)) else add(tensor_to_scalar(sum(copy(theta), 0)), x)\n\
           sub(y, y_hat)\n\
         }\n\
         out = jac_row(lm_model, to_tensor([1.0, 2.0]), cast(1.0, f32), cast(3.0, f32))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source =
        fs::read_to_string(out_dir.join("tensor_grad_local_wrapper.c")).expect("generated c");
    assert!(
        source.contains("chelis_tensor* __binding_0_value;"),
        "expected tensor-valued local grad result to stay tensor-typed:\n{source}"
    );
    assert!(
        source.contains("chelis_tensor_from_value_list(")
            && source.contains("__host_tensor_arg_1")
            && source.contains("tensor_grad_local_wrapper__global__tensor_0"),
        "expected local-wrapper grad to specialize into a tensor helper with a hoisted tensor arg:\n{source}"
    );
    assert!(
        !source.contains("`grad` is not representable")
            && !source.contains("unsupported builtin")
            && !source.contains("int __binding_0_value;")
            && !source.contains("= to_tensor;")
            && !source.contains("__result = call("),
        "local-wrapper grad lowering must not degrade to fallback, unresolved builtins, or int locals:\n{source}"
    );

    let status = gcc_compile_generated(&out_dir, "tensor_grad_local_wrapper.c");
    assert!(status.success(), "gcc failed with status {status}");
}

/// Regression test for the Coral UPSTREAM_BUGS.md pattern:
/// `grad(loss, wrt=theta)(theta, x)` applied to a multi-param named top-level def
/// (two tensor arguments, differentiating w.r.t. the first).
/// Verifies: build exits 0, generated C compiles with gcc, and the gradient values
/// are numerically correct — grad of sum(theta*x) w.r.t. theta equals x.
#[test]
fn build_c_grad_named_fn_multi_param_wrt_builds_and_is_numerically_correct() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_coral.ch");
    let out_dir = dir.path().join("grad-coral-build-out");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[f32] =\n\
           sum(mul(theta, x), 0)\n\
         def compute_grad(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[2, f32] =\n\
           grad(loss, wrt=theta)(theta, x)\n\
         out = compute_grad(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("grad_coral.c")).expect("generated c");
    assert!(
        !source.contains("__result = call(") && !source.contains("unsupported builtin"),
        "generated C must not contain unresolved call stubs:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "grad_coral.c", "grad_coral");
    assert!(status.success(), "gcc link failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("grad_coral"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    // grad(sum(theta*x), wrt=theta) = x = [3.0, 4.0]
    assert!(
        stdout.contains("out = tensor(shape=[2], data=[3.0, 4.0])"),
        "gradient of sum(theta*x) w.r.t. theta must equal x=[3.0, 4.0], got:\n{stdout}"
    );
}

/// Regression test for the Coral UPSTREAM_BUGS.md pattern:
/// `grad(loss, wrt=(x))(theta, x)` differentiates w.r.t. the second argument.
/// Verifies: build exits 0, generated C compiles, and grad of sum(theta*x) w.r.t. x equals theta.
#[test]
fn build_c_grad_named_fn_wrt_second_param_is_numerically_correct() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_wrt_x.ch");
    let out_dir = dir.path().join("grad-wrt-x-build-out");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[f32] =\n\
           sum(mul(theta, x), 0)\n\
         out = grad(loss, wrt=(x))(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("grad_wrt_x.c")).expect("generated c");
    assert!(
        !source.contains("__result = call(") && !source.contains("unsupported builtin"),
        "generated C must not contain unresolved call stubs:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "grad_wrt_x.c", "grad_wrt_x");
    assert!(status.success(), "gcc link failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("grad_wrt_x"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    // grad(sum(theta*x), wrt=x) = theta = [1.0, 2.0]
    assert!(
        stdout.contains("out = tensor(shape=[2], data=[1.0, 2.0])"),
        "gradient of sum(theta*x) w.r.t. x must equal theta=[1.0, 2.0], got:\n{stdout}"
    );
}

#[test]
fn build_c_recursive_tensor_function_stays_on_host_path() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("recursive_tensor.ch");
    let out_dir = dir.path().join("recursive-tensor-build-out");
    write_file(
        &path,
        "def recur[n](x: tensor[n, f32], i: int64) -> tensor[n, f32] =\n\
           if lte(i, cast(0, int64)) then x else recur(x, sub(i, cast(1, int64)))\n\
         out = recur(to_tensor([1.0, 2.0]), cast(2, int64))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("recursive_tensor.c")).expect("generated c");
    assert!(
        source.contains("static inline chelis_tensor* recur("),
        "expected recursive tensor helper to stay in the host lane:\n{source}"
    );
    assert!(
        source.contains("__result = recur("),
        "expected recursive call to remain a C function call rather than DAG helper expansion:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "recursive_tensor.c", "recursive_tensor");
    assert!(status.success(), "gcc failed with status {status}");
}

#[test]
fn build_c_tensor_fold_callback_with_if_stays_on_host_path() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("tensor_fold_if.ch");
    let out_dir = dir.path().join("tensor-fold-if-build-out");
    write_file(
        &path,
        "def chooser(x: tensor[2, f32]) -> tensor[2, f32] =\n\
           fold(fn (acc, y) -> if y < 0.0 then acc else add(acc, x), to_tensor([0.0, 0.0]), to_list(to_tensor([1.0, -1.0])))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("tensor_fold_if.c")).expect("generated c");
    assert!(
        source.contains("chelis_tensor* chooser("),
        "expected chooser to remain a host-emitted tensor function:\n{source}"
    );
    assert!(
        source.contains("if (__cond"),
        "expected fold callback branch to stay on the host path:\n{source}"
    );

    let status = gcc_compile_generated(&out_dir, "tensor_fold_if.c");
    assert!(status.success(), "gcc failed with status {status}");
}

#[test]
fn build_c_tensor_fold_let_binding_with_if_stays_on_host_path() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("tensor_fold_let_if.ch");
    let out_dir = dir.path().join("tensor-fold-let-if-build-out");
    write_file(
        &path,
        "def fold_if_tensor(xs: tensor[2, f32]) -> tensor[2, f32] = {\n\
           state0 = to_tensor([0.0, 0.0])\n\
           step = fold(fn (acc, y) -> if y < 0.0 then acc else acc, state0, to_list(xs))\n\
           step\n\
         }\n\
         out = fold_if_tensor(to_tensor([1.0, -2.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("tensor_fold_let_if.c")).expect("generated c");
    assert!(
        source.contains("int64_t __fold_len_") && source.contains("for (int64_t __i = 0;"),
        "expected let-bound fold result to lower through a host fold loop:\n{source}"
    );
    assert!(
        !source.contains("= fold;"),
        "fold builtin must not degrade into a bare identifier in generated C:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "tensor_fold_let_if.c", "tensor_fold_let_if");
    assert!(status.success(), "gcc failed with status {status}");
}

#[test]
fn build_c_preserves_unreachable_host_defs_for_driver_linking() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("library_surface.ch");
    let out_dir = dir.path().join("library-surface-build-out");
    write_file(
        &path,
        "def square(x: f32) -> f32 = mul(x, x)\n\
         def cube(x: f32) -> f32 = mul(square(x), x)\n\
         def main() -> f32 = square(cast(2.0, f32))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let generated = out_dir.join("library_surface.c");
    let source = fs::read_to_string(&generated).expect("generated c");
    assert!(
        source.contains("double cube(double x)"),
        "expected unreachable host def to survive build pruning for downstream drivers:\n{source}"
    );

    fs::write(
        &generated,
        source.replace("double main", "double chelis_entry"),
    )
    .expect("rename generated entry point");
    write_file(
        &out_dir.join("driver.c"),
        r#"#include <stdio.h>

double cube(double x);

int main(void) {
    printf("%.1f\n", cube(3.0));
    return 0;
}
"#,
    );

    let status = gcc_link_sources(
        &out_dir,
        &["driver.c", "library_surface.c"],
        "library_surface_driver",
    );
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("library_surface_driver"))
        .output()
        .expect("compiled driver should run");
    assert!(
        run_output.status.success(),
        "compiled driver failed with status {} stderr:\n{}",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr)
    );
    assert_eq!(
        String::from_utf8(run_output.stdout)
            .expect("stdout utf8")
            .trim(),
        "27.0"
    );
}

#[test]
fn build_c_preserves_generic_unreachable_tensor_defs_without_raw_dim_symbols() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("library_tensor_surface.ch");
    let out_dir = dir.path().join("library-tensor-surface-build-out");
    write_file(
        &path,
        "def double_it[n](x: tensor[n, f32]) -> tensor[n, f32] = add(copy(x), x)\n\
         def main() -> f32 = cast(1.0, f32)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("library_tensor_surface.c")).expect("generated c");
    assert!(
        source.contains("chelis_tensor* double_it(chelis_tensor* x)"),
        "expected generic tensor def to remain callable from downstream code:\n{source}"
    );
    assert!(
        !source.contains("(int[]){ d"),
        "generic tensor helper dims must be rebound to caller-visible symbols before C emission:\n{source}"
    );

    let status = gcc_compile_generated(&out_dir, "library_tensor_surface.c");
    assert!(status.success(), "gcc failed with status {status}");
}

fn assert_reef_std_embedding_builds_to_valid_c() {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join("embedding-app");
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
name = "embedding-app"
version = "0.1.0"
compiler = "=0.2.1"
module_prefix = "Demo"

[dependencies]
chelis-std = { version = "0.1.0" }
"#,
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Embedding (forward)

export (main)

def main(
  ids: tensor[2, 3, int64],
  table: tensor[8, 4, f32]
) -> tensor[2, 3, 4, f32] =
  forward(ids, table)
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
    let toolchain = cpu_toolchain_for_sources(&out_dir, &["main.c"]);
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(&out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.args(["-c", "main.c", "-o", "main.o"]);
    let status = cmd.status().expect("gcc should run");
    assert!(
        status.success(),
        "gcc object compile failed with status {status}"
    );
}

#[test]
fn phase3h_numeric_acceptance_oracle() {
    build_c_runs_tensor_structural_ops_and_matches_eval_output();
    assert_reef_std_embedding_builds_to_valid_c();
}

#[test]
fn check_rejects_static_invalid_phase3h_einsum_extent_mismatch() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("einsum_extent_bad.ch");
    write_file(
        &path,
        r#"
a = pad_sequences([[1.0, 2.0], [3.0, 4.0]], 0.0)
b = pad_sequences([[5.0, 6.0], [7.0, 8.0], [9.0, 10.0]], 0.0)
out = einsum("ij,jk->ik", a, b)
"#,
    );

    let json = run_json_check(&path);
    assert!(json["score"].as_f64().unwrap() < 1.0);
    let errors = json["errors"].as_array().expect("errors array");
    assert!(
        !errors.is_empty(),
        "check should report deterministic einsum error"
    );
    let messages = errors
        .iter()
        .filter_map(|error| error["message"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(messages.contains("einsum"));
    assert!(messages.contains("inconsistent extents"));
}

#[test]
fn check_rejects_static_invalid_phase3h_scatter_duplicate_replace() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scatter_dup_replace_bad.ch");
    write_file(
        &path,
        r#"
base = pad_sequences([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0)
ids: List[int64] = [cast(1, int64), cast(1, int64)]
idx = to_tensor(ids)
updates = pad_sequences([[5.0, 5.0], [6.0, 6.0]], 0.0)
out = scatter(base, idx, updates, 0, "replace")
"#,
    );

    let json = run_json_check(&path);
    assert!(json["score"].as_f64().unwrap() < 1.0);
    let errors = json["errors"].as_array().expect("errors array");
    assert!(
        !errors.is_empty(),
        "check should report deterministic scatter duplicate-index error"
    );
    let messages = errors
        .iter()
        .filter_map(|error| error["message"].as_str())
        .collect::<Vec<_>>()
        .join("\n");
    assert!(messages.contains("scatter"));
    assert!(messages.contains("duplicate target index"));
}

#[test]
fn build_c_phase3h_runtime_value_errors_exit_cleanly_instead_of_aborting() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("scatter_runtime_bad.ch");
    let out_dir = dir.path().join("scatter-dup-build");
    write_file(
        &source,
        r#"
def apply(
  base: tensor[3, 2, f32],
  idx: tensor[2, int64],
  updates: tensor[2, 2, f32]
) -> tensor[3, 2, f32] =
  scatter(base, idx, updates, 0, "replace")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    write_file(
        &out_dir.join("runner.c"),
        r#"#include "chelis_runtime.h"
#include "scatter_runtime_bad.h"

int main(void) {
    int base_shape[2] = {3, 2};
    int idx_shape[1] = {2};
    int updates_shape[2] = {2, 2};

    chelis_tensor *base = chelis_alloc(2, base_shape, CHELIS_F32);
    chelis_tensor *idx = chelis_alloc(1, idx_shape, CHELIS_I32);
    chelis_tensor *updates = chelis_alloc(2, updates_shape, CHELIS_F32);
    idx->data[0] = 1.0f;
    idx->data[1] = 1.0f;
    updates->data[0] = 5.0f;
    updates->data[1] = 5.0f;
    updates->data[2] = 6.0f;
    updates->data[3] = 6.0f;

    chelis_tensor *output = apply(base, idx, updates);

    if (output != NULL) {
        chelis_free(output);
    }
    chelis_free(base);
    chelis_free(idx);
    chelis_free(updates);
    return 0;
}
"#,
    );

    let status = gcc_link_sources(
        &out_dir,
        &["runner.c", "scatter_runtime_bad.c"],
        "scatter_runtime_bad",
    );
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("scatter_runtime_bad"))
        .output()
        .expect("compiled binary should run");
    assert!(
        !run_output.status.success(),
        "compiled binary should fail on duplicate scatter replace"
    );
    assert!(
        String::from_utf8_lossy(&run_output.stderr)
            .contains("scatter replace mode rejects duplicate target index"),
        "expected duplicate-index stderr, got {}",
        String::from_utf8_lossy(&run_output.stderr)
    );
    assert_eq!(
        run_output.status.code(),
        Some(1),
        "expected clean runtime failure exit code, got {}",
        run_output.status
    );
    #[cfg(unix)]
    assert_eq!(
        run_output.status.signal(),
        None,
        "expected normal exit instead of signal, got {}",
        run_output.status
    );
}

#[test]
fn build_hip_accepts_dict_foundation_host_program() {
    let temp = tempdir().expect("tempdir");
    let out_dir = temp.path().join("hip-build");
    let source = dict_foundation_example();

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
        .stdout(predicate::str::contains("dict_foundation_hip.cpp"));

    assert!(out_dir.join("dict_foundation_hip.cpp").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("chelis_hip_runtime.h").exists());
}

#[test]
fn build_hip_accepts_iter_foundation_host_program() {
    let temp = tempdir().expect("tempdir");
    let out_dir = temp.path().join("hip-iter-build");
    let source = iter_foundation_example();

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
        .stdout(predicate::str::contains("iter_foundation_hip.cpp"))
        .stdout(predicate::str::contains("Compile: hipcc"))
        .stdout(predicate::str::contains("-lm"))
        .stdout(predicate::str::contains("-fopenmp").not());

    assert!(out_dir.join("iter_foundation_hip.cpp").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("chelis_hip_runtime.h").exists());
}

#[test]
fn build_c_runs_scalar_string_foundation_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("build-out");
    let source = scalar_string_foundation_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = gcc_link_generated(
        &out_dir,
        "scalar_string_foundation.c",
        "scalar_string_foundation",
    );
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("scalar_string_foundation"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
fn phase3m_rust_runtime_acceptance_oracle() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("phase3m-out");
    let source = scalar_string_foundation_example();

    let build = Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("-lchelis_runtime"))
        .stdout(predicate::str::contains("libchelis_runtime.a"))
        .get_output()
        .stdout
        .clone();

    let build_stdout = String::from_utf8(build).expect("build stdout utf8");
    assert!(out_dir.join("scalar_string_foundation.c").exists());
    assert!(out_dir.join("scalar_string_foundation.h").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("libchelis_runtime.a").exists());
    assert!(!out_dir.join("chelis_runtime.c").exists());
    assert!(
        !build_stdout.contains("chelis_runtime.c"),
        "Phase 3m oracle must not surface the deleted C runtime"
    );

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = gcc_link_generated(
        &out_dir,
        "scalar_string_foundation.c",
        "scalar_string_foundation",
    );
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("scalar_string_foundation"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
fn build_c_emits_host_function_for_mixed_tensor_scalar_program() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mixed_3c.ch");
    let out_dir = dir.path().join("mixed-build-out");
    write_file(
        &path,
        "def check_loss(x: tensor[4, f32], threshold: f32) -> string =\n  if tensor_to_scalar(mean(x, 0)) < threshold then \"converged\" else \"training\"\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("Compile object:"));

    let source = fs::read_to_string(out_dir.join("mixed_3c.c")).expect("generated source");
    assert!(source.contains("chelis_string check_loss(chelis_tensor* x, double threshold)"));
    assert!(source.contains("check_loss__tensor_0"));
    assert!(source.contains("chelis_tensor_to_f64"));
}

#[test]
fn build_c_runs_recursive_adt_program_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("jsonish.ch");
    let out_dir = dir.path().join("jsonish-build-out");
    write_file(
        &path,
        r#"type Json =
  | JsonNull
  | JsonInt(int64)
  | JsonString(string)
  | JsonArray(List[Json])

sample = JsonArray([JsonString("hi"), JsonInt(cast(3, int64))])
result = match sample with {
  | JsonNull => "null"
  | JsonInt(n) => to_string(n)
  | JsonString(s) => s
  | JsonArray(items) => string_concat("items=", to_string(len(items)))
}
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = gcc_link_generated(&out_dir, "jsonish.c", "jsonish");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("jsonish"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
#[ignore = "manual gate: hipcc is environment-dependent"]
fn build_hip_runs_scalar_string_foundation_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("hip-build-out");
    let source = scalar_string_foundation_example();

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
        .success();

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = hipcc_link_generated(
        &out_dir,
        "scalar_string_foundation_hip.cpp",
        "scalar_string_foundation_hip",
    );
    assert!(status.success(), "hipcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("scalar_string_foundation_hip"))
        .output()
        .expect("compiled hip binary should run");
    assert!(
        run_output.status.success(),
        "compiled hip binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
}

#[test]
fn build_fails_cleanly_when_chelis_runtime_dir_is_wrong() {
    let dir = tempdir().expect("tempdir");
    let bad_runtime_dir = dir.path().join("missing-runtime");
    fs::create_dir_all(&bad_runtime_dir).expect("create bad runtime dir");
    let out_dir = dir.path().join("bad-build");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_RUNTIME_DIR", &bad_runtime_dir)
        .args([
            "build",
            mnist_example().to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("CHELIS_RUNTIME_DIR"))
        .stderr(predicate::str::contains("libchelis_runtime.a"));
}

#[test]
fn build_honors_chelis_runtime_dir_override() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("env-runtime-build");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_RUNTIME_DIR", runtime_library_dir())
        .args([
            "build",
            mnist_example().to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("libchelis_runtime.a"));

    assert!(out_dir.join("libchelis_runtime.a").exists());
    assert!(!out_dir.join("chelis_runtime.c").exists());
}

#[test]
#[ignore = "manual gate: hipcc is environment-dependent"]
fn phase3m_rust_runtime_hip_manual_gate() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("phase3m-hip-out");
    let source = scalar_string_foundation_example();

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
        .stdout(predicate::str::contains("-lchelis_runtime"))
        .stdout(predicate::str::contains("libchelis_runtime.a"));

    assert!(out_dir.join("scalar_string_foundation_hip.cpp").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("libchelis_runtime.a").exists());
    assert!(out_dir.join("chelis_hip_runtime.h").exists());
    assert!(!out_dir.join("chelis_runtime.c").exists());

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", source.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = hipcc_link_generated(
        &out_dir,
        "scalar_string_foundation_hip.cpp",
        "scalar_string_foundation_hip",
    );
    assert!(status.success(), "hipcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("scalar_string_foundation_hip"))
        .output()
        .expect("compiled hip binary should run");
    assert!(
        run_output.status.success(),
        "compiled hip binary failed with status {}",
        run_output.status
    );
    assert_eq!(run_output.stdout, eval_stdout);
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
fn fmt_inplace_preserves_illustrative_example_parseability() {
    let dir = tempdir().expect("tempdir");
    let illustrative_root = example_path("../../examples/illustrative");
    for source in illustrative_examples() {
        let relative = source
            .strip_prefix(&illustrative_root)
            .expect("illustrative example should stay under root");
        let path = dir.path().join(relative);
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("create illustrative temp parents");
        }
        fs::copy(&source, &path).expect("copy illustrative example");

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
        .stdout(predicate::str::contains("a =").not());
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
        .stdout(predicate::str::contains("a = (a : tensor[2, 3, f32])"))
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
fn deep_annotate_emits_type_metadata_on_fn() {
    let dir = tempdir().expect("tempdir");
    let surf_path = dir.path().join("ident.ch");
    fs::write(&surf_path, "def f(x: f32) -> f32 = x\n").expect("write surf source");

    // Default (no --annotate) should leave fn meta empty.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", surf_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("(fn {}").or(predicate::str::contains("(fn {} ")));

    // --annotate should thread the inferred type onto the fn node.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["deep", "--annotate", surf_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("(fn {type: (t-fn"))
        .stdout(predicate::str::contains(
            "(t-fn {} (t-prim {} f32) (t-prim {} f32))",
        ));
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
compiler = "=0.2.1"
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
fn reef_std_embedding_module_checks_and_builds() {
    assert_reef_std_embedding_builds_to_valid_c();
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
compiler = "=0.2.1"
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
compiler = "=0.2.1"
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
compiler = "=0.2.1"
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
compiler = "=0.2.1"
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
compiler = "=0.2.1"
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
    write_file(&path, "x = add(1, true)\n");
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
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("libchelis_runtime.a").exists());
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
            out_dir.join("libchelis_runtime.a").display().to_string(),
        ))
        .stdout(predicate::str::contains("Peak device memory formula:"))
        .stdout(predicate::str::contains("Estimated peak device memory:"));

    assert!(out_dir.join("mnist_hip.cpp").exists());
    assert!(out_dir.join("mnist_hip.h").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("libchelis_runtime.a").exists());
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
fn build_hip_multidef_tensor_entry_uses_single_entry_abi() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bug3c.ch");
    let out_dir = dir.path().join("hip-output");
    write_file(
        &path,
        "def combine(a: tensor[4, f32], b: tensor[4, f32]) -> tensor[4, f32] = add(a, b)\n\
         def main(x: tensor[4, f32], y: tensor[4, f32]) -> tensor[4, f32] = combine(x, y)\n",
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
        .success();

    let hip_src = fs::read_to_string(out_dir.join("bug3c_hip.cpp")).expect("hip source");
    let hip_header = fs::read_to_string(out_dir.join("bug3c_hip.h")).expect("hip header");
    assert!(
        hip_header.contains("extern \"C\" void bug3c("),
        "expected HIP build to keep the single-entry ABI, got:\n{hip_header}"
    );
    assert!(
        hip_src.contains("expected %d inputs, got %d\\n\", 2, n_in"),
        "expected HIP build to preserve the two-input entry signature, got:\n{hip_src}"
    );
    assert!(
        !hip_src.contains("input `combine`")
            && !hip_src.contains("expected %d outputs, got %d\\n\", 2, n_out"),
        "HIP build must not treat helper defs as extra inputs or outputs, got:\n{hip_src}"
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
        "x: tensor[32, f32] = x\ny: tensor[32, f32] = dropout(x, 0.5)\n",
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
        "def bad(x: tensor[4, f32]): tensor[4, f32] = {\n  y: tensor[4, f32] = relu(x)\n  add(x, y)\n}\n",
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
        r#"def bad(pair: (tensor[4, f32], int32)): int32 = {
  n: int32 = match pair with {
    | (x, _) => 1
  }
  again: (tensor[4, f32], int32) = pair
  n
}
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
    write_file(&path, "x: int32 = with device(\"gpu:0\") { 1 }\n");

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
    write_file(&surf_path, "a = 1\nb = 2\n");

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
        "def f(axis) = { y = axis; y }\ndef g() = par { a; b }\n",
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

#[test]
fn validate_deep_rejects_invalid_effects_children() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad_effects.dp");
    write_file(&path, "(effects {} 1)\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "`effects` must contain bare names or `(resource {} ...)` entries",
        ));
}

#[test]
fn validate_deep_rejects_invalid_resource_arity() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad_resource.dp");
    write_file(&path, "(resource {} x y)\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("expected exactly 1 child"));
}

#[test]
fn reef_book_workflow_commands_are_valid() {
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

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "build", pkg.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Built demo 0.1.0"));
}
