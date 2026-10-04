// Tests only: Rust std functions on the clippy disallowed list compute
// reference or input values here; the list holds production code to
// chelis-crmath (chelis#2957).
#![allow(clippy::disallowed_methods)]
mod common;

use assert_cmd::Command;
use chelis_shell::{ShellSymbol, SymbolKind, read_shell, write_shell};
use common::authored_c_symbol;
use predicates::prelude::*;
use serde_json::Value;
use std::env;
use std::fs;
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

fn key_builtin_aliases_example() -> PathBuf {
    example_path("../../examples/key_builtin_aliases.ch")
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

fn hash_order_determinism_example() -> PathBuf {
    example_path("../../examples/hash_order_determinism.ch")
}

fn opaque_invariants_example() -> PathBuf {
    example_path("../../examples/opaque_invariants.ch")
}

fn opaque_invariants_simplex_example() -> PathBuf {
    example_path("../../examples/opaque_invariants_simplex.ch")
}

fn executable_examples() -> [PathBuf; 14] {
    [
        dict_foundation_example(),
        hello_tensor_example(),
        iter_foundation_example(),
        key_builtin_aliases_example(),
        list_foundation_example(),
        linreg_example(),
        mnist_example(),
        opaque_invariants_example(),
        opaque_invariants_simplex_example(),
        scalar_string_foundation_example(),
        tensor_structural_ops_example(),
        transformer_block_example(),
        hash_order_determinism_example(),
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

#[test]
fn issue_2160_authored_c_symbol_encodes_utf8_bytes() {
    assert_eq!(authored_c_symbol("é"), "chelis_fn_c3a9");
}

fn run_cost_json(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["cost", path.to_str().unwrap(), "--json"])
        .output()
        .expect("chelis cost --json should run");
    assert!(
        output.status.success(),
        "chelis cost --json failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout).expect("cost output should be json")
}

#[test]
fn cost_json_counts_lowered_copy_nodes_and_concrete_bytes() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("copy_cost.ch");
    write_file(
        &path,
        "def double_it(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = add(copy(x), x)\n",
    );

    let json = run_cost_json(&path);

    assert_eq!(json["file"], path.display().to_string());
    assert_eq!(json["total_copy_count"], 1);
    assert_eq!(json["total_bytes_copied"], 24);
    let functions = json["functions"].as_array().expect("functions array");
    assert_eq!(functions.len(), 1, "{json}");
    assert_eq!(functions[0]["name"], "double_it");
    assert_eq!(functions[0]["copy_count"], 1);
    assert_eq!(functions[0]["bytes_copied"], 24);
}

#[test]
fn cost_json_counts_implicit_copy_from_consuming_fanout() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("implicit_copy_cost.ch");
    write_file(
        &path,
        "def consume(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = realize(x)\n\
         def double_it(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = add(consume(x), consume(x))\n",
    );

    let json = run_cost_json(&path);

    assert_eq!(json["total_copy_count"], 1);
    assert_eq!(json["total_bytes_copied"], 24);
    let functions = json["functions"].as_array().expect("functions array");
    let double_it = functions
        .iter()
        .find(|function| function["name"] == "double_it")
        .expect("double_it summary");
    assert_eq!(double_it["copy_count"], 1);
    assert_eq!(double_it["bytes_copied"], 24);
}

#[test]
fn cost_json_counts_explicit_and_inserted_copy_nodes_once_each() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mixed_copy_cost.ch");
    write_file(
        &path,
        "def consume(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = realize(x)\n\
         def explicit_one(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = add(copy(x), x)\n\
         def implicit_one(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] = add(consume(x), consume(x))\n",
    );

    let json = run_cost_json(&path);

    assert_eq!(json["total_copy_count"], 2);
    assert_eq!(json["total_bytes_copied"], 48);
    let functions = json["functions"].as_array().expect("functions array");
    let explicit_one = functions
        .iter()
        .find(|function| function["name"] == "explicit_one")
        .expect("explicit_one summary");
    let implicit_one = functions
        .iter()
        .find(|function| function["name"] == "implicit_one")
        .expect("implicit_one summary");
    assert_eq!(explicit_one["copy_count"], 1);
    assert_eq!(implicit_one["copy_count"], 1);
}

#[test]
fn cost_json_omits_bytes_for_symbolic_copy_shapes() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_copy_cost.ch");
    write_file(
        &path,
        "def double_it[batch](x: tensor[batch, f32]) -> tensor[batch, f32] = add(copy(x), x)\n",
    );

    let json = run_cost_json(&path);

    assert_eq!(json["total_copy_count"], 1);
    assert!(
        json.get("total_bytes_copied").is_none(),
        "symbolic total bytes must be omitted: {json}"
    );
    let function = &json["functions"].as_array().expect("functions array")[0];
    assert_eq!(function["copy_count"], 1);
    assert!(
        function.get("bytes_copied").is_none(),
        "symbolic function bytes must be omitted: {json}"
    );

    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["cost", path.to_str().unwrap()])
        .output()
        .expect("chelis cost should run");
    assert!(
        output.status.success(),
        "chelis cost failed: stdout={} stderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    assert!(stdout.contains("copy_count=1"), "{stdout}");
    assert!(stdout.contains("bytes_copied=batch * 4"), "{stdout}");
}

#[test]
fn cost_json_accepts_reef_linked_package_names() {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join("linked-cost");
    fs::create_dir_all(pkg.join("src")).expect("create src");
    write_file(
        &pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "linked-cost"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "LinkedCost"
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(
        &pkg.join("src/helper.ch"),
        r#"module LinkedCost.Helper
export (double)
def double(x: tensor[2, f32]) -> tensor[2, f32] = add(x, x)
"#,
    );
    let main_path = pkg.join("src/main.ch");
    write_file(
        &main_path,
        r#"module LinkedCost.Main
import LinkedCost.Helper (double)
export (main)
def main(x: tensor[2, f32]) -> tensor[2, f32] = double(x)
"#,
    );

    let json = run_cost_json(&main_path);

    assert_eq!(json["file"], main_path.display().to_string());
    assert!(
        json["functions"]
            .as_array()
            .is_some_and(|functions| !functions.is_empty()),
        "linked package cost output must include at least one function: {json}"
    );
}

#[test]
fn cost_json_fixture_baseline_matches_documented_examples() {
    let baseline_path = example_path("tests/fixtures/copy_drop_fixture_fitness_baseline.json");
    let baseline: Value =
        serde_json::from_str(&fs::read_to_string(&baseline_path).expect("read baseline"))
            .expect("baseline json");
    let examples = baseline["examples"].as_array().expect("examples array");

    for example in examples {
        let file = example["file"].as_str().expect("example file");
        let path = example_path(&format!("../../{file}"));
        let json = run_cost_json(&path);

        assert_eq!(
            json["total_copy_count"], example["total_copy_count"],
            "copy-count baseline mismatch for {file}"
        );
        assert_eq!(
            json["total_bytes_copied"], example["total_bytes_copied"],
            "bytes-copied baseline mismatch for {file}"
        );
    }
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
    cmd.arg(out_dir.join("libchelis_runtime.a"));
    cmd.args(&toolchain.link_flags);
    cmd.args(["-o", binary]);
    cmd.status().expect("gcc should run")
}

fn gcc_compile_generated_object(out_dir: &Path, source: &str) -> std::process::ExitStatus {
    let toolchain = cpu_toolchain_for_sources(out_dir, &[source]);
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.args(["-I.", "-c", source]);
    cmd.status().expect("gcc should run")
}

fn gcc_link_sources(out_dir: &Path, sources: &[&str], binary: &str) -> std::process::ExitStatus {
    let toolchain = cpu_toolchain_for_sources(out_dir, sources);
    let mut cmd = StdCommand::new(&toolchain.compiler);
    cmd.current_dir(out_dir);
    cmd.arg("-O2");
    cmd.args(&toolchain.compile_flags);
    cmd.args(sources);
    cmd.arg(out_dir.join("libchelis_runtime.a"));
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
            "chelis_device_owner.cpp",
            "libchelis_runtime.a",
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
        "def f(a: tensor[batch, in_dim, f32], b: tensor[in_dim, out_dim, f32]) -> tensor[batch, out_dim, f32] = (matmul(a, b) : tensor[batch, out_dim, f32])\n",
    );
}

fn write_symbolic_softmax_program(path: &Path) {
    write_file(
        path,
        "def f(x: tensor[batch, seq, f32]) -> tensor[batch, seq, f32] = (softmax(x, 1) : tensor[batch, seq, f32])\n",
    );
}

fn write_symbolic_row_sum_program(path: &Path) {
    write_file(
        path,
        "def f(x: tensor[batch, seq, f32]) -> tensor[batch, f32] = (sum(x, 1) : tensor[batch, f32])\n",
    );
}

fn write_symbolic_layer_norm_program(path: &Path) {
    write_file(
        path,
        "def f(x: tensor[batch, 128, f32], gamma: tensor[128, f32], beta: tensor[128, f32]) -> tensor[batch, 128, f32] = (layer_norm(x, gamma, beta, 0.00001f32) : tensor[batch, 128, f32])\n",
    );
}

fn write_symbolic_hidden_layer_norm_program(path: &Path) {
    write_file(
        path,
        "def f(x: tensor[batch, hidden, f32], gamma: tensor[hidden, f32], beta: tensor[hidden, f32]) -> tensor[batch, hidden, f32] = (layer_norm(x, gamma, beta, 0.00001f32) : tensor[batch, hidden, f32])\n",
    );
}

// Issue #207: `chelis check` now exits non-zero when the JSON
// `errors` array is non-empty. These helpers are called from both
// clean-program and error-expecting tests in this file, so they
// drop the `.success()` assertion and capture stdout regardless.
fn run_json_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn run_json_check_show_inferred(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap(), "--show-inferred"])
        .output()
        .expect("run chelis check --show-inferred");
    serde_json::from_slice(&output.stdout).expect("check --show-inferred output should be json")
}

#[test]
fn check_default_output_omits_signature_inference_metadata() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("readonly.ch");
    write_file(&path, "def readonly(x, y: tensor[4, f32]) = add(x, y)\n");

    let json = run_json_check(&path);
    assert!(
        json.get("inferred_signatures").is_none(),
        "default check output must stay unchanged, got {json}"
    );
}

#[test]
fn check_show_inferred_prints_signature_inference_metadata() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("readonly.ch");
    write_file(&path, "def readonly(x, y: tensor[4, f32]) = add(x, y)\n");

    let json = run_json_check_show_inferred(&path);
    let signatures = json["inferred_signatures"]
        .as_array()
        .expect("inferred_signatures array");
    let readonly = signatures
        .iter()
        .find(|entry| entry["function"] == "readonly")
        .expect("readonly signature metadata");
    assert_eq!(
        readonly["display_signature"],
        "(&tensor[4, f32], tensor[4, f32]) -> tensor[4, f32]"
    );
    assert_eq!(readonly["params"][0]["name"], "x");
    assert_eq!(readonly["params"][0]["inferred_read_only"], true);
    assert_eq!(readonly["params"][0]["written"], false);
}

// Hull Phase 0a Packet B, commit 1: `chelis check --show-inferred
// --json` must emit a STRUCTURED, lossless type tree and effect row
// alongside the human display strings, so a consumer (Hull) does not
// have to re-parse a type printer. This pins the structured shape for a
// function carrying the IO effect.
#[test]
fn check_show_inferred_emits_structured_type_and_effect_row() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("io_fn.ch");
    // `debug` carries the IO effect and returns its argument unchanged,
    // so `logged` infers `(string) -> string` with an IO effect row.
    write_file(&path, "def logged(msg: string) -> string = debug(msg)\n");

    let json = run_json_check_show_inferred(&path);
    let signatures = json["inferred_signatures"]
        .as_array()
        .expect("inferred_signatures array");
    let logged = signatures
        .iter()
        .find(|entry| entry["function"] == "logged")
        .expect("logged signature metadata");

    // Human display strings remain present and unchanged.
    assert_eq!(logged["display_signature"], "(string) -> string");

    // Structured signature: a function from one string to a string.
    let sig = &logged["display_signature_structured"];
    assert_eq!(sig["kind"], "fn");
    assert_eq!(sig["args"].as_array().expect("args").len(), 1);
    assert_eq!(sig["args"][0]["kind"], "prim");
    assert_eq!(sig["args"][0]["name"], "string");
    assert_eq!(sig["ret"]["kind"], "prim");
    assert_eq!(sig["ret"]["name"], "string");

    // The checked signature tree is also present and equals the display
    // tree for this monomorphic function.
    assert_eq!(logged["checked_signature_structured"], *sig);

    // Structured per-parameter type tree.
    let param = &logged["params"][0];
    assert_eq!(param["name"], "msg");
    assert_eq!(param["display_type_structured"]["kind"], "prim");
    assert_eq!(param["display_type_structured"]["name"], "string");
    assert_eq!(param["checked_type_structured"]["kind"], "prim");
    assert_eq!(param["checked_type_structured"]["name"], "string");

    // Structured effect row: exactly one IO effect, internally tagged.
    let effect_row = logged["effect_row"].as_array().expect("effect_row array");
    assert_eq!(effect_row.len(), 1);
    assert_eq!(effect_row[0]["kind"], "io");
    // Human Display spelling matches `Effect::Display` (IO, not io).
    assert_eq!(
        logged["effect_row_display"]
            .as_array()
            .expect("effect_row_display array"),
        &vec![Value::from("IO")]
    );
}

// Hull Phase 0a Packet B, commit 1, structured tensor + negative
// parity: a PURE function over tensors must carry an EMPTY effect row
// (distinct from "effects unknown"), and the structured tensor type
// must reconstruct dims and the concrete precision losslessly. The
// borrowed parameter must serialize as a `ref` wrapping a `tensor`.
#[test]
fn check_show_inferred_pure_tensor_fn_has_empty_effect_row_and_structured_tensor() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("readonly.ch");
    write_file(&path, "def readonly(x, y: tensor[4, f32]) = add(x, y)\n");

    let json = run_json_check_show_inferred(&path);
    let signatures = json["inferred_signatures"]
        .as_array()
        .expect("inferred_signatures array");
    let readonly = signatures
        .iter()
        .find(|entry| entry["function"] == "readonly")
        .expect("readonly signature metadata");

    // Negative parity: a pure function emits an empty effect row, not a
    // missing field and not an unknown sentinel.
    assert_eq!(
        readonly["effect_row"].as_array().expect("effect_row").len(),
        0
    );
    assert_eq!(
        readonly["effect_row_display"]
            .as_array()
            .expect("effect_row_display")
            .len(),
        0
    );

    // Structured tensor: the borrowed first arg is `ref(tensor[lit 4, f32])`.
    let sig = &readonly["display_signature_structured"];
    assert_eq!(sig["kind"], "fn");
    let arg0 = &sig["args"][0];
    assert_eq!(arg0["kind"], "ref");
    let inner = &arg0["inner"];
    assert_eq!(inner["kind"], "tensor");
    assert_eq!(inner["dims"][0]["kind"], "lit");
    assert_eq!(inner["dims"][0]["size"], 4);
    assert_eq!(inner["precision"]["kind"], "concrete");
    assert_eq!(inner["precision"]["name"], "f32");

    // The owned second arg is a bare `tensor[lit 4, f32]` (no ref).
    let arg1 = &sig["args"][1];
    assert_eq!(arg1["kind"], "tensor");
    assert_eq!(arg1["dims"][0]["size"], 4);
    assert_eq!(arg1["precision"]["name"], "f32");
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
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["fmt", path.to_str().unwrap(), "--check"])
            .assert()
            .success();
    }
}

/// `examples/illustrative/README.md` promises that every file there passes
/// both `chelis fmt --check` and `chelis check`. The check runs on a copy of
/// the tree because checking a package source writes its `reef.lock`.
#[test]
fn fmt_check_and_check_accept_illustrative_examples() {
    let illustrative_root = example_path("../../examples/illustrative");
    let dir = tempdir().expect("tempdir");
    copy_dir_recursive(&illustrative_root, dir.path());
    for path in illustrative_examples() {
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["fmt", path.to_str().unwrap(), "--check"])
            .assert()
            .success();

        let copy = dir.path().join(
            path.strip_prefix(&illustrative_root)
                .expect("illustrative example should stay under root"),
        );
        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["check", copy.to_str().unwrap()])
            .output()
            .expect("run chelis check");
        assert!(output.status.success(), "{path:?}");
        let json: Value = serde_json::from_slice(&output.stdout).expect("check output is json");
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
fn fmt_check_accepts_canonical_stdlib_sources() {
    for path in stdlib_sources() {
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unbound variable: input"));
}

// G7 silent-no-output sub-bug pin: a Surf input that contains only
// `def` declarations with no top-level evaluable expression must
// surface a stderr warning so humans don't get a silent "success".
// Exit code stays 0 to preserve backward compat for scripted
// consumers. See `docs/archive/investigations/item2_sibling_sweep_findings.md`
// §G7 and `docs/investigations/cli_eval_empty_roots_diagnosis.md`.
#[test]
fn eval_def_only_emits_warning_on_stderr() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("def_only.ch");
    write_file(
        &path,
        "def double(x: &tensor[3, f32]) -> tensor[3, f32] = x + x\n\
         def triple(x: &tensor[3, f32]) -> tensor[3, f32] = x + x + x\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::is_empty())
        .stderr(predicate::str::contains(
            "input contains only def declarations; nothing to evaluate",
        ));
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        // chelis#732 P1 ([05-OBS-4]): rank-0 realizations render bare.
        .stdout(predicate::str::contains("grads.0 = 1.0"))
        .stdout(predicate::str::contains("grads.1 = 2.5"));
}

#[test]
fn eval_supports_host_scalars_and_strings() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            r#"bitxor(bitand(cast(7, i64), cast(3, i64)), shl(cast(1, i64), cast(2, i64)))"#,
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("7"));
}

#[test]
fn eval_surfaces_debug_transcript() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", r#"debug("trace")"#])
        .assert()
        .success()
        // Issue #912 [05-OBS-6]: single roots are now labelled.
        .stdout(predicate::str::contains("trace\neval_result = trace"));
}

/// Run `chelis eval --json EXPR` and parse stdout as JSON. Asserts the
/// command succeeded and stdout is a single JSON document.
fn run_eval_json_expr(expr: &str) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", expr])
        .output()
        .expect("run chelis eval --json");
    assert!(
        output.status.success(),
        "eval --json should succeed for `{expr}`: stderr={}",
        String::from_utf8_lossy(&output.stderr)
    );
    serde_json::from_slice(&output.stdout)
        .unwrap_or_else(|err| panic!("eval --json stdout should be JSON for `{expr}`: {err}"))
}

// Hull Phase 0a Packet B, commit 2: `chelis eval --json` emits the raw
// EvalResult as JSON on stdout. A host scalar integer expression yields
// a single root whose value is internally tagged `i64`.
#[test]
fn eval_json_emits_int64_scalar() {
    let json = run_eval_json_expr("mod(cast(17, i64), cast(5, i64))");
    let roots = json["roots"].as_array().expect("roots array");
    assert_eq!(roots.len(), 1);
    assert_eq!(
        roots[0]["name"], "eval_result",
        "the private expression wrapper must not leak into machine-facing root names"
    );
    assert_eq!(
        json["manifest"]["entries"][0]["name"], "eval_result",
        "realized roots and manifest entries must use the same public expression label"
    );
    assert_eq!(roots[0]["value"]["type"], "scalar");
    assert_eq!(roots[0]["value"]["value"]["dtype"], "int64");
    assert_eq!(roots[0]["value"]["value"]["value"], 2);
}

// A tensor expression yields a `tensor` value carrying shape + data.
#[test]
fn eval_json_emits_tensor_shape_and_data() {
    let json = run_eval_json_expr("to_tensor([1.0, 2.0, 3.0])");
    let roots = json["roots"].as_array().expect("roots array");
    assert_eq!(roots.len(), 1);
    let value = &roots[0]["value"];
    assert_eq!(value["type"], "tensor");
    assert_eq!(
        value["value"]["shape"].as_array().expect("shape"),
        &vec![Value::from(3)]
    );
    assert_eq!(
        // Execution wire v3 (chelis#729): tagged per-dtype payload.
        value["value"]["data"]["bits"].as_array().expect("data"),
        &vec![
            Value::from("3f800000"),
            Value::from("40000000"),
            Value::from("40400000")
        ]
    );
}

// A tuple value (internally tagged `tuple`) with two i64 elements. A
// top-level `(a, b)` binding is split by the evaluator into per-element
// roots, so a genuine `tuple` ExecutionValue is exercised by nesting the
// tuple inside a host list, where it survives as a single value.
#[test]
fn eval_json_emits_tuple_of_int64() {
    let json = run_eval_json_expr("[(cast(7, i64), cast(8, i64))]");
    let roots = json["roots"].as_array().expect("roots array");
    assert_eq!(roots.len(), 1);
    let list = &roots[0]["value"];
    assert_eq!(list["type"], "list");
    let tuple = &list["value"][0];
    assert_eq!(tuple["type"], "tuple");
    let elems = tuple["value"].as_array().expect("tuple elements");
    assert_eq!(elems.len(), 2);
    assert_eq!(elems[0]["type"], "scalar");
    assert_eq!(elems[0]["value"]["dtype"], "int64");
    assert_eq!(elems[0]["value"]["value"], 7);
    assert_eq!(elems[1]["type"], "scalar");
    assert_eq!(elems[1]["value"]["dtype"], "int64");
    assert_eq!(elems[1]["value"]["value"], 8);
}

// A top-level tuple binding splits into per-component roots. This pins
// the actual `chelis eval` behavior so the JSON contract is honest: a
// bare `(a, b)` does NOT produce a single `tuple` root.
#[test]
fn eval_json_top_level_tuple_splits_into_roots() {
    let json = run_eval_json_expr("(cast(7, i64), cast(8, i64))");
    let roots = json["roots"].as_array().expect("roots array");
    assert_eq!(roots.len(), 2);
    // Both components are scalar tensors through the IR evaluator path.
    assert_eq!(roots[0]["value"]["type"], "tensor");
    assert_eq!(roots[1]["value"]["type"], "tensor");
}

// `--file` form (non-reef legacy path) emits JSON for an evaluable
// top-level binding.
#[test]
fn eval_json_file_form_emits_json() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scalar.ch");
    write_file(&path, "answer = mod(cast(43, i64), cast(41, i64))\n");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval --json --file");
    assert!(output.status.success(), "eval --json --file should succeed");
    let json: Value = serde_json::from_slice(&output.stdout).expect("stdout JSON");
    let roots = json["roots"].as_array().expect("roots array");
    let answer = roots
        .iter()
        .find(|r| r["name"] == "answer")
        .expect("answer root");
    assert_eq!(answer["value"]["type"], "scalar");
    assert_eq!(answer["value"]["value"]["dtype"], "int64");
    assert_eq!(answer["value"]["value"]["value"], 2);
}

#[test]
fn eval_vmap_inferred_reference_row_keeps_reduction_rank() {
    let dir = tempdir().expect("tempdir");
    for row_type in ["&_", "&tensor[2, 2, f32]"] {
        let path = dir.path().join(if row_type == "&_" {
            "inferred_ref.ch"
        } else {
            "explicit_ref.ch"
        });
        write_file(
            &path,
            &format!(
                "out: tensor[2, 2, f32] = vmap(fn (v: {row_type}) -> sum(v, 0i32))(\
                 to_tensor([[[1.0f32, 2.0f32], [3.0f32, 4.0f32]], \
                 [[5.0f32, 6.0f32], [7.0f32, 8.0f32]]]))\n"
            ),
        );
        let check = run_json_check(&path);
        assert_eq!(check["score"], 1, "{row_type}: {check}");
        assert_eq!(
            check["errors"],
            serde_json::json!([]),
            "{row_type}: {check}"
        );

        let output = Command::cargo_bin("chelis")
            .expect("binary")
            .args(["eval", "--json", "--file", path.to_str().unwrap()])
            .output()
            .expect("evaluate vmap reference row");
        assert!(
            output.status.success(),
            "{row_type}: {}",
            String::from_utf8_lossy(&output.stderr)
        );
        let json: Value = serde_json::from_slice(&output.stdout).expect("eval JSON");
        let roots = json["roots"].as_array().expect("roots array");
        let result = roots
            .iter()
            .find(|root| root["name"] == "out")
            .expect("out root");
        assert_eq!(result["value"]["value"]["shape"], serde_json::json!([2, 2]));
        assert_eq!(
            result["value"]["value"]["data"]["bits"],
            serde_json::json!(["40800000", "40c00000", "41400000", "41600000"]),
            "{row_type}"
        );
    }
}

// Empty-roots input (only `def` declarations) emits valid JSON
// `{"roots":[]}` on stdout with exit 0, instead of the human-mode
// stderr-only breadcrumb. Negative parity for the non-empty cases.
#[test]
fn eval_json_def_only_emits_empty_roots_json() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("defonly.ch");
    write_file(&path, "def helper(x: i64) -> i64 = add(x, cast(1, i64))\n");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval --json --file");
    assert!(output.status.success(), "def-only eval --json exits 0");
    let stdout = String::from_utf8(output.stdout).expect("utf8 stdout");
    // EvalResult stamps its execution payload version (v4 since the key
    // execution values, chelis#2413).
    assert_eq!(
        stdout.trim(),
        r#"{"schema_version":4,"roots":[],"manifest":{"target":"Eval","entries":[],"requires_main":false}}"#
    );
    let json: Value = serde_json::from_str(stdout.trim()).expect("valid JSON");
    assert_eq!(json["roots"].as_array().expect("roots").len(), 0);
}

// Negative: a failing eval in `--json` mode still errors. Stdout carries
// no partial JSON; the error surfaces on stderr with a nonzero exit.
#[test]
fn eval_json_unbound_name_errors_with_empty_stdout() {
    let json_output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--json", "add(input, 1.0)"])
        .output()
        .expect("run chelis eval --json");
    assert!(
        !json_output.status.success(),
        "unbound name must fail in --json mode"
    );
    assert!(
        json_output.stdout.is_empty(),
        "stdout must stay empty on error, got {:?}",
        String::from_utf8_lossy(&json_output.stdout)
    );
    assert!(
        String::from_utf8_lossy(&json_output.stderr).contains("unbound variable: input"),
        "error must name the unbound variable on stderr"
    );
}

// Parity: text-mode (non-JSON) output is unchanged by the `--json`
// addition. The same scalar expression renders the human form on stdout.
#[test]
fn eval_text_mode_output_unchanged_alongside_json_flag() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "mod(cast(17, i64), cast(5, i64))"])
        .assert()
        .success()
        // Issue #912 [05-OBS-6]: single roots are now labelled.
        .stdout("eval_result = 2\n");
}

#[test]
fn check_collection_callback_errors_name_the_helper_contract() {
    let dir = tempdir().expect("tempdir");
    let cases = [
        (
            "filter_non_bool.ch",
            r#"
xs: List[i64] = [cast(1, i64)]
bad = filter(fn (x: i64) -> add(x, cast(1, i64)), xs)
"#,
            vec!["filter", "callback", "bool"],
        ),
        (
            "partition_non_bool.ch",
            r#"
xs: List[i64] = [cast(1, i64)]
bad = partition(fn (x: i64) -> add(x, cast(1, i64)), xs)
"#,
            vec!["partition", "callback", "bool"],
        ),
        (
            "fold_acc_mismatch.ch",
            r#"
xs: List[i64] = [cast(1, i64)]
bad = fold(fn (acc: string, x: i64) -> string_concat(acc, to_string(x)), cast(0, i64), xs)
"#,
            vec!["fold", "accumulator", "string", "i64"],
        ),
        (
            "scan_acc_mismatch.ch",
            r#"
xs: List[i64] = [cast(1, i64)]
bad = scan(fn (acc: string, x: i64) -> string_concat(acc, to_string(x)), cast(0, i64), xs)
"#,
            vec!["scan", "accumulator", "string", "i64"],
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--file",
            list_foundation_example().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("len=4, items=4, shape=2x2"))
        // chelis#732 P1: float list elements render with one fractional
        // digit; the genuinely int-typed token lists keep bare integers.
        .stdout(predicate::str::contains("\n[1.0, 2.0, 3.0]\n"))
        .stdout(predicate::str::contains("\n[2.0, 3.0, 4.0]\n"))
        .stdout(predicate::str::contains("\n[1, 2, 3]\n"))
        .stdout(predicate::str::contains("\n[[1, 2], [3]]\n"))
        .stdout(predicate::str::contains("[1.0, 2.0, 3.0, 4.0]\n"));
}

#[test]
fn eval_supports_dicts_and_iteration_collections() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        // chelis#732 P1: i64 sort indices print as integers.
        .stdout(predicate::str::contains(
            "sorted_indices = tensor(shape=[2], data=[0, 1])",
        ));
}

/// Cross-lane stdout comparison is plain BYTE equality since chelis#732
/// Phase 2 (section C2.3): both lanes render identical stored bits through
/// the one frozen [05-OBS] grammar, so the Phase 1 interim value
/// comparator that lived here is deleted as that phase promised. Any
/// difference is a real divergence and reports as the raw diff.
fn assert_stdout_value_parity(c_out: &[u8], eval_out: &[u8], label: &str) {
    let c_text = String::from_utf8_lossy(c_out);
    let eval_text = String::from_utf8_lossy(eval_out);
    assert_eq!(
        c_text, eval_text,
        "[{label}] compiled and eval stdout must be byte-identical \
         (chelis#732 section C2.3)"
    );
}

#[test]
fn build_c_runs_key_builtin_aliases_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("key-builtin-aliases-build-out");
    let source = key_builtin_aliases_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            "--emit-c",
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
    let eval_text = String::from_utf8_lossy(&eval_stdout);
    assert!(eval_text.contains("main.0 = key(0000000000000007)"));
    assert!(eval_text.contains("main.3 = tensor(shape=[2, 2], data=[key("));

    let status = gcc_link_generated(&out_dir, "key_builtin_aliases.c", "key_builtin_aliases");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("key_builtin_aliases"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    assert_stdout_value_parity(&run_output.stdout, &eval_stdout, "key_builtin_aliases");
}

#[test]
fn build_c_runs_list_foundation_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("list-build-out");
    let source = list_foundation_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
    assert_stdout_value_parity(&run_output.stdout, &eval_stdout, "list_foundation");
}

#[test]
fn build_c_runs_dict_foundation_and_matches_eval_output() {
    let temp = tempdir().expect("tempdir");
    let out_dir = temp.path().join("build");
    let source = dict_foundation_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let generated =
        fs::read_to_string(out_dir.join("iter_foundation.c")).expect("read generated iter C");
    let runtime_header =
        fs::read_to_string(out_dir.join("chelis_runtime.h")).expect("read copied runtime header");

    for symbol in [
        "chelis_list_with_capacity",
        "chelis_list_push_moved",
        "chelis_list_extend_moved",
        "chelis_list_append_owned",
        "chelis_list_concat_owned",
        "chelis_dict_insert_owned",
        "chelis_dict_merge_owned",
        "chelis_dict_remove_owned",
        "chelis_string_concat_owned",
    ] {
        assert!(
            !runtime_header.contains(symbol),
            "emitter-internal accumulator `{symbol}` must not enter the published C header"
        );
    }

    for declaration in [
        "chelis_list *chelis_list_with_capacity(int64_t capacity);",
        "void chelis_list_push_moved(chelis_list *list, chelis_value value);",
        "void chelis_list_extend_moved(chelis_list *list, chelis_list *src);",
        "chelis_list *chelis_list_append_owned(chelis_list *list, chelis_value value);",
        "chelis_list *chelis_list_concat_owned(chelis_list *lhs, const chelis_list *rhs);",
        "chelis_dict *chelis_dict_insert_owned(chelis_dict *dict, chelis_value key, chelis_value value);",
        "chelis_dict *chelis_dict_merge_owned(chelis_dict *lhs, const chelis_dict *rhs);",
        "chelis_dict *chelis_dict_remove_owned(chelis_dict *dict, chelis_value key);",
        "chelis_string chelis_string_concat_owned(chelis_string lhs, chelis_string rhs);",
    ] {
        assert!(
            generated.contains(declaration),
            "generated C must privately declare `{declaration}`"
        );
    }

    for (symbol, expected_calls) in [
        ("chelis_list_with_capacity(", 6),
        ("chelis_list_push_moved(", 5),
        ("chelis_list_extend_moved(", 1),
    ] {
        let occurrences = generated.matches(symbol).count();
        assert_eq!(
            occurrences - 1,
            expected_calls,
            "iter_foundation must exercise every in-place combinator path for `{symbol}`"
        );
    }
    assert!(
        !generated.contains(" = chelis_list_append("),
        "compiled combinator loops must not rebuild accumulators with list_append"
    );
    assert!(
        !generated.contains(" = chelis_list_concat("),
        "compiled flat_map must not rebuild its accumulator with list_concat"
    );

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
    assert_stdout_value_parity(&run_output.stdout, &eval_stdout, "tensor_structural_ops");
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
    // Issue #912 [05-OBS-6]: both lanes now label roots, so compare directly.
    assert_eq!(actual, expected_value);
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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

    let gram = authored_c_symbol("gram");
    write_file(
        &out_dir.join("driver.c"),
        &r#"#include "chelis_runtime.h"
#include "gram_bug5.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    int64_t shape[2] = {2, 3};
    chelis_tensor *a = chelis_alloc(2, shape, CHELIS_DTYPE_F32);
    float values[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    chelis_tensor_write *a_guard = chelis_tensor_begin_write(a);
    chelis_write_view a_view = chelis_tensor_write_view(a_guard);
    for (int i = 0; i < 6; ++i) {
        ((float *)a_view.data)[i] = values[i];
    }
    chelis_tensor_end_write(a_guard);

    chelis_tensor *out = CHELIS_TEST_GRAM(a);
    float expected[9] = {
        17.0f, 22.0f, 27.0f,
        22.0f, 29.0f, 36.0f,
        27.0f, 36.0f, 45.0f
    };
    chelis_read_view out_view = chelis_tensor_read_view(out);
    for (int i = 0; i < 9; ++i) {
        if (fabsf(((const float *)out_view.data)[i] - expected[i]) > 1e-4f) {
            fprintf(stderr, "mismatch at %d: got %f expected %f\n", i, ((const float *)out_view.data)[i], expected[i]);
            return 1;
        }
    }

    chelis_tensor_release(out);
    chelis_tensor_release(a);
    return 0;
}
"#
        .replace("CHELIS_TEST_GRAM", &gram),
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
fn build_c_user_defined_exports_remain_linkable_when_main_is_emitted() {
    // A root-bearing build is both executable and a published C translation
    // unit. Its authored definitions must match the external declarations in
    // the generated header; `main` emission cannot silently hide them.
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        c_src.contains(&format!("chelis_tensor* {}(", authored_c_symbol("combine")))
            && !c_src.contains(&format!(
                "static inline chelis_tensor* {}(",
                authored_c_symbol("combine")
            )),
        "expected combine() to retain external linkage in the executable translation unit, got:\n{c_src}"
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        header.contains(&format!("chelis_tuple* {}(", authored_c_symbol("eig_pair"))),
        "expected tuple-returning C ABI in generated header, got:\n{header}"
    );
    let generated = out_dir.join("tuple_abi.c");
    let source = fs::read_to_string(&generated).expect("generated c");
    fs::write(
        &generated,
        source.replace("int main(void)", "int chelis_manifest_main(void)"),
    )
    .expect("rename the generated observation driver for library-link probing");

    let eig_pair = authored_c_symbol("eig_pair");
    write_file(
        &out_dir.join("driver.c"),
        &r#"#include "chelis_runtime.h"
#include "tuple_abi.h"
#include <math.h>
#include <stdio.h>

int main(void) {
    chelis_tuple *out = CHELIS_TEST_EIG_PAIR();
    chelis_value lhs_value = chelis_tuple_get(out, 0);
    chelis_value rhs_value = chelis_tuple_get(out, 1);
    const chelis_tensor *lhs = chelis_tensor_borrow_value(lhs_value);
    const chelis_tensor *rhs = chelis_tensor_borrow_value(rhs_value);

    if (chelis_tensor_numel(lhs) != 2 || chelis_tensor_numel(rhs) != 2) {
        fprintf(stderr, "unexpected tuple tensor sizes\n");
        return 1;
    }
    chelis_read_view lhs_view = chelis_tensor_read_view(lhs);
    chelis_read_view rhs_view = chelis_tensor_read_view(rhs);
    if (fabsf(((const float *)lhs_view.data)[0] - 1.0f) > 1e-4f || fabsf(((const float *)lhs_view.data)[1] - 2.0f) > 1e-4f) {
        fprintf(stderr, "lhs mismatch\n");
        return 1;
    }
    if (fabsf(((const float *)rhs_view.data)[0] - 3.0f) > 1e-4f || fabsf(((const float *)rhs_view.data)[1] - 4.0f) > 1e-4f) {
        fprintf(stderr, "rhs mismatch\n");
        return 1;
    }

    chelis_value_release(lhs_value);
    chelis_value_release(rhs_value);
    chelis_tuple_release(out);
    return 0;
}
"#
        .replace("CHELIS_TEST_EIG_PAIR", &eig_pair),
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        source.contains(&format!("chelis_tensor* {}(", authored_c_symbol("combine"))),
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
        "def want_2x2(a: tensor[2, 2, f32]) -> tensor[f32] = trace(a, 0, 1)\n\
         def main(a: tensor[3, 3, f32]) -> tensor[f32] = want_2x2(a)\n",
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
    assert!(
        !json["errors"]
            .as_array()
            .expect("errors array")
            .iter()
            .any(|error| error["kind"] == "TypeMismatch"),
        "dimension mismatch fixture must not include an unrelated trace return TypeMismatch: {json}"
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
        "def want_2x2(a: tensor[2, 2, f32]) -> tensor[f32] = trace(a, 0, 1)\n\
         def bad_consumer[m, n](a: tensor[m, n, f32]) -> tensor[f32] = want_2x2(a)\n\
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
    assert!(
        !json["errors"]
            .as_array()
            .expect("errors array")
            .iter()
            .any(|error| error["kind"] == "TypeMismatch"),
        "polymorphic dimension fixture must not include an unrelated trace return TypeMismatch: {json}"
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        source.contains("float __arg"),
        "expected generated host C to keep f32 temps at their declared width:\n{source}"
    );
    assert!(
        !source.contains("double __arg"),
        "expected generated host C not to widen f32 temps to double:\n{source}"
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
        // chelis#2523: `step` is bound inside the declaration that applies it.
        // Its projections on `state` are obligations that only an application
        // resolves, and a later top-level declaration is not a binding site for
        // them ([04-INF-1]).
        "xs = to_list(to_tensor([1.0, 2.0]))\n\
         state0 = (to_tensor([0.0, 0.0]), cast(0.0, f32))\n\
         out = {\n\
           step = fn (state, x: f32) -> {\n\
             l_inner = state.0\n\
             total = state.1\n\
             (l_inner, add(total, x))\n\
           }\n\
           fold(step, state0, xs)\n\
         }\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
         grad_loss = grad(loss, wrt=theta)\n\
         xs: List[f32] = [1.0, 2.0]\n\
         rows = map(fn (x) -> grad_loss(to_tensor([1.0, 2.0]), x), xs)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("grad_rows_map.c")).expect("generated c");
    // WS-4: `xs: List[f32]` makes the map item an `f32`, so the callback
    // param lowers to a 4-byte `float`. Before the precision fix the host
    // lane collapsed the list element to `Float64`/`double`, widening the
    // declared `f32`. The key invariant this test guards is that the item
    // is a float scalar (not silently degraded to `int`).
    assert!(
        source.contains("float __map_item_") && source.contains("float x = __map_item_"),
        "expected map callback item to lower as float rather than int:\n{source}"
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
         grad_loss = grad(loss, wrt=theta)\n\
         xs: List[f32] = [1.0, -2.0]\n\
         rows = map(fn (x) -> grad_loss(to_tensor([1.0, 2.0]), x), xs)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("grad_rows_branching.c")).expect("generated c");
    // WS-4: the `x: f32` scalar param makes the host loss helper return a
    // `float` (the declared width), not the previously-widened `double`.
    // The generated public header declares authored functions externally, so
    // adding a manifest observation driver must not make this definition local.
    assert!(
        source.contains(&format!("float {}(", authored_c_symbol("loss")))
            && !source.contains(&format!(
                "static inline float {}(",
                authored_c_symbol("loss")
            )),
        "expected an externally linked host-side scalar loss helper:\n{source}"
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
        "def lm_residual[n](theta: tensor[n, f32], x: f32, y: f32) -> f32 = {\n\
           y_hat = if lt(x, cast(0.0, f32)) then tensor_to_scalar(sum(copy(theta), 0)) else add(tensor_to_scalar(sum(copy(theta), 0)), x)\n\
           sub(y, y_hat)\n\
         }\n\
         row = grad(lm_residual, wrt=theta)\n\
         def jac[n, m](theta: tensor[n, f32], xs: tensor[m, f32], ys: tensor[m, f32]) -> List[tensor[n, f32]] = {\n\
           pairs = zip(to_list(xs), to_list(ys))\n\
           map(fn (pair: (f32, f32)) -> row(copy(theta), pair.0, pair.1), pairs)\n\
         }\n\
         out = jac(to_tensor([1.0, 2.0]), to_tensor([1.0, -2.0]), to_tensor([3.0, 4.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        source.contains(&format!("chelis_tensor* {}(", authored_c_symbol("row")))
            && !source.contains(&format!(
                "static inline chelis_tensor* {}(",
                authored_c_symbol("row")
            )),
        "expected an externally linked host wrapper for the gradient row helper:\n{source}"
    );
    assert!(
        source.contains("__tensor_arg1_")
            && source.contains("chelis_alloc(0, NULL, CHELIS_DTYPE_F32)"),
        "expected host scalar dependencies to be boxed as rank-0 tensor helper inputs:\n{source}"
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
           grad(target, wrt=theta_local)(theta)\n\
         }\n\
         def lm_model(theta: tensor[2, f32], x: f32, y: f32) -> f32 = {\n\
           y_hat = if lt(x, cast(0.0, f32)) then tensor_to_scalar(sum(copy(theta), 0)) else add(tensor_to_scalar(sum(copy(theta), 0)), x)\n\
           sub(y, y_hat)\n\
         }\n\
         out = jac_row(lm_model, to_tensor([1.0, 2.0]), cast(1.0, f32), cast(3.0, f32))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    // [05-OP-33]: list ingress uses the one exact tagged constructor. There
    // is no untyped or dtype-named compatibility entry point.
    // The retained invocation prepares the actual once before formal ingress;
    // the tensor helper must consume that formal rather than reaching back to
    // the source expression and evaluating it again.
    let assignment_target = |rhs: &str, prefix: &str| {
        let suffix = format!(" = {rhs};");
        source
            .lines()
            .map(str::trim)
            .find(|line| line.starts_with(prefix) && line.ends_with(&suffix))
            .and_then(|line| line.split_once(" = "))
            .map(|(lhs, _)| lhs.to_string())
            .unwrap_or_else(|| panic!("expected `{prefix}* = {rhs};` in generated C:\n{source}"))
    };
    let prepared_line = source
        .lines()
        .map(str::trim)
        .find(|line| {
            line.starts_with("__retained_actual_") && line.contains(" = chelis_tensor_from_values(")
        })
        .unwrap_or_else(|| panic!("expected a prepared f32 tensor actual:\n{source}"));
    let prepared = prepared_line
        .split_once(" = ")
        .map(|(lhs, _)| lhs.to_string())
        .expect("prepared actual assignment");
    let checked_actual = assignment_target(&prepared, "__chelis_entry_actual_");
    let formal_value = assignment_target(&checked_actual, "__retained_actual_");
    let formal_ingress = assignment_target(&formal_value, "__chelis_entry_arg_");
    let tensor_input = assignment_target(&formal_ingress, "__tensor_arg0_");
    let helper_input = source
        .lines()
        .map(str::trim)
        .find(|line| {
            line.starts_with("__inputs_") && line.ends_with(&format!("[0] = {tensor_input};"))
        })
        .unwrap_or_else(|| panic!("expected `{tensor_input}` as helper input 0:\n{source}"));
    let ordered_markers = [
        prepared_line.to_string(),
        format!("{checked_actual} = {prepared};"),
        format!("{formal_value} = {checked_actual};"),
        format!("{formal_ingress} = {formal_value};"),
        format!("{tensor_input} = {formal_ingress};"),
        helper_input.to_string(),
        "tensor_grad_local_wrapper__global__tensor_0__private(__inputs_".to_string(),
    ];
    let mut cursor = 0;
    for marker in ordered_markers {
        let offset = source[cursor..]
            .find(&marker)
            .unwrap_or_else(|| panic!("expected ordered marker `{marker}`:\n{source}"));
        cursor += offset + marker.len();
    }
    assert!(
        prepared_line.contains("CHELIS_DTYPE_F32"),
        "expected the prepared list actual to retain its f32 dtype:\n{source}"
    );
    assert!(
        !source.contains("chelis_tensor_from_value_list("),
        "retired untyped list ingress must not be emitted:\n{source}"
    );
    assert!(
        !source.contains("chelis_tensor_from_value_list_typed("),
        "retired typed-by-name list ingress must not be emitted:\n{source}"
    );
    assert!(
        !source.contains("`grad` is not representable")
            && !source.contains("unsupported builtin")
            && !source.contains("int __binding_0_value;")
            && !source.contains("= to_tensor;")
            && !source.contains("__result = call("),
        "local-wrapper grad lowering must not degrade to fallback, unresolved builtins, or int locals:\n{source}"
    );

    let status = gcc_link_generated(
        &out_dir,
        "tensor_grad_local_wrapper.c",
        "tensor_grad_local_wrapper",
    );
    assert!(status.success(), "gcc failed with status {status}");
    let run_output = StdCommand::new(out_dir.join("tensor_grad_local_wrapper"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    assert!(
        stdout.contains("out = tensor(shape=[2], data=[-1.0, -1.0])"),
        "unexpected local-wrapper gradient output:\n{stdout}"
    );
}

/// chelis#405 oracle: host-lane scalar forward-mode AD. A top-level scalar
/// function `def square(x: f32) -> f32 = mul(x, x)` lands in the host lane,
/// which previously had no AD transform, so `grad(square, wrt=x)(x)` rejected
/// with the `__unresolved_grad` guard. With forward-mode dual lowering it now
/// builds to C: `dsquare(x) = grad(square, wrt=x)(x)` emits `2x`. Verifies:
/// build exits 0, the generated C compiles + links + runs, and `dsquare(2.0)`
/// equals 4.0 within a finite-difference tolerance of 1e-5.
#[test]
fn build_c_scalar_grad_builds_and_is_numerically_correct() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scalar_grad.ch");
    let out_dir = dir.path().join("scalar-grad-build-out");
    write_file(
        &path,
        "def square(x: f32) -> f32 = mul(x, x)\n\
         def dsquare(x: f32) -> f32 = grad(square, wrt=x)(x)\n\
         out = dsquare(2.0)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("scalar_grad.c")).expect("generated c");
    assert!(
        !source.contains("__result = call(")
            && !source.contains("unsupported builtin")
            && !source.contains("__unresolved_grad")
            && !source.contains("__unresolved_vmap"),
        "scalar grad lowering must not degrade to fallback / unresolved markers:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "scalar_grad.c", "scalar_grad");
    assert!(status.success(), "gcc link failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("scalar_grad"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    // Parse `out = <value>` from the printed line.
    let value: f64 = stdout
        .lines()
        .find_map(|line| line.strip_prefix("out = "))
        .and_then(|rhs| rhs.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("could not parse `out = <f64>` from:\n{stdout}"));

    // Finite-difference reference: d/dx(x*x) at x = 2.0.
    let f = |x: f64| x * x;
    let x = 2.0_f64;
    let h = 1e-4_f64;
    let fd = (f(x + h) - f(x - h)) / (2.0 * h);
    assert!(
        (value - fd).abs() < 1e-5,
        "dsquare(2.0) = {value}, finite-difference reference = {fd} (|Δ| >= 1e-5)"
    );
    // Exact analytic value is 4.0.
    assert!(
        (value - 4.0).abs() < 1e-5,
        "dsquare(2.0) must equal 4.0, got {value}"
    );
}

/// chelis#405: multi-parameter scalar forward-mode AD. `grad(f, wrt=(x, y))`
/// over a two-scalar-param function emits one dual pass per parameter and
/// combines the results into a gradient tuple. Verifies the build succeeds,
/// the binary runs, and `grad(x*y + sin(x))` at (1, 3) equals
/// `(y + cos(x), x) = (3 + cos(1), 1) ≈ (3.5403, 1.0)`.
#[test]
fn build_c_scalar_grad_multi_param_wrt_builds_and_is_numerically_correct() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scalar_grad_multi.ch");
    let out_dir = dir.path().join("scalar-grad-multi-build-out");
    write_file(
        &path,
        "def f(x: f32, y: f32) -> f32 = add(mul(x, y), sin(x))\n\
         def df(x: f32, y: f32) -> (f32, f32) = grad(f, wrt=(x, y))(x, y)\n\
         out = df(1.0, 3.0)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("scalar_grad_multi.c")).expect("generated c");
    assert!(
        !source.contains("__result = call(")
            && !source.contains("unsupported builtin")
            && !source.contains("__unresolved_grad"),
        "multi-param scalar grad must not degrade to fallback / unresolved markers:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "scalar_grad_multi.c", "scalar_grad_multi");
    assert!(status.success(), "gcc link failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("scalar_grad_multi"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "binary failed: {}",
        run_output.status
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    let dfdx: f64 = stdout
        .lines()
        .find_map(|line| line.strip_prefix("out.0 = "))
        .and_then(|rhs| rhs.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("could not parse `out.0 = <f64>` from:\n{stdout}"));
    let dfdy: f64 = stdout
        .lines()
        .find_map(|line| line.strip_prefix("out.1 = "))
        .and_then(|rhs| rhs.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("could not parse `out.1 = <f64>` from:\n{stdout}"));
    // df/dx = y + cos(x), df/dy = x at (x, y) = (1.0, 3.0).
    let expect_dx = 3.0 + 1.0_f64.cos();
    assert!(
        (dfdx - expect_dx).abs() < 1e-5,
        "df/dx = {dfdx}, expected {expect_dx}"
    );
    assert!((dfdy - 1.0).abs() < 1e-5, "df/dy = {dfdy}, expected 1.0");
}

/// The List `wrt` is a local parameter whose recursive shape C cannot
/// reconstruct. The checker accepts the gradient; C must reject it with
/// the typed #2740 diagnostic instead of emitting a wrong result.
#[test]
fn build_c_scalar_grad_rejects_container_wrt() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scalar_grad_container.ch");
    let out_dir = dir.path().join("scalar-grad-container-build-out");
    write_file(
        &path,
        "def g(xs: List[f32]) -> f32 = fold(fn (acc: f32, e: f32) -> add(acc, e), 0.0, xs)\n\
         def dg(xs: List[f32]) -> List[f32] = grad(g, wrt=xs)(xs)\n\
         out = dg([1.0, 2.0])\n",
    );

    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).expect("utf-8 stderr");
    for expected in [
        "unsupported: List gradient actual",
        "(lowering)",
        "unimplemented chelis#2740",
    ] {
        assert!(stderr.contains(expected), "missing `{expected}`:\n{stderr}");
    }
}

/// Build a scalar-grad program, link, run, and return the scalar value parsed
/// from the printed `out = <f64>` line. Asserts the build emits no fallback /
/// unresolved markers. Shared by the scalar-AD block-body, user-call, and
/// Black-Scholes Greeks tests below.
fn build_run_scalar_grad(stem: &str, source: &str) -> f64 {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    let out_dir = dir.path().join(format!("{stem}-build-out"));
    write_file(&path, source);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let c_source = fs::read_to_string(out_dir.join(format!("{stem}.c"))).expect("generated c");
    assert!(
        !c_source.contains("__result = call(")
            && !c_source.contains("unsupported builtin")
            && !c_source.contains("__unresolved_grad")
            && !c_source.contains("__unresolved_vmap"),
        "scalar grad lowering must not degrade to fallback / unresolved markers:\n{c_source}"
    );

    let status = gcc_link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(status.success(), "gcc link failed with status {status}");

    let run_output = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    stdout
        .lines()
        .find_map(|line| line.strip_prefix("out = "))
        .and_then(|rhs| rhs.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("could not parse `out = <f64>` from:\n{stdout}"))
}

/// chelis#405 negative parity: a (self-)recursive scalar def used as a `grad`
/// target must fail closed. The dual transform inlines user-defined calls;
/// without a depth bound a recursive callee would loop forever. The
/// `MAX_DUAL_INLINE_DEPTH` cap makes the transform return `None` at depth, so
/// the build falls through to the `__unresolved_grad` rejection with a clear
/// diagnostic rather than hanging or emitting wrong C.
#[test]
fn build_c_scalar_grad_recursive_callee_fails_closed() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scalar_grad_recursive.ch");
    let out_dir = dir.path().join("scalar-grad-recursive-build-out");
    write_file(
        &path,
        "def f(x: f32) -> f32 = mul(x, f(x))\n\
         def df(x: f32) -> f32 = grad(f, wrt=x)(x)\n\
         out = df(cast(2.0, f32))\n",
    );

    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .timeout(std::time::Duration::from_secs(60))
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure();
    let stderr = String::from_utf8(assert.get_output().stderr.clone()).expect("utf-8 stderr");
    assert!(
        stderr.contains("can't lower these defs"),
        "recursive scalar grad target must reject cleanly, got:\n{stderr}"
    );
}

/// chelis#405: host-lane scalar forward-mode AD through a `let`-block body.
/// The original implementation only handled single-expression scalar defs;
/// the canonical real driver (Black-Scholes Greeks) binds intermediates in a
/// block. `h(x) = { y = mul(x, x); add(y, x) } = x^2 + x`, so
/// `grad(h)(3.0) = 2*3 + 1 = 7.0`. Locks let-binding threading in `dual_eval`.
#[test]
fn build_c_scalar_grad_block_body_builds_and_is_numerically_correct() {
    let value = build_run_scalar_grad(
        "scalar_grad_block",
        "def h(x: f32) -> f32 = {\n\
         \x20 y = mul(x, x)\n\
         \x20 add(y, x)\n\
         }\n\
         def dh(x: f32) -> f32 = grad(h, wrt=x)(x)\n\
         out = dh(3.0)\n",
    );
    // d/dx(x^2 + x) at x = 3.0 is 2*3 + 1 = 7.0.
    let f = |x: f64| x * x + x;
    let x = 3.0_f64;
    let fd = (f(x + 1e-4) - f(x - 1e-4)) / (2.0 * 1e-4);
    assert!(
        (value - fd).abs() < 1e-4,
        "dh(3.0) = {value}, finite-difference reference = {fd}"
    );
    assert!(
        (value - 7.0).abs() < 1e-5,
        "dh(3.0) must equal 7.0, got {value}"
    );
}

/// chelis#405: host-lane scalar AD must differentiate through a call to a
/// user-defined scalar def (inlined into the dual tree). `outer(x) =
/// add(inner(x), x)` with `inner(x) = mul(x, x)`, so `outer(x) = x^2 + x` and
/// `grad(outer)(3.0) = 7.0`. Locks `dual_eval_user_call`.
#[test]
fn build_c_scalar_grad_through_user_defined_call_is_numerically_correct() {
    let value = build_run_scalar_grad(
        "scalar_grad_usercall",
        "def inner(x: f32) -> f32 = mul(x, x)\n\
         def outer(x: f32) -> f32 = add(inner(x), x)\n\
         def douter(x: f32) -> f32 = grad(outer, wrt=x)(x)\n\
         out = douter(3.0)\n",
    );
    assert!(
        (value - 7.0).abs() < 1e-5,
        "douter(3.0) must equal 7.0 (grad of x^2 + x at 3), got {value}"
    );
}

/// chelis#405 real-world driver: Black-Scholes scalar Greeks. `delta` and
/// `vega` differentiate a `call_price` that combines `let`-block bodies,
/// nested user-defined scalar calls (`d1`, `d2`, `normal_cdf`), and
/// transcendentals (`log`, `sqrt`, `exp`). This is the hello-chelis capstone
/// pattern named in the issue. A self-contained `normal_cdf` (Abramowitz &
/// Stegun rational approximation) stands in for the Nautilus import so the
/// test needs no reef dependency. The AD output is checked against a
/// high-accuracy `f64` central finite difference of the same formula.
#[test]
fn build_c_scalar_grad_black_scholes_greeks_are_numerically_correct() {
    const PRELUDE: &str = "def normal_cdf(x: f32) -> f32 = {\n\
         \x20 k = div(1.0, add(1.0, mul(0.2316419, x)))\n\
         \x20 poly = mul(k, add(0.31938153, mul(k, sub(0.356563782, mul(k, add(1.781477937, mul(k, sub(-1.821255978, mul(k, 1.330274429)))))))))\n\
         \x20 pdf = mul(0.3989422804014327, exp(neg(div(mul(x, x), 2.0))))\n\
         \x20 sub(1.0, mul(pdf, poly))\n\
         }\n\
         def d1(s: f32, k: f32, r: f32, sigma: f32, t: f32) -> f32 = {\n\
         \x20 num = add(log(div(s, k)), mul(add(r, mul(0.5, mul(sigma, sigma))), t))\n\
         \x20 den = mul(sigma, sqrt(t))\n\
         \x20 div(num, den)\n\
         }\n\
         def d2(s: f32, k: f32, r: f32, sigma: f32, t: f32) -> f32 = sub(d1(s, k, r, sigma, t), mul(sigma, sqrt(t)))\n\
         def call_price(s: f32, k: f32, r: f32, sigma: f32, t: f32) -> f32 = {\n\
         \x20 d_1 = d1(s, k, r, sigma, t)\n\
         \x20 d_2 = d2(s, k, r, sigma, t)\n\
         \x20 discount = exp(neg(mul(r, t)))\n\
         \x20 sub(mul(s, normal_cdf(d_1)), mul(mul(k, discount), normal_cdf(d_2)))\n\
         }\n";

    // delta = d(call_price)/ds at (s, k, r, sigma, t) = (100, 100, 0.05, 0.2, 1).
    let delta = build_run_scalar_grad(
        "bs_delta",
        &format!(
            "{PRELUDE}\
             def delta(s: f32, k: f32, r: f32, sigma: f32, t: f32) -> f32 = grad(call_price, wrt=s)(s, k, r, sigma, t)\n\
             out = delta(cast(100.0, f32), cast(100.0, f32), cast(0.05, f32), cast(0.2, f32), cast(1.0, f32))\n"
        ),
    );
    // vega = d(call_price)/dsigma at the same point.
    let vega = build_run_scalar_grad(
        "bs_vega",
        &format!(
            "{PRELUDE}\
             def vega(s: f32, k: f32, r: f32, sigma: f32, t: f32) -> f32 = grad(call_price, wrt=sigma)(s, k, r, sigma, t)\n\
             out = vega(cast(100.0, f32), cast(100.0, f32), cast(0.05, f32), cast(0.2, f32), cast(1.0, f32))\n"
        ),
    );

    // f64 reference of the same Horner-nested formula chelis evaluates.
    let normal_cdf = |x: f64| -> f64 {
        let k = 1.0 / (1.0 + 0.2316419 * x);
        let poly = k
            * (0.319381530
                + k * (0.356563782 - k * (1.781477937 + k * (-1.821255978 - k * 1.330274429))));
        let pdf = 0.3989422804014327 * (-x * x / 2.0).exp();
        1.0 - pdf * poly
    };
    let call_price = |s: f64, k: f64, r: f64, sigma: f64, t: f64| -> f64 {
        let d1 = (f64::ln(s / k) + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
        let d2 = d1 - sigma * t.sqrt();
        s * normal_cdf(d1) - k * (-r * t).exp() * normal_cdf(d2)
    };
    let (s, k, r, sigma, t) = (100.0_f64, 100.0, 0.05, 0.2, 1.0);
    let h = 1e-4_f64;
    let fd_delta =
        (call_price(s + h, k, r, sigma, t) - call_price(s - h, k, r, sigma, t)) / (2.0 * h);
    let fd_vega =
        (call_price(s, k, r, sigma + h, t) - call_price(s, k, r, sigma - h, t)) / (2.0 * h);

    // f32 AD vs f64 central difference: relative tolerance covers f32 rounding.
    assert!(
        (delta - fd_delta).abs() < 1e-3 * (1.0 + fd_delta.abs()),
        "delta = {delta}, finite-difference reference = {fd_delta}"
    );
    assert!(
        (vega - fd_vega).abs() < 1e-3 * (1.0 + fd_vega.abs()),
        "vega = {vega}, finite-difference reference = {fd_vega}"
    );
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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

/// Locally-bound `grad` alias form: `let g = grad(f, wrt=…); g(x)` must
/// lower in the C backend the same way the inline form `grad(f)(x)` already
/// does. The host backend recognizes `grad`/`vmap`/`vmap-grad` only in
/// direct callee position of an `app` node — without explicit β-substitution
/// of the alias the binding lowers to an `__unresolved_grad` builtin and
/// `host_program_unresolved_call_sites` rejects the program pre-codegen.
/// Verifies: build exits 0, the generated C contains no unresolved `call(…)`
/// stubs, the binary runs, and the gradient values match the inline form
/// numerically.
#[test]
fn build_c_grad_locally_bound_alias_form_lowers() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_local_alias.ch");
    let out_dir = dir.path().join("grad-local-alias-build-out");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[f32] =\n\
           sum(mul(theta, x), 0)\n\
         def compute_grad(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[2, f32] = {\n\
           g = grad(loss, wrt=theta)\n\
           g(theta, x)\n\
         }\n\
         out = compute_grad(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("grad_local_alias.c")).expect("generated c");
    assert!(
        !source.contains("__result = call(")
            && !source.contains("unsupported builtin")
            && !source.contains("__unresolved_grad")
            && !source.contains("__unresolved_vmap"),
        "generated C must not contain unresolved call stubs or unresolved-callable markers:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "grad_local_alias.c", "grad_local_alias");
    assert!(status.success(), "gcc link failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("grad_local_alias"))
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
        "locally-bound grad alias must produce the same gradient as the inline form, got:\n{stdout}"
    );
}

/// Parity test: `chelis eval` (IR evaluator path) and `chelis build
/// --target c` + run (C backend path) must agree on the locally-bound `grad`
/// alias form. Bucket 1 made the IR evaluator accept the alias form; this
/// guards against the C backend silently regressing relative to the
/// evaluator after the host-lane β-substitution pass. Output parity is
/// asserted on the alias form's value matching the inline form's value
/// — both as printed by the C runtime and as printed by `chelis eval`.
#[test]
fn build_c_grad_locally_bound_alias_form_matches_inline_form_output() {
    fn build_and_run(out_dir: &Path, source_path: &Path, source: &str, name: &str) -> String {
        write_file(source_path, source);
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                "--emit-c",
                source_path.to_str().unwrap(),
                "--target",
                "c",
                "--output",
                out_dir.to_str().unwrap(),
            ])
            .assert()
            .success();
        let status = gcc_link_generated(out_dir, &format!("{name}.c"), name);
        assert!(status.success(), "gcc link failed with status {status}");
        let run_output = StdCommand::new(out_dir.join(name))
            .output()
            .expect("compiled binary should run");
        assert!(
            run_output.status.success(),
            "compiled binary failed with status {}",
            run_output.status
        );
        String::from_utf8(run_output.stdout).expect("utf-8 stdout")
    }

    fn eval_to_string(source_path: &Path) -> String {
        let bytes = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["eval", "--file", source_path.to_str().unwrap()])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        String::from_utf8(bytes).expect("utf-8 eval stdout")
    }

    let dir = tempdir().expect("tempdir");
    let alias_dir = dir.path().join("alias-build");
    let inline_dir = dir.path().join("inline-build");
    let alias_src = dir.path().join("grad_local_alias_parity.ch");
    let inline_src = dir.path().join("grad_local_inline_parity.ch");

    let alias_program = "def loss(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[f32] =\n\
           sum(mul(theta, x), 0)\n\
         def compute_grad(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[2, f32] = {\n\
           g = grad(loss, wrt=theta)\n\
           g(theta, x)\n\
         }\n\
         out = compute_grad(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n";
    let inline_program = "def loss(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[f32] =\n\
           sum(mul(theta, x), 0)\n\
         def compute_grad(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[2, f32] =\n\
           grad(loss, wrt=theta)(theta, x)\n\
         out = compute_grad(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n";

    let alias_run_stdout = build_and_run(
        &alias_dir,
        &alias_src,
        alias_program,
        "grad_local_alias_parity",
    );
    let inline_run_stdout = build_and_run(
        &inline_dir,
        &inline_src,
        inline_program,
        "grad_local_inline_parity",
    );
    assert_eq!(
        alias_run_stdout, inline_run_stdout,
        "C-backend run output for the locally-bound grad alias form must match the inline form"
    );

    // chelis eval prints the last top-level value; both programs share the
    // same final `out` definition so eval output must agree across forms,
    // and must also agree with the C-backend's printed `out = …` line up
    // to the prefix.
    let alias_eval_stdout = eval_to_string(&alias_src);
    let inline_eval_stdout = eval_to_string(&inline_src);
    assert_eq!(
        alias_eval_stdout, inline_eval_stdout,
        "`chelis eval` output for the locally-bound grad alias form must match the inline form"
    );
    let trimmed_eval = alias_eval_stdout.trim_end();
    assert!(
        alias_run_stdout.contains(trimmed_eval),
        "C-backend run output must include the eval-printed gradient value;\n  eval: {trimmed_eval}\n  run:  {alias_run_stdout}"
    );
}

/// Regression test for the Coral UPSTREAM_BUGS.md pattern:
/// `grad(loss, wrt=x)(theta, x)` differentiates w.r.t. the second argument.
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
         out = grad(loss, wrt=x)(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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

// ─────────────────────────────────────────────────────────────────────────────
// Item 2c control + target: `grad(named_fn)(theta)` where `named_fn`'s body
// is canonical Surf using pipes vs. the equivalent nested-call form. Both
// shapes are spec-blessed (pipe is canonical Surf style per
// `spec/01-nomenclature.md` §3.6); both should compile through the C
// backend. The pipe form is currently rejected by
// `host_program_unresolved_call_sites` (`crates/chelis-ir/src/host.rs:1314`)
// because the host-lane summarizer doesn't recognize pipe-lowered function
// bodies as inlinable. See `docs/archive/investigations/c_backend_grad_piped_body_diagnosis.md`.
// ─────────────────────────────────────────────────────────────────────────────

/// Control fixture: `grad(named_fn)(theta)` where `named_fn`'s body is
/// written as nested calls (no pipes). Compiles cleanly today and must keep
/// compiling — locks the baseline so the pipe-form fix doesn't regress it.
#[test]
fn build_c_grad_over_named_fn_with_nested_call_body_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_nopipe_sumsq.ch");
    let out_dir = dir.path().join("grad-nopipe-sumsq-out");
    write_file(
        &path,
        "def sumsq(theta: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, theta), 0))\n\
         def gradient(theta: tensor[3, f32]) -> tensor[3, f32] = grad(sumsq)(theta)\n\
         out = gradient(to_tensor([1.0, 2.0, 3.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("grad_nopipe_sumsq.c")).expect("generated c");
    assert!(
        !source.contains("__result = call(") && !source.contains("unsupported builtin"),
        "generated C must not contain unresolved call stubs:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "grad_nopipe_sumsq.c", "grad_nopipe_sumsq");
    assert!(status.success(), "gcc link failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("grad_nopipe_sumsq"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    // grad(sum(theta*theta)) = 2 * theta = [2.0, 4.0, 6.0]
    assert!(
        stdout.contains("out = tensor(shape=[3], data=[2.0, 4.0, 6.0])"),
        "grad of sum(theta*theta) w.r.t. theta must be 2*theta=[2.0, 4.0, 6.0], got:\n{stdout}"
    );
}

/// Target fixture: same shape as the control, except `sumsq`'s body is
/// written as the canonical pipe form
/// `mul(theta, theta) |> sum(0) |> tensor_to_scalar`. Today the C backend
/// rejects this with the long-form "host lane can't resolve" diagnostic
/// (`crates/chelis-cli/src/main.rs:1571`, gated by
/// `host_program_unresolved_call_sites` at `crates/chelis-ir/src/host.rs:1314`).
/// After Item 2c, this must compile, link, and produce the same gradient
/// as the nested-call control fixture.
#[test]
fn build_c_grad_over_named_fn_with_pipe_body_builds() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_pipe_sumsq.ch");
    let out_dir = dir.path().join("grad-pipe-sumsq-out");
    write_file(
        &path,
        "def sumsq(theta: tensor[3, f32]) -> f32 = mul(theta, theta) |> sum(0) |> tensor_to_scalar\n\
         def gradient(theta: tensor[3, f32]) -> tensor[3, f32] = grad(sumsq)(theta)\n\
         out = gradient(to_tensor([1.0, 2.0, 3.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("grad_pipe_sumsq.c")).expect("generated c");
    assert!(
        !source.contains("__result = call(") && !source.contains("unsupported builtin"),
        "generated C must not contain unresolved call stubs:\n{source}"
    );

    let status = gcc_link_generated(&out_dir, "grad_pipe_sumsq.c", "grad_pipe_sumsq");
    assert!(status.success(), "gcc link failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("grad_pipe_sumsq"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    // grad(sum(theta*theta)) = 2 * theta = [2.0, 4.0, 6.0] — must match the
    // nested-call control fixture's output exactly.
    assert!(
        stdout.contains("out = tensor(shape=[3], data=[2.0, 4.0, 6.0])"),
        "grad of sum(theta*theta) w.r.t. theta must be 2*theta=[2.0, 4.0, 6.0], got:\n{stdout}"
    );
}

#[test]
fn build_c_recursive_tensor_function_stays_on_host_path() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("recursive_tensor.ch");
    let out_dir = dir.path().join("recursive-tensor-build-out");
    write_file(
        &path,
        "def recur[n](x: tensor[n, f32], i: i64) -> tensor[n, f32] =\n\
           if lte(i, cast(0, i64)) then x else recur(x, sub(i, cast(1, i64)))\n\
         out = recur(to_tensor([1.0, 2.0]), cast(2, i64))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        source.contains(&format!("chelis_tensor* {}(", authored_c_symbol("recur")))
            && !source.contains(&format!(
                "static inline chelis_tensor* {}(",
                authored_c_symbol("recur")
            )),
        "expected externally linked recursive tensor helper to stay in the host lane:\n{source}"
    );
    assert!(
        source.contains(&format!(
            "__result = {}__chelis_owned_body(",
            authored_c_symbol("recur")
        )),
        "expected recursive call to target the consuming C body rather than the external borrow adapter or a DAG helper:\n{source}"
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        source.contains(&format!("chelis_tensor* {}(", authored_c_symbol("chooser"))),
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    // WS-4: `cube(x: f32)` emits a 4-byte `float` signature, not the
    // previously-widened `double`. The downstream driver must match the
    // real ABI, so its forward declaration and call use `float` too.
    assert!(
        source.contains(&format!("float {}(float x)", authored_c_symbol("cube"))),
        "expected unreachable host def to survive build pruning for downstream drivers:\n{source}"
    );

    fs::write(
        &generated,
        source.replace("int main(void)", "int chelis_manifest_main(void)"),
    )
    .expect("rename the generated observation driver for library-link probing");
    let cube = authored_c_symbol("cube");
    write_file(
        &out_dir.join("driver.c"),
        &r#"#include <stdio.h>

float CHELIS_TEST_CUBE(float x);

int main(void) {
    printf("%.1f\n", (double)CHELIS_TEST_CUBE(3.0f));
    return 0;
}
"#
        .replace("CHELIS_TEST_CUBE", &cube),
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        source.contains(&format!(
            "chelis_tensor* {}(chelis_tensor* x)",
            authored_c_symbol("double_it")
        )),
        "expected generic tensor def to remain callable from downstream code:\n{source}"
    );
    assert!(
        !source.contains("(int64_t[]){ d"),
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "embedding-app"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

export (main)

def main(
  ids: tensor[2, 3, i64],
  table: tensor[8, 4, f32]
) -> tensor[2, 3, 4, f32] =
  gather(table, ids, 0)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "build",
            "--emit-c",
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

/// Per-primitive eval-vs-backend parity for the `round` (#395) and
/// `scatter_elements` (#396) RISC primitives: build to C, compile, run,
/// and assert the binary's stdout is byte-identical to `chelis eval` on
/// the same source. `round` exercises ties-to-even (0.5 -> 0, 2.5 -> 2,
/// distinguishing `rintf` from ties-away-from-zero `roundf`);
/// `scatter_elements` exercises the ONNX element-wise contract with an
/// i32-index tensor (the index-read path that a float-reinterpret bug
/// would silently corrupt).
#[test]
fn build_c_runs_round_and_scatter_elements_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("round_scatter_elements.ch");
    let out_dir = dir.path().join("round-scatter-elements-build-out");
    write_file(
        &path,
        "data = to_tensor([[cast(0.0, f32), cast(0.0, f32)], \
         [cast(0.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(0.0, f32)]])\n\
         indices = to_tensor([[cast(1, i32), cast(0, i32)], \
         [cast(2, i32), cast(0, i32)]])\n\
         updates = to_tensor([[cast(5.0, f32), cast(6.0, f32)], \
         [cast(7.0, f32), cast(8.0, f32)]])\n\
         scattered = scatter_elements(data, indices, updates, 0)\n\
         rounded = round(to_tensor([cast(0.5, f32), cast(1.5, f32), \
         cast(2.5, f32), cast(-2.5, f32)]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    let status = gcc_link_generated(
        &out_dir,
        "round_scatter_elements.c",
        "round_scatter_elements",
    );
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("round_scatter_elements"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status
    );
    // Value-identical across lanes; byte identity returns at chelis#732
    // Phase 2 (the int-typed `indices` line renders `.0`-free in eval now).
    assert_stdout_value_parity(&run_output.stdout, &eval_stdout, "round_scatter_elements");
}

#[test]
#[ignore = "manual gate: Phase 3h numeric acceptance oracle exceeds the default inner-loop budget"]
fn phase3h_numeric_acceptance_oracle() {
    build_c_runs_tensor_structural_ops_and_matches_eval_output();
    build_c_runs_round_and_scatter_elements_matches_eval_output();
    assert_reef_std_embedding_builds_to_valid_c();
}

/// A built program carries the correctly rounded kernels it calls
/// (spec/design/correctly_rounded_math.md section 4.2): the emitted unit
/// defines `chelis_cr_expf` with internal linkage, defines no kernel it does
/// not call, and names no math library. The old route through
/// `chelis_math.h` (Accelerate vForce on macOS, Sleef on Linux) gave
/// different bits per host.
#[test]
fn build_c_host_carries_the_correctly_rounded_kernel_it_calls() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("transcendental.ch");
    let out_dir = dir.path().join("c-output");
    write_file(
        &path,
        r#"
def softplus(x: tensor[4, f32]) -> tensor[4, f32] = exp(x)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let main_c = fs::read_to_string(out_dir.join("transcendental.c")).expect("emitted .c");
    assert_eq!(
        main_c
            .matches("static float chelis_cr_expf(float x)")
            .count(),
        1,
        "the unit must define the exp kernel it calls exactly once:\n{main_c}"
    );
    assert!(
        !main_c.contains("chelis_cr_logf(") && !main_c.contains("chelis_cr_exp("),
        "the unit must not carry kernels it does not call:\n{main_c}"
    );
    for library in [
        "chelis_math.h",
        "vvexpf",
        "Sleef_",
        "CHELIS_EXPF8",
        "Accelerate",
    ] {
        assert!(
            !main_c.contains(library),
            "generated C must name no math library (`{library}`):\n{main_c}"
        );
    }
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
fn check_accepts_static_scatter_duplicate_replace_as_last_write_wins() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scatter_dup_replace_bad.ch");
    write_file(
        &path,
        r#"
base = pad_sequences([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]], 0.0)
ids: List[i64] = [cast(1, i64), cast(1, i64)]
idx = to_tensor(ids)
updates = pad_sequences([[5.0, 5.0], [6.0, 6.0]], 0.0)
out = scatter(base, idx, updates, 0, "replace")
"#,
    );

    let json = run_json_check(&path);
    assert_eq!(json["score"].as_f64(), Some(1.0));
    let errors = json["errors"].as_array().expect("errors array");
    assert!(
        errors.is_empty(),
        "duplicate replace-scatter indices follow deterministic last-write-wins: {errors:?}"
    );
}

#[test]
fn build_c_scatter_duplicate_replace_is_deterministic_last_write_wins() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("scatter_runtime_bad.ch");
    let out_dir = dir.path().join("scatter-dup-build");
    write_file(
        &source,
        r#"
def apply(
  base: tensor[3, 2, f32],
  idx: tensor[2, i64],
  updates: tensor[2, 2, f32]
) -> tensor[3, 2, f32] =
  scatter(base, idx, updates, 0, "replace")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let apply = authored_c_symbol("apply");
    write_file(
        &out_dir.join("runner.c"),
        &r#"#include "chelis_runtime.h"
#include "scatter_runtime_bad.h"
#include <math.h>

int main(void) {
    int64_t base_shape[2] = {3, 2};
    int64_t idx_shape[1] = {2};
    int64_t updates_shape[2] = {2, 2};

    chelis_tensor *base = chelis_alloc(2, base_shape, CHELIS_DTYPE_F32);
    chelis_tensor *idx = chelis_alloc(1, idx_shape, CHELIS_DTYPE_I64);
    chelis_tensor *updates = chelis_alloc(2, updates_shape, CHELIS_DTYPE_F32);
    chelis_tensor_write *idx_guard = chelis_tensor_begin_write(idx);
    chelis_write_view idx_view = chelis_tensor_write_view(idx_guard);
    ((int64_t*)idx_view.data)[0] = 1;
    ((int64_t*)idx_view.data)[1] = 1;
    chelis_tensor_end_write(idx_guard);
    chelis_tensor_write *updates_guard = chelis_tensor_begin_write(updates);
    chelis_write_view updates_view = chelis_tensor_write_view(updates_guard);
    ((float *)updates_view.data)[0] = 5.0f;
    ((float *)updates_view.data)[1] = 5.0f;
    ((float *)updates_view.data)[2] = 6.0f;
    ((float *)updates_view.data)[3] = 6.0f;
    chelis_tensor_end_write(updates_guard);

    chelis_tensor *output = CHELIS_TEST_APPLY(base, idx, updates);
    const float expected[6] = {0.0f, 0.0f, 6.0f, 6.0f, 0.0f, 0.0f};
    chelis_read_view output_view = chelis_tensor_read_view(output);
    for (int i = 0; i < 6; i++) {
        if (fabsf(((const float *)output_view.data)[i] - expected[i]) > 1e-6f) {
            return 2;
        }
    }
    chelis_tensor_release(output);
    chelis_tensor_release(base);
    chelis_tensor_release(idx);
    chelis_tensor_release(updates);
    return 0;
}
"#
        .replace("CHELIS_TEST_APPLY", &apply),
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
        run_output.status.success(),
        "compiled duplicate replace-scatter must observe last-write-wins; status {} stderr: {}",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr)
    );
}

#[test]
fn build_hip_accepts_dict_foundation_host_program() {
    let temp = tempdir().expect("tempdir");
    let out_dir = temp.path().join("hip-build");
    let source = dict_foundation_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("-lchelis_runtime").not())
        .stdout(predicate::str::contains("libchelis_runtime.a"))
        .get_output()
        .stdout
        .clone();

    let build_stdout = String::from_utf8(build).expect("build stdout utf8");
    let staged_archive = out_dir.join("libchelis_runtime.a");
    let compile_line = build_stdout
        .lines()
        .find(|line| line.starts_with("Compile: "))
        .expect("build reports a compile command");
    assert!(
        compile_line.contains(&format!(" {} ", staged_archive.display())),
        "the compile command must link the staged archive by path: {compile_line}"
    );
    assert!(
        build_stdout.contains(&format!(
            "Staged runtime {} (sha256 {})",
            staged_archive.display(),
            chelis_runtime_bundle::carried_sha256().expect("carried runtime digest")
        )),
        "{build_stdout}"
    );
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    // WS-4: the declared `f32` scalar param emits a 4-byte `float` C type.
    // Before the precision fix the host lane collapsed every float to
    // `Float64` and widened this to `double`, silently disagreeing with
    // the declared `f32` width.
    assert!(
        source.contains(&format!(
            "chelis_string {}(chelis_tensor* x, float threshold)",
            authored_c_symbol("check_loss")
        )),
        "{source}"
    );
    assert!(source.contains(&format!("{}__tensor_0", authored_c_symbol("check_loss"))));
    assert!(source.contains("chelis_tensor_to_scalar"));
}

#[test]
fn build_c_runs_recursive_adt_program_and_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("jsonish.ch");
    let out_dir = dir.path().join("jsonish-build-out");
    write_file(
        &path,
        r#"type Jsonish =
  | JsonNull
  | JsonInt(i64)
  | JsonString(string)
  | JsonArray(List[Jsonish])

sample = JsonArray([JsonString("hi"), JsonInt(cast(3, i64))])
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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

/// chelis#1354: chelis stages the runtime built into it; a runtime location
/// variable is rejected before any output is written, whatever the target.
#[test]
fn build_rejects_chelis_runtime_dir_before_writing_outputs() {
    let dir = tempdir().expect("tempdir");
    let runtime_dir = dir.path().join("runtime");
    fs::create_dir_all(&runtime_dir).expect("create runtime dir");
    fs::write(
        runtime_dir.join("libchelis_runtime.a"),
        b"!<arch>\nforeign runtime",
    )
    .expect("write foreign archive");

    for target in ["c", "hip", "metal"] {
        let out_dir = dir.path().join(format!("build-{target}"));
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .env("CHELIS_RUNTIME_DIR", &runtime_dir)
            .args([
                "build",
                "--emit-c",
                mnist_example().to_str().unwrap(),
                "--target",
                target,
                "--output",
                out_dir.to_str().unwrap(),
            ])
            .assert()
            .failure()
            .stderr(predicate::str::contains("CHELIS_RUNTIME_DIR is set"))
            .stderr(predicate::str::contains("Unset CHELIS_RUNTIME_DIR"));
        assert!(!out_dir.exists(), "a rejected {target} build wrote output");
    }
}

/// Regression for issue #261: `chelis build --target c` must reject
/// `reduce_window_*` over a runtime-symbolic windowed axis with a clean
/// `unsupported_feature` diagnostic, rather than silently emit a
/// mis-allocated, out-of-bounds kernel whose output diverges from the
/// evaluator (or panic in the emitter). `pad_sequences` produces a
/// runtime-bound trailing extent, which is the windowed axis here. The
/// host runtime / IR evaluator handle this case correctly; only the
/// ahead-of-time build path is restricted. See
/// `spec/05-risc-primitives.md` §2.3.1.
#[test]
fn build_c_rejects_reduce_window_over_runtime_symbolic_axis() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rw_symbolic.ch");
    write_file(
        &path,
        "padded = pad_sequences([[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]], 0.0)\n\
         windowed = reduce_window_max(padded, [2i64], [1i64])\n",
    );
    let out_dir = dir.path().join("rw-build");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "requires statically-known windowed-axis extents",
        ))
        .stderr(predicate::str::contains("reduce_window"))
        // Must be the clean guard, not the emitter backstop panic.
        .stderr(predicate::str::contains("panicked").not());
}

/// Regression for PR #261 review finding #1: `chelis build --target c` on a
/// bf16 `reduce_window_*` must fail with a clean `unsupported_feature`
/// diagnostic, not an emitter `panic!`. The C backend admits bf16 generally,
/// but the C windowed-reduction emitter is f32-only.
/// See `spec/05-risc-primitives.md` §2.3.1.
#[test]
fn build_c_rejects_bf16_reduce_window_with_clean_diagnostic() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rw_bf16.ch");
    write_file(
        &path,
        "def pool_bf16(x: tensor[1, 1, 4, 4, bf16]) -> tensor[1, 1, 3, 3, bf16] = \
         reduce_window_max(&x, [2i64, 2i64], [1i64, 1i64])\n",
    );
    let out_dir = dir.path().join("rw-bf16-build");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("f32"))
        .stderr(predicate::str::contains("reduce_window"))
        // Must be the clean guard, not the emitter backstop panic.
        .stderr(predicate::str::contains("panicked").not());
}

/// chelis#2339 owns the still-inexact HIP reduction cell. Selecting a host
/// entry must not turn the C implementation into a silent device fallback.
#[test]
fn build_device_targets_reject_host_reduce_window_max_without_c_fallback() {
    for target in ["hip", "metal"] {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join(format!("rw_{target}.ch"));
        write_file(
            &path,
            "def pool(x: tensor[1, 1, 4, 4, f32]) -> tensor[1, 1, 3, 3, f32] = \
             reduce_window_max(&x, [2i64, 2i64], [1i64, 1i64])\n",
        );
        let out_dir = dir.path().join(format!("rw-{target}-build"));

        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                "--emit-c",
                path.to_str().unwrap(),
                "--target",
                target,
                "--output",
                out_dir.to_str().unwrap(),
            ])
            .assert()
            .failure()
            .stderr(predicate::str::contains("reduce_window_max"))
            .stderr(predicate::str::contains("unimplemented chelis#2339"))
            .stderr(predicate::str::contains(format!("(codegen:{target})")))
            .stderr(predicate::str::contains("panicked").not());
    }
}

/// Device window extrema remain fenced until their exact semantics are
/// implemented; this is an implementation gap, not [05-RWIN-2].
#[test]
fn build_hip_host_rejects_unimplemented_window_dtype_cleanly() {
    let dir = tempdir().unwrap();
    let path = dir.path().join("rw_hip_bf16.ch");
    write_file(
        &path,
        "def pool_hip(x: tensor[1, 1, 4, 4, bf16]) -> tensor[1, 1, 3, 3, bf16] = \
         reduce_window_max(&x, [2i64, 2i64], [1i64, 1i64])\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            dir.path().join("out").to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("reduce_window"))
        .stderr(predicate::str::contains("bf16"))
        .stderr(predicate::str::contains("unimplemented chelis#2339"))
        .stderr(predicate::str::contains("(codegen:hip)"))
        .stderr(predicate::str::contains("panicked").not());
}

/// chelis#1354: runtime archives left beside the chelis executable, newer
/// than the build, never replace the runtime chelis carries. The former
/// resolver took the newest archive found near the executable.
#[test]
fn build_stages_the_carried_runtime_not_newer_archives_nearby() {
    let dir = tempdir().expect("tempdir");
    let bin_dir = dir.path().join("bin");
    fs::create_dir_all(bin_dir.join("deps")).expect("create bin/deps");
    fs::create_dir_all(dir.path().join("lib")).expect("create lib");
    let chelis = bin_dir.join("chelis");
    fs::copy(env!("CARGO_BIN_EXE_chelis"), &chelis).expect("copy chelis");
    let foreign: &[u8] = b"!<arch>\nforeign runtime";
    for archive in [
        bin_dir.join("deps/libchelis_runtime-ffffffffffffffff.a"),
        bin_dir.join("libchelis_runtime.a"),
        dir.path().join("lib/libchelis_runtime.a"),
    ] {
        fs::write(&archive, foreign).expect("write foreign archive");
    }
    let out_dir = dir.path().join("out");

    let output = StdCommand::new(&chelis)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env_remove("CHELIS_RUNTIME_DIR")
        .args([
            "build",
            "--emit-c",
            mnist_example().to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run chelis");
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        output.stderr.is_empty(),
        "successful builds are silent on stderr"
    );

    let staged_archive = out_dir.join("libchelis_runtime.a");
    let staged = fs::read(&staged_archive).expect("staged archive");
    assert_ne!(staged, foreign, "a foreign archive was staged");
    let carried = chelis_runtime_bundle::carried_sha256().expect("carried runtime digest");
    let stdout = String::from_utf8(output.stdout).expect("stdout utf8");
    assert!(
        stdout.contains(&format!(
            "Staged runtime {} (sha256 {carried})",
            staged_archive.display()
        )),
        "{stdout}"
    );
    let receipt: Value = serde_json::from_slice(
        &fs::read(out_dir.join("chelis_runtime.receipt.json")).expect("staging receipt"),
    )
    .expect("receipt JSON");
    assert_eq!(receipt["archive_sha256"], carried.as_str());
    assert_eq!(receipt["mode"], chelis_runtime_bundle::MODE);
}

/// chelis#1354: `chelis runtime export` writes exactly the runtime chelis
/// carries, for packaging, and rejects a runtime location variable first.
#[test]
fn runtime_export_writes_the_carried_runtime() {
    use sha2::{Digest, Sha256};

    let dir = tempdir().expect("tempdir");
    let export_dir = dir.path().join("runtime");
    let stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env_remove("CHELIS_RUNTIME_DIR")
        .args(["runtime", "export", export_dir.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let carried = chelis_runtime_bundle::carried_sha256().expect("carried runtime digest");
    let archive = export_dir.join("libchelis_runtime.a");
    let stdout = String::from_utf8(stdout).expect("stdout utf8");
    assert!(
        stdout.contains(&format!(
            "Staged runtime {} (sha256 {carried})",
            archive.display()
        )),
        "{stdout}"
    );
    let digest: String = Sha256::digest(fs::read(&archive).expect("exported archive"))
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect();
    assert_eq!(digest, carried);
    for (name, contents) in chelis_runtime_bundle::PUBLIC_HEADERS {
        assert_eq!(
            fs::read_to_string(export_dir.join(name)).expect("exported header"),
            *contents
        );
    }
    assert!(export_dir.join("chelis_runtime.receipt.json").is_file());

    let rejected = dir.path().join("rejected");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_RUNTIME_DIR", dir.path())
        .args(["runtime", "export", rejected.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("Unset CHELIS_RUNTIME_DIR"));
    assert!(
        !rejected.exists(),
        "a rejected export created its directory"
    );
}

#[test]
#[ignore = "manual gate: hipcc is environment-dependent"]
fn phase3m_rust_runtime_hip_manual_gate() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("phase3m-hip-out");
    let source = scalar_string_foundation_example();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("-lchelis_runtime").not())
        .stdout(predicate::str::contains("Staged runtime"));

    assert!(out_dir.join("scalar_string_foundation_hip.cpp").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("libchelis_runtime.a").exists());
    assert!(out_dir.join("chelis_hip_runtime.h").exists());
    assert!(!out_dir.join("chelis_runtime.c").exists());

    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--inplace"])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["fmt", path.to_str().unwrap(), "--inplace"])
            .assert()
            .success();

        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", hello_tensor_example().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&path, output).expect("write deep");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--check"])
        .assert()
        .success();
}

#[test]
fn fmt_deep_rejects_a_bare_name_in_a_runtime_position() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bare_runtime_name.dp");
    write_file(&path, "(def {} f x)\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("bare name"));
}

#[test]
fn fmt_check_succeeds_for_canonical_surf() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mnist.ch");
    fs::copy(mnist_example(), &path).expect("copy");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--inplace"])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        "def f[n](x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--check"])
        .assert()
        .success();
}

// #290: `chelis fmt` must keep the grouping parens around a function-typed
// parameter in a `sig`. Arrow types are right-associative, so `(a -> b) -> c`
// (one function-typed argument) is a different type from the curried 3-ary
// `a -> b -> c`. The formatter previously stripped the parens, changing the
// arity and making higher-order sigs impossible to write fmt-clean.
#[test]
fn fmt_inplace_preserves_hof_argument_parens() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("hof.ch");
    write_file(
        &path,
        "module T\nsig f[a, b, c]: (a -> b) -> c\ndef f(g, x) = x\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--inplace"])
        .assert()
        .success();

    let after = fs::read_to_string(&path).expect("read formatted file");
    assert!(
        after.contains("sig f[a, b, c]: (a -> b) -> c"),
        "fmt stripped the function-typed argument's grouping parens; got:\n{after}"
    );
    assert!(
        !after.contains("sig f[a, b, c]: a -> b -> c"),
        "fmt flattened the HOF sig to the curried form; got:\n{after}"
    );

    // The formatted output must be stable under a second pass (idempotence).
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--check"])
        .assert()
        .success();
}

// #290 negative parity: a genuinely curried sig must NOT gain spurious parens,
// and a redundant right-position group `a -> (b -> c)` canonicalizes to the
// bare flat form.
#[test]
fn fmt_inplace_leaves_curried_sig_unparenthesized() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("curried.ch");
    write_file(
        &path,
        "module T\nsig f[a, b, c]: a -> (b -> c)\ndef f(x, y, z) = x\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--inplace"])
        .assert()
        .success();

    let after = fs::read_to_string(&path).expect("read formatted file");
    assert!(
        after.contains("sig f[a, b, c]: a -> b -> c"),
        "redundant right-position arrow parens were not canonicalized away; got:\n{after}"
    );
    assert!(
        !after.contains("(b -> c)"),
        "right-position arrow kept spurious parens; got:\n{after}"
    );
}

// #144: `chelis fmt --inplace` must preserve `--` line comments and
// `{- -}` block comments instead of deleting them, and the result must
// pass `fmt --check` (be idempotent).
#[test]
fn fmt_inplace_preserves_surf_comments() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("commented.ch");
    write_file(
        &path,
        "module School.Comment_Test\n\
         -- regular comment 1\n\
         --- triple-dash\n\
         {- block comment -}\n\
         def main() -> f32 = cast(1.0, f32)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--inplace"])
        .assert()
        .success();

    let after = fs::read_to_string(&path).expect("read back");
    assert!(
        after.contains("-- regular comment 1"),
        "line comment was stripped by fmt; got:\n{after}"
    );
    assert!(
        after.contains("--- triple-dash"),
        "triple-dash comment was stripped by fmt; got:\n{after}"
    );
    assert!(
        after.contains("{- block comment -}"),
        "block comment was stripped by fmt; got:\n{after}"
    );

    // The formatted-with-comments output must itself be canonical.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--check"])
        .assert()
        .success();
}

#[test]
fn fmt_check_fails_for_noncanonical_deep() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("program.dp");
    // A valid declaration with non-canonical formatting (all on one line
    // instead of indented multi-line):
    write_file(
        &path,
        "(def {} x (app {} (var {} mean) (app {} (var {} neg) (app {} (var {} sum) (var {} very_long_intermediate_name) (lit {type: (t-prim {} i32)} 0)))))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--check"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("is not canonically formatted"));
}

#[test]
fn fmt_rejects_check_and_inplace_together() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("program.dp");
    write_file(&path, "(def {} x (lit {type: (t-prim {} i32)} 1))\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap(), "--check", "--inplace"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "does not allow `--inplace` and `--check` together",
        ));
}

#[test]
fn surf_default_preserves_top_level_value_bindings() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("forward.ch");
    write_matmul_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["surf", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("a = (a : tensor[2, 3, f32])"))
        .stdout(predicate::str::contains(
            "out = (matmul(a, b) : tensor[2, 4, f32])",
        ))
        .stdout(predicate::str::contains("def forward").not());
}

#[test]
fn surf_verbose_preserves_debug_style_annotations() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("forward.ch");
    write_matmul_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["surf", path.to_str().unwrap(), "--verbose"])
        .assert()
        .success()
        .stdout(predicate::str::contains("a = (a : tensor[2, 3, f32])"))
        .stdout(predicate::str::contains(
            "(matmul(a, b) : tensor[2, 4, f32])",
        ));
}

#[test]
fn surf_roundtrip_preserves_canonical_def_return_types() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("typed.ch");
    write_file(
        &path,
        "def f[n](x: tensor[n, f32]) -> tensor[n, f32] = relu(x)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["surf", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "def f[n](x: tensor[n, f32]) -> tensor[n, f32] =",
        ))
        .stdout(predicate::str::contains("sig f").not());
}

#[test]
fn surf_deep_cast_and_par_emit_canonical_spellings() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("cast_par.dp");
    write_file(
        &path,
        "(def {} result (fn {} (params {}) (par {}\n  (cast {} (lit {} 1.0) (t-prim {} f32))\n  (cast {} (lit {} 2.0) (t-prim {} f32)))))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("par {"))
        .stdout(predicate::str::contains("cast(1.0, f32)"))
        .stdout(predicate::str::contains("cast(2.0, f32)"))
        .stdout(predicate::str::contains(" as ").not())
        .stdout(predicate::str::contains("par(").not());
}

#[test]
fn surf_rejects_malformed_deep_cast_before_decompiling() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("malformed_cast.dp");
    write_file(
        &path,
        "(def {} result (fn {} (params {}) (cast {} (lit {} 1.0))))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["surf", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("wrong child count for `cast`"));
}

#[test]
fn phase3e_pipe_first_acceptance_oracle() {
    let dir = tempdir().expect("tempdir");
    let surf_path = dir.path().join("mnist.ch");
    let deep_path = dir.path().join("mnist.dp");
    fs::copy(mnist_example(), &surf_path).expect("copy");

    let deep = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", surf_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&deep_path, deep).expect("write deep program");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["surf", deep_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("let h1").not())
        .stdout(predicate::str::contains("let logits").not())
        .stdout(predicate::str::contains("probs = softmax(logits, 1)"))
        .stdout(predicate::str::contains("out = mean(neg_loss, 0)"))
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", surf_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("(fn {}").or(predicate::str::contains("(fn {} ")));

    // --annotate should thread the inferred type onto the fn node.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["reef", "build", pkg.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(format!(
            "Built chelis-std {}",
            chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION
        )));

    let stem = format!(
        "chelis-std-{}",
        chelis_std_bundle::BUNDLED_CHELIS_STD_VERSION
    );
    assert!(pkg.join("reef.lock").exists());
    assert!(pkg.join(format!("dist/{stem}.chb")).exists());
    assert!(pkg.join(format!("dist/{stem}.tar.zst")).exists());
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "demo-app"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

export (main)

def main(
  x: tensor[32, 784, f32],
  w: tensor[784, 10, f32],
  b: tensor[10, f32]
) -> tensor[32, 10, f32] = {
  bias = insert(&b, 0, shape(&x, cast(0, i32)))
  wx = matmul(&x, &w)
  add(wx, bias)
}
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args([
            "build",
            "--emit-c",
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
#[ignore = "manual gate: Std.Embedding build acceptance exceeds the default inner-loop budget"]
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["reef", "publish", std_pkg.to_str().unwrap()])
        .assert()
        .success();

    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "sig-app"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.4.0" }}
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Io.Safetensors (load_tensors)

export (main)

def main(path: string) -> string =
  load_tensors(path)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        &format!(
            r#"[package]
name = "dep"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Common"
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
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
        &format!(
            r#"[package]
name = "app"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Demo"

[dependencies]
dep = {{ path = "../dep" }}
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        &format!(
            r#"[package]
name = "dep"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Common"
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        type_variable_restrictions: public.type_variable_restrictions,
        collection_obligations: public.collection_obligations,
        effects: public.effects,
        has_body: true,
    });
    module.exports.sort_by(|a, b| a.name.cmp(&b.name));
    write_shell(&shell_path, &shell).expect("rewrite shell");

    write_file(
        &app_pkg.join("reef.toml"),
        &format!(
            r#"[package]
name = "app"
version = "0.1.0"
compiler = "={ver}"
module_prefix = "Demo"

[dependencies]
dep = {{ version = "0.1.0" }}
"#,
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    );
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Common.Api (hidden)

export (main)

def main(x: f32) -> f32 = hidden(x)
"#,
    );

    // Wave-1 red-team M1 (#207 follow-up): `chelis check` now routes
    // `prepare_program_for_file` failures (including reef checksum and
    // missing-export errors) through the JSON `errors[]` array on
    // stdout, so the iff invariant (`exit != 0 iff errors[] non-empty`)
    // holds for every per-file failure mode rather than only the Ok-arm
    // type errors. Look on stdout for the diagnostic string.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .failure()
        .stdout(
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
#[ignore = "experimental HIP semantics are tracked by #2104, not the core release gate"]
fn build_hip_accepts_symbolic_dims_and_binds_them_from_input_metadata() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic.ch");
    let out_dir = dir.path().join("hip-out");
    write_file(
        &path,
        "def f(xs: tensor[batch, features, f32]) -> tensor[batch, features, f32] = xs\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    assert!(source.contains("chelis_device_metadata batch = chelis_tensor_shape(inputs[0], 0);"));
    assert!(
        source.contains("chelis_device_metadata features = chelis_tensor_shape(inputs[0], 1);")
    );
}

#[test]
#[ignore = "experimental HIP semantics are tracked by #2104, not the core release gate"]
fn build_symbolic_matmul_succeeds_on_c_and_hip_targets() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_matmul.ch");
    let c_out = dir.path().join("c-out");
    let hip_out = dir.path().join("hip-out");
    write_symbolic_matmul_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    assert!(c_source.contains("int64_t batch = chelis_tensor_shape(inputs[0], 0);"));
    assert!(c_source.contains("int64_t in_dim = chelis_tensor_shape(inputs[0], 1);"));
    // chelis#1277: the C guard reads BOTH operands from the class's own
    // witnesses instead of comparing against the declared variable, so that a
    // member scoped to one signature is never compared with a variable the
    // occurrence walk declared for another. The property this row names -
    // `in_dim`'s second occurrence is guarded against the axis it was
    // declared from - is unchanged, and the declaration two lines above
    // pins which axis that is. HIP still compares against the variable and
    // its assertion below is unchanged, which is the cross-lane difference
    // this slice records rather than hides.
    assert!(
        c_source.contains("chelis_tensor_shape(inputs[1], 0) != chelis_tensor_shape(inputs[0], 1)")
    );

    let hip_source =
        fs::read_to_string(hip_out.join("symbolic_matmul_hip.cpp")).expect("generated hip");
    assert!(
        hip_source.contains("chelis_device_metadata batch = chelis_tensor_shape(inputs[0], 0);")
    );
    assert!(
        hip_source.contains("chelis_device_metadata in_dim = chelis_tensor_shape(inputs[0], 1);")
    );
    assert!(hip_source.contains("chelis_tensor_shape(inputs[1], 0) != in_dim"));
}

#[test]
#[ignore = "experimental HIP semantics are tracked by #2104, not the core release gate"]
fn build_hip_accepts_symbolic_softmax() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_softmax.ch");
    let out_dir = dir.path().join("hip-out");
    write_symbolic_softmax_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    assert!(source.contains("chelis_device_metadata batch = chelis_tensor_shape(inputs[0], 0);"));
    assert!(source.contains("chelis_device_metadata seq = chelis_tensor_shape(inputs[0], 1);"));
    assert!(source.contains("kernel_maxred_ax1"));
    assert!(source.contains("kernel_sum_ax1"));
}

#[test]
#[ignore = "experimental HIP semantics are tracked by #2104, not the core release gate"]
fn build_hip_accepts_symbolic_row_sum() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_sum.ch");
    let out_dir = dir.path().join("hip-out");
    write_symbolic_row_sum_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    assert!(source.contains("chelis_device_metadata batch = chelis_tensor_shape(inputs[0], 0);"));
    assert!(source.contains("chelis_device_metadata seq = chelis_tensor_shape(inputs[0], 1);"));
    assert!(source.contains("kernel_sum_ax1"));
}

#[test]
#[ignore = "experimental HIP semantics are tracked by #2104, not the core release gate"]
fn build_hip_accepts_symbolic_leading_dims_for_layer_norm() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_layer_norm.ch");
    let out_dir = dir.path().join("hip-out");
    write_symbolic_layer_norm_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    assert!(source.contains("chelis_device_metadata batch = chelis_tensor_shape(inputs[0], 0);"));
    assert!(source.contains("kernel_sum_ax1"));
}

#[test]
fn build_hip_rejects_symbolic_normalized_axis_for_layer_norm() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("symbolic_hidden_layer_norm.ch");
    write_symbolic_hidden_layer_norm_program(&path);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "hip",
        ])
        .assert()
        .failure()
        // Inherited CI unblock: the PR base and current main still expected
        // the pre-949b376e7 wording after production adopted this precise
        // lowering diagnostic.
        .stderr(predicate::str::contains(
            "layer_norm requires a concrete extent for axis 1 in IR lowering",
        ));
}

#[test]
fn build_hip_rejects_pad_host_fallback() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pad.ch");
    let hip_out = dir.path().join("hip-pad-out");
    let c_out = dir.path().join("c-pad-out");
    write_file(
        &path,
        "def f(x: tensor[4, f32]) -> tensor[6, f32] = pad(&x, [[1i64, 1i64]], 0.0)\n",
    );

    let json = run_json_check(&path);
    assert_eq!(json["score"].as_f64(), Some(1.0));
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            c_out.to_str().unwrap(),
        ])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            hip_out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "HIP pad host fallback is unsupported",
        ));
    assert!(!hip_out.join("pad_hip.cpp").exists());
}

#[test]
fn build_hip_ignores_unselected_movement_helpers() {
    let dir = tempdir().expect("tempdir");
    for (name, helper) in [
        (
            "pad",
            "def padder(x: tensor[4, f32]) -> tensor[6, f32] = pad(&x, [[1i64, 1i64]], 0.0)\n",
        ),
        (
            "shrink",
            "def shrinker(x: tensor[6, f32]) -> tensor[4, f32] = shrink(&x, [[1i64, 5i64]])\n",
        ),
    ] {
        let surf_path = dir.path().join(format!("unselected_{name}.ch"));
        let deep_path = dir.path().join(format!("unselected_{name}.dp"));
        write_file(
            &surf_path,
            &format!("{helper}def main(x: tensor[4, f32]) -> tensor[4, f32] = mul(x, x)\n"),
        );
        let deep = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["deep", surf_path.to_str().unwrap()])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        fs::write(&deep_path, deep).expect("write desugared Deep control");

        for (label, path) in [("surf", &surf_path), ("deep", &deep_path)] {
            let out_dir = dir.path().join(format!("unselected_{name}_{label}_hip"));
            Command::cargo_bin("chelis")
                .expect("binary")
                .env("CHELIS_STYLE_GATE_DISABLE", "1")
                .args([
                    "build",
                    "--emit-c",
                    path.to_str().unwrap(),
                    "--target",
                    "hip",
                    "--output",
                    out_dir.to_str().unwrap(),
                ])
                .assert()
                .success();
            let source = fs::read_to_string(out_dir.join(format!("unselected_{name}_hip.cpp")))
                .expect("HIP source");
            assert!(!source.contains("kernel_pad"));
            assert!(!source.contains("kernel_shrink"));
        }
    }
}

#[test]
fn build_hip_rejects_shrink_host_fallback() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("shrink.ch");
    let hip_out = dir.path().join("hip-shrink-out");
    let c_out = dir.path().join("c-shrink-out");
    write_file(
        &path,
        "def f(x: tensor[6, f32]) -> tensor[4, f32] = shrink(&x, [[1i64, 5i64]])\n",
    );

    let json = run_json_check(&path);
    assert_eq!(json["score"].as_f64(), Some(1.0));
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            c_out.to_str().unwrap(),
        ])
        .assert()
        .success();

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            hip_out.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "HIP shrink host fallback is unsupported",
        ));
    assert!(!hip_out.join("shrink_hip.cpp").exists());
}

#[test]
#[ignore = "experimental HIP semantics are tracked by #2104, not the core release gate"]
fn build_hip_emits_pad_kernel() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pad.ch");
    let out_dir = dir.path().join("hip-pad-out");
    // Parameterized form `pad(&x, [[lo, hi]], fill)` per spec §2.4 and
    // issue Chelis-Lang/chelis#187. WS-8A: the HIP backend now lowers pad
    // to a typed per-output-element kernel instead of rejecting it.
    write_file(
        &path,
        "def f(x: tensor[4, f32]) -> tensor[6, f32] = pad(&x, [[1i64, 1i64]], 0.0)\n",
    );

    let json = run_json_check(&path);
    assert_eq!(json["score"].as_f64().unwrap(), 1.0);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let hip_src = fs::read_to_string(out_dir.join("pad_hip.cpp")).expect("hip source");
    assert!(
        hip_src.contains("kernel_pad"),
        "HIP pad build must emit the pad kernel source"
    );
}

#[test]
#[ignore = "experimental HIP semantics are tracked by #2104, not the core release gate"]
fn build_hip_emits_shrink_kernel() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("shrink.ch");
    let out_dir = dir.path().join("hip-shrink-out");
    // WS-8A: shrink now lowers to a typed per-output-element kernel on HIP.
    write_file(
        &path,
        "def f(x: tensor[6, f32]) -> tensor[4, f32] = shrink(&x, [[1i64, 5i64]])\n",
    );

    let json = run_json_check(&path);
    assert_eq!(json["score"].as_f64().unwrap(), 1.0);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let hip_src = fs::read_to_string(out_dir.join("shrink_hip.cpp")).expect("hip source");
    assert!(
        hip_src.contains("kernel_shrink"),
        "HIP shrink build must emit the shrink kernel source"
    );
}

#[test]
fn build_hip_emits_sparse_gather_kernel() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("gather.ch");
    let out_dir = dir.path().join("hip-gather-out");
    write_file(
        &path,
        "def f(table: tensor[1000, 128, f32], indices: tensor[64, i64]) -> tensor[64, 128, f32] = gather(table, indices, 0)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let hip_src = fs::read_to_string(out_dir.join("gather_hip.cpp")).expect("hip source");
    assert!(hip_src.contains("kernel_gather_i64"));
    assert!(hip_src.contains("const long long *indices"));
}

#[test]
fn build_c_emits_sparse_gather_loop_for_int32_indices() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("gather_c.ch");
    let out_dir = dir.path().join("c-gather-out");
    write_file(
        &path,
        "def f(table: tensor[1000, 128, f32], indices: tensor[64, i32]) -> tensor[64, 128, f32] = gather(table, indices, 0)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let c_src = fs::read_to_string(out_dir.join("gather_c.c")).expect("c source");
    assert!(c_src.contains("chelis_tensor_sparse_plan("));
    assert!(c_src.contains("chelis_sparse_index_slot("));
    assert!(c_src.contains("chelis_sparse_data_index("));
    assert!(c_src.contains("chelis_sparse_check_target("));
    assert!(c_src.contains("chelis_sparse_plan_release("));
    assert!(c_src.contains("CHELIS_DTYPE_I32"));
    assert!(!c_src.contains("chelis_tensor_gather("));
    assert!(
        !c_src.contains("(int64_t[]){ 64, 1000, 128 }"),
        "C sparse gather must not allocate the dense [N,V,D] one-hot/product shape"
    );
}

#[test]
fn build_hip_rejects_sparse_gather_with_non_load_cast_indices() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("gather_cast.ch");
    write_file(
        &path,
        "def f(table: tensor[1000, 128, f32], raw: tensor[64, f32]) -> tensor[64, 128, f32] = gather(table, cast(raw, i64), 0)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "hip",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "sparse gather requires indices to be loaded input tensors",
        ))
        .stderr(predicate::str::contains(
            "Non-load integer index producers need integer HIP codegen",
        ));
}

#[test]
fn build_hip_creates_missing_output_directory_and_reports_runtime_path() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("hip_output.ch");
    let out_dir = dir.path().join("nested/hip-output");
    write_file(
        &path,
        "def main(x: tensor[4, f32], y: tensor[4, f32]) -> tensor[4, f32] = \
         relu(add(x, y))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
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

    assert!(out_dir.join("hip_output_hip.cpp").exists());
    assert!(out_dir.join("hip_output_hip.h").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("libchelis_runtime.a").exists());
    assert!(out_dir.join("chelis_hip_runtime.h").exists());
}

#[test]
fn build_hip_admitted_sum_program_emits_fused_kernel_and_launch() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("fused.ch");
    let out_dir = dir.path().join("hip-output");
    write_file(
        &path,
        "def main(x: tensor[3, 4, f32], y: tensor[3, 4, f32]) -> tensor[3, f32] = \
         sum(neg(add(x, y)), 1)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let hip_src = fs::read_to_string(out_dir.join("fused_hip.cpp")).expect("hip source");
    assert!(
        hip_src.contains("kernel_fused_sum_"),
        "an admitted HIP sum build should emit a fused reduction kernel on the real CLI path"
    );
    assert!(
        hip_src.contains("chelis_launch_kernel"),
        "an admitted HIP sum build should emit a kernel launch on the real CLI path"
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
    write_file(
        &source,
        "def matmul_kernel(a: tensor[2, 3, f32], b: tensor[3, 4, f32]) -> tensor[2, 4, f32] = (matmul(a, b) : tensor[2, 4, f32])\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
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
fn build_hip_unbound_observation_root_fails_before_writing_an_artifact() {
    let dir = tempdir().expect("tempdir");
    let out_dir = dir.path().join("hip-output");
    let source = dir.path().join("matmul.ch");
    write_matmul_program(&source);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "hip",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stderr(
            predicate::str::contains("unsupported:")
                .and(predicate::str::contains("[05-UNS-1]"))
                .and(predicate::str::contains("root `a`"))
                .and(predicate::str::contains("codegen:hip"))
                .and(predicate::str::contains("required input(s) `a`")),
        );
    assert!(
        !out_dir.exists(),
        "an unavailable owed root must fail before any partial artifact is written"
    );
}

/// chelis#2413: the key analogue of the retired unhandled-`Random` check. A
/// keyless `dropout` is an arity error at `chelis check`, and the same draw
/// given a key checks clean.
#[test]
fn check_reports_a_keyless_dropout_as_an_arity_error() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dropout.ch");
    write_file(
        &path,
        "x: tensor[32, f32] = x\ny: tensor[32, f32] = dropout(x, 0.5)\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().unwrap();
    assert!(
        errors.iter().any(|error| {
            error["kind"].as_str() == Some("ArityMismatch")
                && error["message"].as_str().is_some_and(|message| {
                    message.contains("`dropout(x, rate)` is the retired counter-stream spelling")
                })
        }),
        "{json}"
    );
    assert!(json["score"].as_f64().unwrap() < 1.0);

    write_file(
        &path,
        "x: tensor[32, f32] = x\ny: tensor[32, f32] = dropout(key_from_seed(1i64), x, 0.5)\n",
    );
    let json = run_json_check(&path);
    assert_eq!(json["errors"], serde_json::json!([]), "{json}");
    assert_eq!(json["score"].as_f64(), Some(1.0), "{json}");
}

#[test]
fn check_reports_linearity_errors() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("linearity.ch");
    write_file(
        &path,
        "def bad(x: tensor[4, f32]) -> tensor[4, f32] = {\n  y: tensor[4, f32] = realize(x)\n  add(x, y)\n}\n",
    );

    let json = run_json_check(&path);
    let errors = json["errors"].as_array().unwrap();
    assert!(errors.iter().any(|error| {
        error["kind"].as_str() == Some("UseAfterConsume")
            && error["message"]
                .as_str()
                .is_some_and(|message| message.contains("realize"))
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
def bad(x: tensor[4, f32]) -> tensor[4, f32] = bad_bool(x)
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
macro dup_relu(x) = add(realize(x), x)
def bad(x: tensor[4, f32]) -> tensor[4, f32] = dup_relu(x)
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
fn check_reports_match_linearity_without_old_ir_rejection() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("match_linearity.ch");
    write_file(
        &path,
        r#"def bad(pair: (tensor[4, f32], i32)) -> i32 = {
  n: i32 = match pair with {
    | (x, _) => 1
  }
  again: (tensor[4, f32], i32) = pair
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
        error["message"]
            .as_str()
            .is_some_and(|message| message.contains("`match` is not supported by IR lowering"))
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
def f(x: tensor[4, f32]) -> tensor[4, f32] = relu_ref(x)
"#,
    );

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
def f(x: tensor[4, f32]) -> tensor[4, f32] = keep(x)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
def f(x: tensor[4, f32]) -> tensor[4, f32] = relu_ref(x)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["surf", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("defmacro"));
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["fmt", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("defmacro"));
}

#[test]
fn build_rejects_gpu_device_region_for_c_target() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("device.ch");
    write_file(&path, "x: i32 = with device(\"gpu:0\") { 1 }\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--emit-c", path.to_str().unwrap(), "--target", "c"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot satisfy resource region"));
}

#[test]
fn tide_quit_exits_cleanly() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["tide", "mcp", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Start the Tide MCP server"));
}

#[test]
fn tide_lsp_help_is_available() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["tide", "lsp", "--help"])
        .assert()
        .success()
        .stdout(predicate::str::contains("Start the Tide LSP server"));
}

#[test]
fn tide_lsp_stdio_flag_is_accepted_and_exits_on_eof() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["tide", "lsp", "--stdio"])
        .assert()
        .success();
}

#[test]
fn cove_help_is_available() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["cove", "--file", "/definitely/missing/chelis-file.ch"])
        .assert()
        .failure()
        .stderr(predicate::str::contains("read Cove file"));
}

#[test]
fn cove_requires_interactive_terminal_for_valid_launch() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
    let configuration =
        fs::read_to_string(editor_file("language-configuration.json")).expect("language config");
    let configuration: Value =
        serde_json::from_str(&configuration).expect("valid language config json");
    assert_eq!(configuration["comments"]["lineComment"], "--");
    assert_eq!(
        configuration["comments"]["blockComment"],
        serde_json::json!(["{-", "-}"])
    );

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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", hello_tensor_example().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&deep_path, output).expect("write deep output");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", "--flat", hello_tensor_example().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&deep_path, output).expect("write deep output");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["validate", "--deep", deep_path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated deep:"));
}

#[test]
fn deep_defaults_to_pretty_output_and_flat_flag_preserves_flat_per_form_rendering() {
    let pretty = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", surf_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&deep_path, output).expect("write deep output");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["validate", "--surf", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("validation failed"));
}

#[test]
fn validate_surf_accepts_newline_block_and_axis_identifier() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("ok.ch");
    write_file(
        &path,
        "def f(axis) = {\n  y = axis\n  y\n}\ndef g() = par { a; b }\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["validate", "--desugar", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("validated desugar:"));
}

#[test]
fn validate_deep_rejects_unknown_tag() {
    // chelis#1088: the fixture moved inside a declaration. A top-level
    // `(mystery {} x)` is now a [03-PROG-1] rejection, so leaving it at top
    // level would make this test pass for a reason unrelated to the unknown
    // head it is about.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad.dp");
    write_file(&path, "(def {} f (mystery {} x))\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("unknown tag"))
        .stderr(predicate::str::contains("mystery"));
}

#[test]
fn validate_deep_identifies_a_headless_top_level_form_like_check_does() {
    // chelis#1088: `validate --deep` used to run its Pest grammar first, so a
    // headless top-level form died as `expected program` and the reader never
    // learned its [03-PROG-2] class. The stamped ingress now decides first.
    let dir = tempdir().expect("tempdir");
    for (name, source, identification) in [
        ("bare_int.dp", "42\n", "a bare integer literal"),
        (
            "untagged.dp",
            "((var {} f) (var {} x))\n",
            "a list without a tag symbol",
        ),
        ("empty.dp", "", "empty program"),
        ("comments.dp", "; only a comment\n", "empty program"),
    ] {
        let path = dir.path().join(name);
        write_file(&path, source);
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["validate", "--deep", path.to_str().unwrap()])
            .assert()
            .failure()
            .stderr(predicate::str::contains(identification));
    }
}

#[test]
fn validate_deep_rejects_invalid_effects_children() {
    // Effect payloads are validated at shared metadata ingress before the
    // executable grammar or semantic effect consumers run.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad_effects.dp");
    write_file(
        &path,
        "(defsig {} f (t-fn {eff: (effects {} 1)} (t-prim {} f32)))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("metadata `eff`"));
}

#[test]
fn validate_deep_rejects_invalid_resource_arity() {
    // chelis#1088: likewise nested where a `resource` entry really appears.
    // Shared metadata admission owns this nested shape rejection.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bad_resource.dp");
    write_file(
        &path,
        "(defsig {} f (t-fn {eff: (effects {} (resource {} \"x\" \"y\"))} (t-prim {} f32)))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["validate", "--deep", path.to_str().unwrap()])
        .assert()
        .failure()
        .stderr(predicate::str::contains("resource"))
        .stderr(predicate::str::contains("metadata `eff`"));
}

#[test]
fn deep_build_rejects_selected_orphan_signature_before_pruning() {
    let dir = tempdir().expect("tempdir");
    let valid = dir.path().join("paired.dp");
    let invalid = dir.path().join("orphan.dp");
    let paired = "(defsig {} present (t-fn {} (t-prim {} string)))\n(def {} present (fn {} (params {}) (lit {type: (t-prim {} string)} \"ok\")))\n";
    write_file(&valid, paired);
    write_file(
        &invalid,
        &format!("{paired}(defsig {{}} missing (t-fn {{}} (t-prim {{}} string)))\n"),
    );

    let check = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", invalid.to_str().unwrap()])
        .output()
        .expect("check Deep program");
    let check_report = format!(
        "{}{}",
        String::from_utf8_lossy(&check.stdout),
        String::from_utf8_lossy(&check.stderr)
    );
    assert!(!check.status.success(), "{check_report}");
    assert!(check_report.contains("UnboundVariable"), "{check_report}");

    for target in ["c", "hip", "metal"] {
        Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                "--emit-c",
                valid.to_str().unwrap(),
                "--target",
                target,
                "-o",
                dir.path().join(format!("valid-{target}")).to_str().unwrap(),
            ])
            .assert()
            .success();

        let build = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args([
                "build",
                "--emit-c",
                invalid.to_str().unwrap(),
                "--target",
                target,
                "-o",
                dir.path()
                    .join(format!("invalid-{target}"))
                    .to_str()
                    .unwrap(),
            ])
            .output()
            .expect("build Deep program");
        let error = String::from_utf8_lossy(&build.stderr);
        assert!(
            !build.status.success(),
            "{target} accepted orphan defsig: {error}"
        );
        assert!(error.contains("UnboundVariable"), "{target}: {error}");
    }
}

#[test]
fn reef_book_workflow_commands_are_valid() {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join("demo");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["reef", "build", pkg.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("Built demo 0.1.0"));
}

// ===========================================================================
// Phase M (Metal backend) — M1 CLI dispatch tests.
//
// These verify that `--target metal` is wired through main.rs alongside
// `--target c` and `--target hip`. M2 will add structural assertions on the
// emitted .mm; M6 owns runtime correctness on Apple Silicon.
// ===========================================================================

#[test]
fn target_metal_emits_mm_header_and_runtime_artifacts() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("simple_add.ch");
    let out_dir = dir.path().join("metal-output");
    write_file(
        &src,
        "def simple_add(a: tensor[4, f32], b: tensor[4, f32]) -> tensor[4, f32] = add(a, b)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("simple_add_metal.mm"))
        .stdout(predicate::str::contains("chelis_metal_runtime.h"))
        .stdout(predicate::str::contains(
            "Apple Silicon unified memory means this is also peak system RAM",
        ))
        .stdout(predicate::str::contains("clang++"))
        .stdout(predicate::str::contains("-framework Metal"))
        .stdout(predicate::str::contains("-framework Foundation"));

    let mm_src = fs::read_to_string(out_dir.join("simple_add_metal.mm")).expect("metal source");
    assert!(
        mm_src.contains("extern \"C\" void simple_add("),
        "metal .mm should declare the C-ABI entry point, got:\n{mm_src}"
    );
    assert!(
        mm_src.contains("#import \"chelis_metal_runtime.h\""),
        "metal .mm should import the Metal runtime header, got:\n{mm_src}"
    );

    let header = fs::read_to_string(out_dir.join("simple_add_metal.h")).expect("metal header");
    assert!(header.contains("extern \"C\" void simple_add("));

    assert!(out_dir.join("chelis_metal_runtime.h").exists());
    assert!(out_dir.join("chelis_runtime.h").exists());
    assert!(out_dir.join("libchelis_runtime.a").exists());

    // `chelis_hip_runtime.h` must NOT leak into a metal build directory.
    assert!(!out_dir.join("chelis_hip_runtime.h").exists());
}

#[test]
fn target_metal_unknown_target_message_lists_metal() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("hello.ch");
    write_file(
        &path,
        "def hello(a: tensor[2, f32], b: tensor[2, f32]) -> tensor[2, f32] = add(a, b)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "vulkan",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "unknown target 'vulkan': expected 'c', 'hip', or 'metal'",
        ));
}

#[test]
fn target_metal_emits_pad_kernel() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pad.ch");
    let out_dir = dir.path().join("out");
    // Parameterized pad per spec §2.4 and issue Chelis-Lang/chelis#187.
    // WS-8A: the Metal backend now lowers pad to a typed MSL kernel
    // instead of rejecting it.
    write_file(
        &path,
        "def f(x: tensor[4, f32]) -> tensor[6, f32] = pad(&x, [[1i64, 1i64]], 0.0)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let mm_src = fs::read_to_string(out_dir.join("pad_metal.mm")).expect("metal source");
    assert!(
        mm_src.contains("k_pad"),
        "Metal pad build must emit the pad kernel: {mm_src}"
    );
    assert!(
        mm_src.contains("chelis_metal_launch_two_uniforms"),
        "Metal pad must dispatch with the dims + fill uniform pair"
    );
}

#[test]
fn target_metal_emits_shrink_kernel() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("shrink.ch");
    let out_dir = dir.path().join("out");
    // WS-8A: the Metal backend now lowers shrink to a typed MSL kernel.
    write_file(
        &path,
        "def f(x: tensor[4, f32]) -> tensor[2, f32] = shrink(&x, [[1i64, 3i64]])\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let mm_src = fs::read_to_string(out_dir.join("shrink_metal.mm")).expect("metal source");
    assert!(
        mm_src.contains("k_shrink"),
        "Metal shrink build must emit the shrink kernel: {mm_src}"
    );
}

#[test]
fn target_metal_rejects_f64_precision() {
    // WS-M1 widens the active Metal dtype set to all spec dtypes
    // except f64 (spec/04-type-system.md §1.1.3). The f64 rejection
    // diagnostic is now the spec-pinned FP64-ALU message; tests must
    // assert exact-string-match so the contract doesn't drift.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("f64_one.ch");
    write_file(&path, "def f64_one() -> f64 = cast(1.0, f64)\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["build", "--emit-c", path.to_str().unwrap(), "--target", "metal"])
        .assert()
        .failure()
        .stderr(predicate::str::contains(
            "Apple Silicon GPUs lack FP64 ALUs; use `--target c` or `--target hip` for f64 workloads",
        ));
}

#[test]
fn target_metal_admits_f16_add() {
    // WS-M1: f16 is in the active Metal dtype set
    // (spec/04-type-system.md §1.1.3). A simple f16 add must build
    // through `chelis build --target metal` without the CLI gate
    // rejecting on precision grounds.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("f16_add.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &path,
        "def f16_add(x: tensor[3, f16], y: tensor[3, f16]) -> tensor[3, f16] = add(x, y)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let mm_src = fs::read_to_string(out_dir.join("f16_add_metal.mm")).expect("metal source");
    assert!(
        mm_src.contains("device const half*"),
        "f16 add must emit `device const half*` operands: {mm_src}"
    );
}

#[test]
fn target_metal_admits_bf16_add() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bf16_add.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &path,
        "def bf16_add(x: tensor[3, bf16], y: tensor[3, bf16]) -> tensor[3, bf16] = add(x, y)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let mm_src = fs::read_to_string(out_dir.join("bf16_add_metal.mm")).expect("metal source");
    assert!(
        mm_src.contains("#if __METAL_VERSION__ >= 320"),
        "bf16 add must wrap the kernel in the MSL 3.2+ guard: {mm_src}"
    );
    assert!(
        mm_src.contains("device const bfloat*"),
        "bf16 add must emit `device const bfloat*` operands: {mm_src}"
    );
}

#[test]
fn target_metal_admits_i32_add() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("i32_add.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &path,
        "def i32_add(x: tensor[3, i32], y: tensor[3, i32]) -> tensor[3, i32] = add(x, y)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let mm_src = fs::read_to_string(out_dir.join("i32_add_metal.mm")).expect("metal source");
    assert!(
        mm_src.contains("device const int*"),
        "i32 add must emit `device const int*` operands: {mm_src}"
    );
}

#[test]
fn target_metal_admits_i64_add() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("i64_add.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &path,
        "def i64_add(x: tensor[3, i64], y: tensor[3, i64]) -> tensor[3, i64] = add(x, y)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let mm_src = fs::read_to_string(out_dir.join("i64_add_metal.mm")).expect("metal source");
    assert!(
        mm_src.contains("device const long*"),
        "i64 add must emit `device const long*` operands: {mm_src}"
    );
}

#[test]
fn target_metal_link_line_includes_metal_performance_shaders() {
    // WS-M1 added MPSMatrixMultiplication for f32/f16 matmul; the
    // emitted build recipe must include `MetalPerformanceShaders`.
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("mm.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &src,
        "def mm(a: tensor[4, 4, f32], b: tensor[4, 4, f32]) -> tensor[4, 4, f32] = matmul(a, b)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "-framework MetalPerformanceShaders",
        ));
    let mm_src = fs::read_to_string(out_dir.join("mm_metal.mm")).expect("metal source");
    assert!(
        mm_src.contains("chelis_metal_mps_gemm_f32("),
        "f32 matmul must dispatch to MPS via `chelis_metal_mps_gemm_f32`: {mm_src}"
    );
}

#[test]
fn target_metal_rejects_cpu_resource_region() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("cpu_region.ch");
    write_file(&path, "x: i32 = with device(\"cpu\") { 1 }\n");

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "metal",
        ])
        .assert()
        .failure()
        .stderr(predicate::str::contains("cannot satisfy resource region"));
}

// =============================================================================
// Bucket 3: activation primitive parity (relu, sigmoid, tanh, silu, gelu).
//
// These tests close the IR-evaluator/C-backend gap surfaced as
// `unsupported builtin \`relu\` in host runtime` (and siblings). For each
// activation we run a small program through both `chelis eval` (the
// in-process IR evaluator dispatched in `chelis-compiler-api/src/runtime/eval.rs`)
// and `chelis build --target c` (whose generated code uses the
// `chelis_host_*_f32` helpers in `chelis-backend-c/src/host_emit.rs`).
//
// Verification strategy: each program embeds `test_assert_close_tensor`
// against an offline-computed expected tensor with `1e-4` tolerance.
// Both lanes succeed (exit 0, no panic) iff their numerics agree to that
// tolerance. The C lane also runs the compiled binary to exercise the
// host-emit f32 helper end-to-end.
// =============================================================================

/// Run a Bucket 3 activation program through both lanes:
///   1. `chelis eval --file <path>` (IR evaluator, runtime.rs)
///   2. `chelis build --target c --output <out_dir>` followed by gcc + run
///
/// Returns the eval stdout and run stdout for any caller that wants to
/// do additional shape comparison. The function asserts each step
/// succeeds; failures bubble up with the lane name in the panic.
///
/// chelis#730 Phase 1 re-author: the embedded `test_assert_close_tensor`
/// check is EVAL-LANE ONLY. It always was - the compiled lane used to
/// build it as the silent `/* unsupported builtin */ 0` stub, so the
/// binary asserted nothing while this harness claimed both lanes were
/// verified. The stub arm is now a loud build rejection, so the C lane
/// builds a stripped copy of the program (the assert line removed) and
/// its verification is the compiled forward computation running to exit
/// 0; a compiled-lane arm for the test_* builtins is op-owner support
/// work (chelis#703 class).
fn run_activation_parity(name: &str, source_body: &str) -> (Vec<u8>, Vec<u8>) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let build_path = dir.path().join(format!("{name}_c.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, source_body);
    let stripped: String = source_body
        .lines()
        .filter(|line| {
            !line
                .trim_start()
                .starts_with("ok = test_assert_close_tensor")
        })
        .collect::<Vec<_>>()
        .join("\n");
    write_file(&build_path, &stripped);

    // `chelis check` must pass cleanly.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("\"score\": 1"));

    // IR-evaluator lane.
    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();

    // C-backend lane: build the STRIPPED copy (see the doc comment), then
    // compile and run.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            build_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source_file = format!("{name}_c.c");
    let status = gcc_link_generated(&out_dir, &source_file, name);
    assert!(
        status.success(),
        "{name}: gcc compile/link failed with status {status}"
    );
    let run_output = StdCommand::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "{name}: compiled binary failed with status {} stderr:\n{}",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr)
    );
    (eval_stdout, run_output.stdout)
}

#[test]
fn bucket3_relu_runs_in_eval_and_c_lanes() {
    // relu(x) = max(0, x). Exact in any precision.
    run_activation_parity(
        "bucket3_relu",
        r#"
def relu_apply(x: tensor[5, f32]) -> tensor[5, f32] = relu(x)

input = to_tensor([cast(1.0, f32), cast(-2.0, f32), cast(0.0, f32), cast(3.5, f32), cast(-0.5, f32)])
actual = relu_apply(input)
expected = to_tensor([cast(1.0, f32), cast(0.0, f32), cast(0.0, f32), cast(3.5, f32), cast(0.0, f32)])
ok = test_assert_close_tensor(actual, expected, 1e-6, "relu pointwise")
"#,
    );
}

#[test]
fn bucket3_relu_eval_negative_rejects_unequal_input() {
    // Negative test: a deliberately-wrong expected tensor should make
    // `chelis eval` fail, locking the test_assert_close_tensor invariant.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bucket3_relu_neg.ch");
    write_file(
        &path,
        r#"
def relu_apply(x: tensor[3, f32]) -> tensor[3, f32] = relu(x)

input = to_tensor([cast(1.0, f32), cast(-2.0, f32), cast(3.0, f32)])
actual = relu_apply(input)
wrong = to_tensor([cast(1.0, f32), cast(99.0, f32), cast(3.0, f32)])
ok = test_assert_close_tensor(actual, wrong, 1e-6, "relu wrong")
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .failure();
}

#[test]
fn bucket3_sigmoid_runs_in_eval_and_c_lanes() {
    // sigmoid(0) = 0.5; sigmoid(1) ≈ 0.73105858; sigmoid(-1) ≈ 0.26894142.
    // Both lanes go through `expf`-precision math (host-runtime mirrors
    // chelis_host_sigmoid_f32 byte-for-byte modulo 1e-6 ulp).
    run_activation_parity(
        "bucket3_sigmoid",
        r#"
def sig_apply(x: tensor[3, f32]) -> tensor[3, f32] = sigmoid(x)

input = to_tensor([cast(0.0, f32), cast(1.0, f32), cast(-1.0, f32)])
actual = sig_apply(input)
expected = to_tensor([cast(0.5, f32), cast(0.7310586, f32), cast(0.26894143, f32)])
ok = test_assert_close_tensor(actual, expected, 0.0001, "sigmoid pointwise")
"#,
    );
}

#[test]
fn bucket3_tanh_runs_in_eval_and_c_lanes() {
    // tanh(0) = 0; tanh(1) ≈ 0.76159418; tanh(-1) ≈ -0.76159418.
    run_activation_parity(
        "bucket3_tanh",
        r#"
def tanh_apply(x: tensor[3, f32]) -> tensor[3, f32] = tanh(x)

input = to_tensor([cast(0.0, f32), cast(1.0, f32), cast(-1.0, f32)])
actual = tanh_apply(input)
expected = to_tensor([cast(0.0, f32), cast(0.76159418, f32), cast(-0.76159418, f32)])
ok = test_assert_close_tensor(actual, expected, 0.0001, "tanh pointwise")
"#,
    );
}

#[test]
fn bucket3_silu_runs_in_eval_and_c_lanes() {
    // silu(x) = x * sigmoid(x).
    // silu(0) = 0; silu(1) ≈ 0.73105858; silu(-1) ≈ -0.26894142.
    run_activation_parity(
        "bucket3_silu",
        r#"
def silu_apply(x: tensor[3, f32]) -> tensor[3, f32] = silu(x)

input = to_tensor([cast(0.0, f32), cast(1.0, f32), cast(-1.0, f32)])
actual = silu_apply(input)
expected = to_tensor([cast(0.0, f32), cast(0.7310586, f32), cast(-0.26894143, f32)])
ok = test_assert_close_tensor(actual, expected, 0.0001, "silu pointwise")
"#,
    );
}

#[test]
fn bucket3_gelu_tanh_approx_runs_in_eval_and_c_lanes() {
    // gelu(0) = 0; gelu(1) ≈ 0.84119; gelu(-1) ≈ -0.15881.
    // Tanh approximation matches the host-runtime `activation_gelu_f32` helper.
    run_activation_parity(
        "bucket3_gelu",
        r#"
def gelu_apply(x: tensor[3, f32]) -> tensor[3, f32] = gelu(x)

input = to_tensor([cast(0.0, f32), cast(1.0, f32), cast(-1.0, f32)])
actual = gelu_apply(input)
expected = to_tensor([cast(0.0, f32), cast(0.84119, f32), cast(-0.15881, f32)])
ok = test_assert_close_tensor(actual, expected, 0.0001, "gelu pointwise (tanh-approx)")
"#,
    );
}

// Bucket 3b: float-unary primitive parity for `sqrt`/`log`/`exp`/`sin`
// on tensor args. Closes the gap surfaced as `float op expects float
// arg, got Some(Tensor(...))` in #142: the eval-lane dispatcher used
// `float_unop` (scalar-only) for these four builtins, while the
// C-backend `host_emit.rs` already emitted `sqrtf`/`logf`/`expf`/`sinf`
// elementwise. The eval lane now routes tensor args through
// `tensor_float_unop_f32` so both lanes stay byte-identical (to f32 ulp
// tolerance), mirroring the activation block above.

#[test]
fn bucket3b_sqrt_tensor_runs_in_eval_and_c_lanes() {
    // sqrt is exact for these perfect squares in any precision.
    run_activation_parity(
        "bucket3b_sqrt",
        r#"
def sqrt_apply(x: tensor[3, f32]) -> tensor[3, f32] = sqrt(x)

input = to_tensor([cast(4.0, f32), cast(9.0, f32), cast(16.0, f32)])
actual = sqrt_apply(input)
expected = to_tensor([cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)])
ok = test_assert_close_tensor(actual, expected, 1e-6, "sqrt pointwise")
"#,
    );
}

#[test]
fn bucket3b_log_tensor_runs_in_eval_and_c_lanes() {
    // log(1) = 0 exactly; log(e) ≈ 1; log(e^2) ≈ 2.
    run_activation_parity(
        "bucket3b_log",
        r#"
def log_apply(x: tensor[3, f32]) -> tensor[3, f32] = log(x)

input = to_tensor([cast(1.0, f32), cast(2.7182817, f32), cast(7.389056, f32)])
actual = log_apply(input)
expected = to_tensor([cast(0.0, f32), cast(1.0, f32), cast(2.0, f32)])
ok = test_assert_close_tensor(actual, expected, 0.0001, "log pointwise")
"#,
    );
}

#[test]
fn bucket3b_exp_tensor_runs_in_eval_and_c_lanes() {
    // exp(0) = 1 exactly; exp(1) ≈ e; exp(-1) ≈ 1/e.
    run_activation_parity(
        "bucket3b_exp",
        r#"
def exp_apply(x: tensor[3, f32]) -> tensor[3, f32] = exp(x)

input = to_tensor([cast(0.0, f32), cast(1.0, f32), cast(-1.0, f32)])
actual = exp_apply(input)
expected = to_tensor([cast(1.0, f32), cast(2.7182817, f32), cast(0.36787945, f32)])
ok = test_assert_close_tensor(actual, expected, 0.0001, "exp pointwise")
"#,
    );
}

#[test]
fn bucket3b_sin_tensor_runs_in_eval_and_c_lanes() {
    // sin(0) = 0; sin(pi/2) = 1; sin(pi) ~= 0 (to f32 tolerance).
    run_activation_parity(
        "bucket3b_sin",
        r#"
def sin_apply(x: tensor[3, f32]) -> tensor[3, f32] = sin(x)

input = to_tensor([cast(0.0, f32), cast(1.5707964, f32), cast(3.1415927, f32)])
actual = sin_apply(input)
expected = to_tensor([cast(0.0, f32), cast(1.0, f32), cast(0.0, f32)])
ok = test_assert_close_tensor(actual, expected, 0.0001, "sin pointwise")
"#,
    );
}

#[test]
fn bucket3b_sqrt_scalar_still_works_in_eval() {
    // Negative-parity guard: the fix must not regress scalar dispatch.
    // sqrt(16.0) = 4.0 exactly.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bucket3b_sqrt_scalar.ch");
    write_file(
        &path,
        "module Repro\nexport (result)\nresult = sqrt(cast(16.0, f32))\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("4"));
}

#[test]
fn target_metal_accepts_gpu_resource_region() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("gpu_region.ch");
    let out_dir = dir.path().join("metal-output");
    // Same form Phase 2a uses to verify HIP accepts gpu device regions.
    // Use distinct args to satisfy linearity (`a` consumed once).
    write_file(
        &path,
        "def f(a: tensor[4, f32], b: tensor[4, f32]) -> tensor[4, f32] = with device(\"gpu:0\") { add(a, b) }\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "metal",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
}

/// Bucket 6b: `chelis check <dir>` walks the directory tree, runs the
/// per-file fitness pass, and emits one aggregated JSON record so callers
/// (CI, IDEs) can lint a corpus without scripting a fan-out themselves.
#[test]
fn check_directory_walks_ch_files_and_aggregates_json() {
    let dir = tempdir().expect("tempdir");
    write_file(&dir.path().join("a.ch"), "def main() -> i32 = 0\n");
    fs::create_dir_all(dir.path().join("nested")).expect("mkdir nested");
    write_file(
        &dir.path().join("nested").join("b.ch"),
        "def main() -> i32 = 1\n",
    );
    // dot-prefixed file should be skipped by the walker
    write_file(
        &dir.path().join(".scratch.ch"),
        "garbage that would fail to parse\n",
    );
    // non-.ch files are skipped
    write_file(&dir.path().join("README.md"), "# not chelis\n");

    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", dir.path().to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: Value =
        serde_json::from_str(&stdout).expect("check directory output must be valid JSON");
    let files = parsed
        .get("files")
        .and_then(|v| v.as_array())
        .expect("files array must exist");
    let names: Vec<String> = files
        .iter()
        .filter_map(|entry| {
            entry
                .get("file")
                .and_then(|v| v.as_str())
                .map(|s| s.to_string())
        })
        .collect();
    assert!(
        names.iter().any(|n| n.ends_with("a.ch")),
        "expected a.ch in {names:?}"
    );
    assert!(
        names.iter().any(|n| n.ends_with("b.ch")),
        "expected nested/b.ch in {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.contains(".scratch.ch")),
        "dot-prefixed file should be skipped: {names:?}"
    );
    assert!(
        !names.iter().any(|n| n.ends_with("README.md")),
        "non-.ch files must be skipped: {names:?}"
    );
}

#[test]
fn lint_list_reports_registered_rule_severities() {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--list"])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("opaque-without-invariant\tadvisory").and(
                predicate::str::contains("no-em-dash-in-public-strings\terror"),
            ),
        );
}

/// chelis#3130: no lint rule converts between Surf spellings. A file that
/// `prefer-pipe-operator`, `redundant-linearity-call` and
/// `prefer-typed-literal` each reported, with a `[fix]`, draws none of them
/// from `lint`, `lint --fix` or `check`, and none of the ids is selectable.
#[test]
fn lint_has_no_spelling_conversion_rules() {
    const REMOVED: [&str; 3] = [
        "prefer-pipe-operator",
        "redundant-linearity-call",
        "prefer-typed-literal",
    ];
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("spellings.ch");
    let original = "def f(x: tensor[2, f32]) -> tensor[2, f32] = relu(neg(x))\n\
                    def g(x: tensor[2, f32]) -> tensor[2, f32] = realize(copy(x))\n\
                    y = cast(1.0, f64)\n";
    write_file(&path, original);

    let listed = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--list"])
        .output()
        .expect("run lint --list");
    assert!(listed.status.success(), "lint --list failed: {listed:?}");
    let listed = String::from_utf8(listed.stdout).expect("utf-8 stdout");

    let linted = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--check", path.to_str().unwrap()])
        .output()
        .expect("run lint --check");
    assert!(linted.status.success(), "lint --check failed: {linted:?}");
    let linted = String::from_utf8(linted.stdout).expect("utf-8 stdout");
    assert!(!linted.contains("[fix]"), "no fix is offered: {linted}");

    let fixed = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--fix", path.to_str().unwrap()])
        .output()
        .expect("run lint --fix");
    assert!(fixed.status.success(), "lint --fix failed: {fixed:?}");
    assert_eq!(
        fs::read_to_string(&path).expect("read fixture"),
        original,
        "lint --fix leaves every spelling as written"
    );

    let checked = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run check");
    assert!(checked.status.success(), "check failed: {checked:?}");
    let checked = String::from_utf8(checked.stderr).expect("utf-8 stderr");

    for id in REMOVED {
        assert!(!listed.contains(id), "`{id}` is not registered: {listed}");
        assert!(!linted.contains(id), "`{id}` reports nothing: {linted}");
        assert!(
            !checked.contains(id),
            "`check` prints no `{id}` warning: {checked}"
        );
        Command::cargo_bin("chelis")
            .expect("binary")
            .args(["lint", "--rule", id, path.to_str().unwrap()])
            .assert()
            .failure()
            .stderr(predicate::str::contains(format!("no rule with id '{id}'")));
    }
}

#[test]
fn lint_rules_filter_runs_selected_rules_only() {
    let dir = tempdir().expect("tempdir");
    write_file(
        &dir.path().join("meters.ch"),
        "module Units\n@opaque\ntype Meters =\n  | Meters { value: f64 }\ndef make(v: f64) -> Meters = Meters { value: v }\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--rules",
            "recursive-list-cursor",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("opaque-without-invariant").not());

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--rules",
            "opaque-without-invariant",
            dir.path().to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("opaque-without-invariant"));
}

#[test]
fn lint_keep_preserves_no_em_dash_fix_but_still_reports_error() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("keep_dash.rs");
    let dash = '\u{2014}';
    let original = format!(
        "fn main() {{ println!(\"one {dash} two\"); }} // chelis-lint: keep no-em-dash-in-public-strings\n"
    );
    write_file(&path, &original);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--fix", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("no-em-dash-in-public-strings")
                .and(predicate::str::contains("[fix]").not()),
        );

    let rewritten = fs::read_to_string(path).expect("read rewritten");
    assert_eq!(rewritten, original);
}

#[test]
fn lint_allow_suppresses_surf_diagnostic() {
    // `//` is not a Surf comment (#2853), so a directive makes the file
    // unparseable; the lint still reads it and suppresses the named rule.
    let dir = tempdir().expect("tempdir");
    let reported = dir.path().join("reported.ch");
    let allowed = dir.path().join("allowed.ch");
    write_file(&reported, "def bad_Name(x: f32) -> f32 = x\n");
    write_file(
        &allowed,
        "def bad_Name(x: f32) -> f32 = x // chelis-lint: allow surf-value-snake-case\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--rule",
            "surf-value-snake-case",
            reported.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("surf-value-snake-case"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--fix",
            "--rule",
            "surf-value-snake-case",
            allowed.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("surf-value-snake-case").not());
}

#[test]
fn lint_no_em_dash_blocks_check_and_can_fix_clause_case() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("message.rs");
    let dash = '\u{2014}';
    write_file(
        &path,
        &format!("fn main() {{ println!(\"one {dash} two\"); }}\n"),
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--check", path.to_str().unwrap()])
        .assert()
        .failure()
        .stdout(predicate::str::contains("no-em-dash-in-public-strings"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--fix", path.to_str().unwrap()])
        .assert()
        .success();

    let rewritten = fs::read_to_string(path).expect("read rewritten");
    assert!(rewritten.contains("\"one. Two\""));
}

// Guard 2 (test toolchain footgun guards): a blocking lint ERROR must
// be surfaced distinctly, not buried in advisory-warning noise. The
// em-dash §8.6 rule was bitten twice by CI failure debugging that
// misread a buried blocking error. `cmd_lint` now buckets output by
// severity: advisory/warning lines print first, then blocking errors
// last under a delimited header plus a summary line. See
// docs/investigations/test_toolchain_guards_design.md.
#[test]
fn lint_check_surfaces_blocking_error_below_advisory_noise() {
    let dir = tempdir().expect("tempdir");
    let dash = '\u{2014}';
    // A Surf file that triggers a non-blocking advisory
    // (`opaque-without-invariant`). Pair it with a Rust file carrying a
    // blocking em-dash error.
    write_file(
        &dir.path().join("meters.ch"),
        "module Units\n@opaque\ntype Meters =\n  | Meters { value: f64 }\ndef make(v: f64) -> Meters = Meters { value: v }\n",
    );
    write_file(
        &dir.path().join("message.rs"),
        &format!("fn main() {{ println!(\"one {dash} two\"); }}\n"),
    );

    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["lint", "--check", dir.path().to_str().unwrap()])
        .assert()
        .failure();
    let stdout = String::from_utf8(assert.get_output().stdout.clone()).expect("utf8");

    // The blocking-error section header is present, and the em-dash
    // error line appears inside it.
    assert!(
        stdout.contains("blocking lint error(s)"),
        "expected a blocking-error section header, got:\n{stdout}"
    );
    assert!(
        stdout.contains("lint --check failed:"),
        "expected a blocking-error summary line, got:\n{stdout}"
    );
    let header_pos = stdout
        .find("blocking lint error(s)")
        .expect("header present");
    let error_pos = stdout
        .find("no-em-dash-in-public-strings")
        .expect("em-dash error present");
    assert!(
        error_pos > header_pos,
        "the blocking em-dash error must print after the section header, got:\n{stdout}"
    );
    // The advisory line must come before the header (the buckets print
    // advisory/warning first, errors last).
    let advisory_pos = stdout
        .find("advisory:")
        .expect("opaque-without-invariant advisory present");
    assert!(
        advisory_pos < header_pos,
        "advisory/warning lines must print before the blocking-error \
         section, got:\n{stdout}"
    );
}

#[test]
fn lint_no_em_dash_does_not_corrupt_unspaced_dash() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("message.rs");
    let dash = '\u{2014}';
    let original =
        format!("fn main() {{ println!(\"{dash}two\"); println!(\"one{dash}two\"); }}\n");
    write_file(&path, &original);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--fix",
            "--rule",
            "no-em-dash-in-public-strings",
            path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::contains("no-em-dash-in-public-strings"));

    let rewritten = fs::read_to_string(path).expect("read rewritten");
    assert_eq!(rewritten, original);
}

#[test]
fn lint_allow_directive_must_be_a_real_directive() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("allow_string.rs");
    let dash = '\u{2014}';
    write_file(
        &path,
        &format!(
            "fn main() {{ println!(\"#[allow(no-em-dash-in-public-strings)]\"); println!(\"one {dash} two\"); }}\n"
        ),
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--check",
            "--rule",
            "no-em-dash-in-public-strings",
            path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stdout(predicate::str::contains("no-em-dash-in-public-strings"));
}

#[test]
fn lint_deep_semicolon_allow_suppresses_diagnostic() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("allow.dp");
    write_file(
        &path,
        "; chelis-lint: allow deep-user-symbol-charset\n(def {} my-func (lit {} 1))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--check",
            "--rule",
            "deep-user-symbol-charset",
            path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());
}

#[test]
fn lint_semicolon_directive_does_not_suppress_rust_diagnostic() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("fake_semicolon.rs");
    let dash = '\u{2014}';
    write_file(
        &path,
        &format!(
            "; chelis-lint: allow no-em-dash-in-public-strings\nfn main() {{ println!(\"one {dash} two\"); }}\n"
        ),
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--check",
            "--rule",
            "no-em-dash-in-public-strings",
            path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stdout(predicate::str::contains("no-em-dash-in-public-strings"));
}

#[test]
fn lint_no_em_dash_ignores_quoted_text_in_rust_comments() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("comment.rs");
    let dash = '\u{2014}';
    let original = format!("// diagnostic example: println!(\"one {dash} two\");\n");
    write_file(&path, &original);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--fix",
            "--rule",
            "no-em-dash-in-public-strings",
            path.to_str().unwrap(),
        ])
        .assert()
        .success()
        .stdout(predicate::str::is_empty());

    let rewritten = fs::read_to_string(path).expect("read rewritten");
    assert_eq!(rewritten, original);
}

#[test]
fn lint_no_em_dash_flags_python_single_quoted_strings() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("message.py");
    let dash = '\u{2014}';
    write_file(&path, &format!("print('one {dash} two')\n"));

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "lint",
            "--check",
            "--rule",
            "no-em-dash-in-public-strings",
            path.to_str().unwrap(),
        ])
        .assert()
        .failure()
        .stdout(predicate::str::contains("no-em-dash-in-public-strings"));
}

#[test]
fn check_relative_file_emits_non_blocking_lint_warning() {
    let dir = tempdir().expect("tempdir");
    write_file(
        &dir.path().join("meters.ch"),
        "module Units\n@opaque\ntype Meters =\n  | Meters { value: f64 }\ndef make(v: f64) -> Meters = Meters { value: v }\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .args(["check", "meters.ch"])
        .assert()
        .success()
        .stderr(predicate::str::contains("opaque-without-invariant"));
}

/// chelis#1678 [04-FIT-24]: an empty corpus is an error, not the success
/// Bucket 6b first chose. The cases where a directory target holds nothing
/// to check are mostly mistakes -- a typo landing on a sibling, a level
/// whose only sources sit under `target/` -- and a success beside exit 0 is
/// what an agent-driven gate ignores. It is symmetric with an empty `.ch`,
/// which has been an error since #247's M2.
#[test]
fn check_empty_directory_is_an_empty_corpus_error() {
    let dir = tempdir().expect("tempdir");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", dir.path().to_str().unwrap()])
        .assert()
        .code(2)
        .get_output()
        .clone();
    let parsed: Value =
        serde_json::from_slice(&output.stdout).expect("an empty corpus still emits the envelope");
    assert_eq!(parsed["files"], serde_json::json!([]));
    assert_eq!(parsed["errors"][0]["kind"], "empty_corpus", "{parsed}");
}

/// Bucket 6b regression: single-file `chelis check` continues to emit
/// the legacy single-report JSON shape so existing tooling does not break.
#[test]
fn check_single_file_keeps_legacy_report_shape() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("solo.ch");
    write_file(&path, "def main() -> i32 = 0\n");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .clone();
    let stdout = String::from_utf8_lossy(&output.stdout);
    let parsed: Value = serde_json::from_str(&stdout).expect("single-file check must remain JSON");
    assert!(
        parsed.get("score").is_some(),
        "single-file shape must keep `score` at top level: {stdout}"
    );
    assert!(
        parsed.get("files").is_none(),
        "single-file shape must NOT introduce `files` aggregator: {stdout}"
    );
}

// ----- Bucket 1: `grad` / `vmap` / `realize` in the host runtime -----
//
// These cover the closure of "host runtime does not support `grad`" /
// "...`vmap`" / "...`realize`" — `chelis test` and `chelis eval` now
// route those forms through `lower_subexpr_program` + the forward DAG
// evaluator (the same machinery the C backend uses) so the two lanes
// agree on programs that pass `chelis check`.
//
// Implementation: `crates/chelis-compiler-api/src/runtime/transforms.rs::apply_transform`.

/// Positive: `realize(...)` is identity in the host runtime; `chelis
/// eval` now produces the inner tensor's value instead of erroring.
#[test]
fn eval_realize_is_identity_in_host_runtime() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("realize_identity.ch");
    write_file(
        &path,
        "result = realize(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "tensor(shape=[3], data=[1.0, 2.0, 3.0])",
        ));
}

/// Positive: inline `grad(f)(x)` form. For f(x) = x*x, df/dx = 2x, so at
/// x=3 the gradient is 6.
#[test]
fn eval_grad_inline_application_returns_correct_gradient() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_inline.ch");
    write_file(
        &path,
        "def f(x: f32) -> f32 = mul(x, x)\n\
         result = grad(f)(cast(3.0, f32))\n",
    );

    // chelis#732 P1 ([05-OBS-4]): the rank-0 gradient renders bare.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("6.0"));
}

/// Positive: locally-bound `g = grad(f); g(x)` form. Closure is captured
/// at the binding site and applied later; output must match the inline
/// form above.
#[test]
fn eval_grad_locally_bound_then_applied_returns_correct_gradient() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_local_bind.ch");
    write_file(
        &path,
        "def f(x: f32) -> f32 = mul(x, x)\n\
         g = grad(f)\n\
         result = g(cast(3.0, f32))\n",
    );

    // chelis#732 P1 ([05-OBS-4]): the rank-0 gradient renders bare.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains("6.0"));
}

/// Positive: wrapper-fn-param form. `grad(loss, wrt=theta)(theta, x)`
/// inside a wrapper def — this is the form Coral / Shoals use. d/d
/// theta of sum(theta * x) is x = [3.0, 4.0].
#[test]
fn eval_grad_wrapper_fn_param_form_returns_correct_gradient() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_wrapper.ch");
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
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "tensor(shape=[2], data=[3.0, 4.0])",
        ));
}

/// Positive parity probe: the host eval result must agree with the C
/// backend's compiled binary on the wrapper-fn-param form, to within
/// 1e-6 elementwise. Same `wrt=theta` program used by the existing
/// `build_c_grad_named_fn_multi_param_wrt_builds_and_is_numerically_correct`
/// test — we share the source so any divergence between the two lanes
/// shows up here.
#[test]
fn eval_grad_wrapper_form_matches_c_backend_within_tolerance() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_parity.ch");
    write_file(
        &path,
        "def loss(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[f32] =\n\
           sum(mul(theta, x), 0)\n\
         def compute_grad(theta: tensor[2, f32], x: tensor[2, f32]) -> tensor[2, f32] =\n\
           grad(loss, wrt=theta)(theta, x)\n\
         out = compute_grad(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n",
    );

    // Host-eval lane.
    let host_output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(
        host_output.status.success(),
        "chelis eval failed: stderr={}",
        String::from_utf8_lossy(&host_output.stderr)
    );
    let host_stdout = String::from_utf8(host_output.stdout).expect("utf-8");
    assert!(
        host_stdout.contains("data=[3.0, 4.0]"),
        "host eval did not produce the expected gradient: {host_stdout}"
    );

    // C-backend lane.
    let out_dir = dir.path().join("grad-parity-build");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = gcc_link_generated(&out_dir, "grad_parity.c", "grad_parity");
    assert!(status.success(), "gcc link failed with status {status}");
    let c_run = StdCommand::new(out_dir.join("grad_parity"))
        .output()
        .expect("compiled binary should run");
    assert!(c_run.status.success(), "compiled binary failed");
    let c_stdout = String::from_utf8(c_run.stdout).expect("utf-8");
    assert!(
        c_stdout.contains("data=[3.0, 4.0]"),
        "C backend did not produce the expected gradient: {c_stdout}"
    );
}

/// Positive: `vmap(f)(xs)` lifts a scalar-tensor function over the
/// leading axis. f(x) = x * x applied elementwise via vmap to
/// [1.0, 2.0, 3.0] should yield [1.0, 4.0, 9.0].
#[test]
fn eval_vmap_returns_per_element_results() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("vmap_square.ch");
    write_file(
        &path,
        "def f(x: tensor[f32]) -> tensor[f32] = mul(copy(x), copy(x))\n\
         xs = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])\n\
         result = vmap(f)(xs)\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(predicate::str::contains(
            "result = tensor(shape=[3], data=[1.0, 4.0, 9.0])",
        ));
}

/// The core transform fragment admits direct top-level declarations and
/// inline or local `vmap` lambdas whose mapped type structure is explicit.
/// These controls retain those supported forms, including aliases and
/// shadowing within the typed local fragment, while the adjacent negative
/// cases fence underconstrained callable targets before evaluation.
#[test]
fn check_accepts_supported_core_transform_targets() {
    let dir = tempdir().expect("tempdir");
    let surf_path = dir.path().join("direct_transform_targets.ch");
    let deep_path = dir.path().join("direct_transform_targets.dp");
    write_file(
        &surf_path,
        "def loss(x: f32) -> f32 = mul(x, x)\n\
         def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
         gradient = grad(loss)\n\
         mapped = vmap(reduce)\n\
         def inline(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = \
         vmap(fn (v: tensor[4, 3, f32]) -> sum(v, 0i32))(t)\n\
         def local(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
           mapped = fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
           vmap(mapped)(t)\n\
         }\n\
         def local_alias(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
           mapped = fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
           alias = mapped\n\
           vmap(alias)(t)\n\
         }\n\
         def typed_shadow(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
           mapped = fn (v) -> sum(v, 0i32)\n\
           first = mapped(copy(t))\n\
           mapped = fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
           vmap(mapped)(t)\n\
         }\n",
    );

    let deep = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", surf_path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    fs::write(&deep_path, deep).expect("write desugared Deep program");

    for path in [&surf_path, &deep_path] {
        let json = run_json_check(path);
        assert_eq!(json["score"].as_f64(), Some(1.0), "{path:?}: {json}");
        assert!(
            json["errors"].as_array().is_some_and(Vec::is_empty),
            "{path:?}: {json}"
        );
    }

    let flow_controls = [
        (
            "vmap_typed_tuple_component",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               (mapped, keep) = (fn (v: tensor[4, 3, f32]) -> sum(v, 0i32), 0i32)\n\
               vmap(mapped)(t)\n\
             }\n",
        ),
        (
            "vmap_unrelated_untyped_tuple_sibling",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               (ordinary, mapped) = (\n\
                 fn (v) -> sum(v, 0i32),\n\
                 fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
               )\n\
               first = ordinary(copy(t))\n\
               vmap(mapped)(t)\n\
             }\n",
        ),
        (
            "vmap_typed_destructuring_shadow",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               first = mapped(copy(t))\n\
               (mapped, keep) = (\n\
                 fn (v: tensor[4, 3, f32]) -> sum(v, 0i32),\n\
                 0i32\n\
               )\n\
               vmap(mapped)(t)\n\
             }\n",
        ),
        (
            "vmap_typed_block_forwarded_alias",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
               alias = {\n\
                 forwarded = mapped\n\
                 forwarded\n\
               }\n\
               vmap(alias)(t)\n\
             }\n",
        ),
        (
            "vmap_local_binding_ascription",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped: tensor[4, 3, f32] -> tensor[3, f32] = \
                 fn (v) -> sum(v, 0i32)\n\
               vmap(mapped)(t)\n\
             }\n",
        ),
        (
            "vmap_typed_shadow_of_module_alias",
            "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
             mapped = reduce\n\
             def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
               vmap(mapped)(t)\n\
             }\n",
        ),
        (
            "vmap_typed_alias_after_module_alias_shadow",
            "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
             mapped = reduce\n\
             def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
               alias = mapped\n\
               vmap(alias)(t)\n\
             }\n",
        ),
        (
            "vmap_concrete_nested_tuple_parameter",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               vmap(fn (pair: (tensor[4, 3, f32], tensor[4, 3, f32])) -> \
                 sum(pair.1, 0i32))((copy(t), t))\n",
        ),
        (
            "vmap_concrete_nested_ref_parameter",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v: &tensor[4, 3, f32]) -> sum(v, 0i32)\n\
               vmap(mapped)(t)\n\
             }\n",
        ),
        (
            "vmap_symbolic_tensor_dimension",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               vmap(fn (v: tensor[*, 3, f32]) -> sum(v, 0i32))(t)\n",
        ),
        (
            "vmap_match_pattern_shadow",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               first = mapped(copy(t))\n\
               match (fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)) with {\n\
                 | mapped => vmap(mapped)(t)\n\
               }\n\
             }\n",
        ),
        (
            "vmap_typed_tuple_match_binding",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               match (fn (v: tensor[4, 3, f32]) -> sum(v, 0i32), 0i32) with {\n\
                 | (mapped, _) => vmap(mapped)(t)\n\
               }\n",
        ),
        (
            "vmap_typed_nested_match_sibling_isolation",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               match (\n\
                 fn (v) -> sum(v, 0i32),\n\
                 (0i32, fn (v: tensor[4, 3, f32]) -> sum(v, 0i32))\n\
               ) with {\n\
                 | (ordinary, (_, mapped)) => {\n\
                   first = ordinary(copy(t))\n\
                   vmap(mapped)(t)\n\
                 }\n\
               }\n",
        ),
        (
            "vmap_typed_match_shadow_of_module_alias",
            "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
             mapped = reduce\n\
             def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               match (fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)) with {\n\
                 | mapped => vmap(mapped)(t)\n\
               }\n",
        ),
        (
            "non_vmap_block_forwarded_tuple_consumer",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               (mapped, keep) = (fn (v) -> sum(v, 0i32), 0i32)\n\
               alias = {\n\
                 forwarded = mapped\n\
                 forwarded\n\
               }\n\
               alias(t)\n\
             }\n",
        ),
    ];

    for (stem, source) in flow_controls {
        let surf_path = dir.path().join(format!("{stem}.ch"));
        let deep_path = dir.path().join(format!("{stem}.dp"));
        write_file(&surf_path, source);
        let deep = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["deep", surf_path.to_str().unwrap()])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        fs::write(&deep_path, deep).expect("write desugared Deep control");

        for path in [&surf_path, &deep_path] {
            let json = run_json_check(path);
            assert_eq!(
                json["score"].as_f64(),
                Some(1.0),
                "{stem} ({path:?}) must remain in the supported fragment: {json}"
            );
            assert!(
                json["errors"].as_array().is_some_and(Vec::is_empty),
                "{stem} ({path:?}) must not inherit unrelated provenance: {json}"
            );
        }
    }

    let mut multi_module_deep = Vec::new();
    for (stem, source) in [
        (
            "alpha_transform_name",
            "module Alpha\n\
             def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n",
        ),
        (
            "beta_local_shadow",
            "module Beta\n\
             def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               reduce = fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
               vmap(reduce)(t)\n\
             }\n",
        ),
    ] {
        let path = dir.path().join(format!("{stem}.ch"));
        write_file(&path, source);
        let deep = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["deep", path.to_str().unwrap()])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        multi_module_deep.extend(deep);
        multi_module_deep.push(b'\n');
    }
    let multi_module_path = dir.path().join("module_scoped_transform_names.dp");
    fs::write(&multi_module_path, multi_module_deep).expect("write combined module Deep");
    let json = run_json_check(&multi_module_path);
    assert_eq!(
        json["score"].as_f64(),
        Some(1.0),
        "an unrelated module's function name must not taint a local lambda: {json}"
    );
    assert!(
        json["errors"].as_array().is_some_and(Vec::is_empty),
        "module-scoped transform provenance must remain isolated: {json}"
    );
}

/// Top-level aliases and shadows retain their core-fragment fence. The
/// #1887/#2109 row-typing cases now use the inferred slice, so their correct
/// signatures pass and their old false signatures fail at the type boundary.
#[test]
fn check_fences_non_direct_transform_targets() {
    let cases = [
        (
            "grad_chained_local_alias",
            "def relay(x: tensor[2, f32]) -> tensor[2, f32] = mul(x, x)\n\
             def loss(f: tensor[2, f32]) -> tensor[f32] = sum(relay(f), 0i32)\n\
             out = {\n\
               g = loss\n\
               h = g\n\
               grad(h)(to_tensor([1.0f32, 2.0f32]))\n\
             }\n",
            "grad",
        ),
        (
            "grad_top_level_alias",
            "def loss(x: f32) -> f32 = mul(x, x)\n\
             g = loss\n\
             out = grad(g)(1.0f32)\n",
            "grad",
        ),
        (
            "grad_module_to_local_alias",
            "def loss(x: f32) -> f32 = mul(x, x)\n\
             g = loss\n\
             out = {\n\
               h = g\n\
               grad(h)(1.0f32)\n\
             }\n",
            "grad",
        ),
        (
            "grad_shadowed_lambda",
            "def loss(x: tensor[2, f32]) -> tensor[f32] = sum(mul(x, x), 0i32)\n\
             out = {\n\
               loss = fn (x: tensor[2, f32]) -> sum(x, 0i32)\n\
               grad(loss)(to_tensor([1.0f32, 2.0f32]))\n\
             }\n",
            "grad",
        ),
        (
            "vmap_untyped_inline_lambda",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               vmap(fn (v) -> sum(v, 0i32))(t)\n",
            "vmap",
        ),
        (
            "vmap_untyped_local_lambda_false_result",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               vmap(mapped)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_untyped_local_lambda_correct_result",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               vmap(mapped)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_untyped_local_lambda_alias",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               alias = mapped\n\
               vmap(alias)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_wildcard_inline_lambda",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               vmap(fn (v: _) -> sum(v, 0i32))(t)\n",
            "vmap",
        ),
        (
            "vmap_wildcard_local_lambda",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               mapped = fn (v: _) -> sum(v, 0i32)\n\
               vmap(mapped)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_wildcard_local_lambda_alias",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               mapped = fn (v: _) -> sum(v, 0i32)\n\
               alias = mapped\n\
               vmap(alias)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_nested_tuple_type_hole_inline",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               vmap(fn (pair: (tensor[4, 3, f32], _)) -> \
                 sum(pair.1, 0i32))((copy(t), t))\n",
            "vmap",
        ),
        (
            "vmap_nested_ref_type_hole_local_alias",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               mapped = fn (v: &_) -> sum(v, 0i32)\n\
               alias = mapped\n\
               vmap(alias)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_untyped_tuple_destructure",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               (mapped, keep) = (fn (v) -> sum(v, 0i32), 0i32)\n\
               vmap(mapped)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_untyped_tuple_sibling",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               (typed, mapped) = (\n\
                 fn (v: tensor[4, 3, f32]) -> sum(v, 0i32),\n\
                 fn (v) -> sum(v, 0i32)\n\
               )\n\
               vmap(mapped)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_untyped_block_forwarded_alias",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               alias = {\n\
                 forwarded = mapped\n\
                 forwarded\n\
               }\n\
               vmap(alias)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_untyped_alias_survives_typed_shadow",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               alias = mapped\n\
               mapped = fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
               vmap(alias)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_local_alias",
            "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
             def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = reduce\n\
               vmap(mapped)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_module_to_local_alias",
            "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
             g = reduce\n\
             def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = g\n\
               vmap(mapped)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_direct_module_alias",
            "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
             mapped = reduce\n\
             out = vmap(mapped)\n",
            "vmap",
        ),
        (
            "vmap_untyped_shadow_of_module_alias",
            "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
             mapped = reduce\n\
             def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = fn (v) -> sum(v, 0i32)\n\
               alias = mapped\n\
               vmap(alias)(t)\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_untyped_direct_match_binding",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               match (fn (v) -> sum(v, 0i32)) with {\n\
                 | mapped => vmap(mapped)(t)\n\
               }\n",
            "vmap",
        ),
        (
            "vmap_untyped_tuple_match_binding",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               pair = (fn (v) -> sum(v, 0i32), 0i32)\n\
               match pair with {\n\
                 | (mapped, _) => vmap(mapped)(t)\n\
               }\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_untyped_nested_tuple_match_binding",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               pair = ((0i32, fn (v) -> sum(v, 0i32)), 1i32)\n\
               match pair with {\n\
                 | ((_, mapped), _) => vmap(mapped)(t)\n\
               }\n\
             }\n",
            "vmap",
        ),
        (
            "vmap_untyped_as_match_binding",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               match (fn (v) -> sum(v, 0i32)) with {\n\
                 | whole @ mapped => vmap(whole)(t)\n\
               }\n",
            "vmap",
        ),
        (
            "vmap_module_alias_match_binding",
            "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
             mapped = reduce\n\
             def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               match mapped with {\n\
                 | alias => vmap(alias)(t)\n\
               }\n",
            "vmap",
        ),
    ];

    for (stem, source, transform) in cases {
        let dir = tempdir().expect("tempdir");
        let surf_path = dir.path().join(format!("{stem}.ch"));
        let deep_path = dir.path().join(format!("{stem}.dp"));
        write_file(&surf_path, source);
        let deep = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["deep", surf_path.to_str().unwrap()])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        fs::write(&deep_path, deep).expect("write desugared Deep program");

        for path in [&surf_path, &deep_path] {
            let json = run_json_check(path);
            let errors = json["errors"].as_array().expect("errors array");
            let owned_fence_errors = errors
                .iter()
                .filter(|error| {
                    error["kind"] == "TypeMismatch"
                        && error["message"].as_str().is_some_and(|message| {
                            message.contains("core transform fragment")
                                && message.contains(transform)
                        })
                })
                .count();
            let vmap_row_typing = transform == "vmap"
                && !matches!(
                    stem,
                    "vmap_local_alias"
                        | "vmap_module_to_local_alias"
                        | "vmap_direct_module_alias"
                        | "vmap_module_alias_match_binding"
                );
            if vmap_row_typing {
                assert_eq!(
                    owned_fence_errors, 0,
                    "{stem} ({path:?}) is decided by row typing, not the core fence: {json}"
                );
                let correct_result = matches!(
                    stem,
                    "vmap_untyped_local_lambda_correct_result"
                        | "vmap_untyped_local_lambda_alias"
                        | "vmap_untyped_alias_survives_typed_shadow"
                        | "vmap_untyped_shadow_of_module_alias"
                );
                if correct_result {
                    assert_eq!(json["score"].as_f64(), Some(1.0), "{stem}: {json}");
                    assert!(errors.is_empty(), "{stem}: {json}");
                } else {
                    assert!(
                        json["score"].as_f64().is_some_and(|score| score < 1.0),
                        "{stem} ({path:?}) must reject its false result: {json}"
                    );
                }
            } else {
                assert!(
                    json["score"].as_f64().is_some_and(|score| score < 1.0),
                    "{stem} ({path:?}) must not receive a perfect check score: {json}"
                );
                assert_eq!(
                    owned_fence_errors, 1,
                    "{stem} ({path:?}) must carry one owned core-transform fence: {json}"
                );
            }
        }
    }
}

/// The launch-core fence evaluates forwarded callable values structurally at
/// every module/local binding, match-result, and direct-target boundary. These
/// rows keep the end-to-end classifier honest rather than testing only the
/// assigned alias shape that first exposed #2109.
#[test]
fn check_tracks_core_transform_values_across_forwarding_results() {
    let cases = [
        (
            "direct_tuple_projection_untyped",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               pair = (fn (v) -> sum(v, 0i32), 0i32)\n\
               vmap(pair.0)(t)\n\
             }\n",
            true,
        ),
        (
            "inline_tuple_projection_untyped",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               vmap((fn (v) -> sum(v, 0i32), 0i32).0)(t)\n",
            true,
        ),
        (
            "transparent_let_target_untyped",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               vmap({\n\
                 mapped = fn (v) -> sum(v, 0i32)\n\
                 mapped\n\
               })(t)\n",
            true,
        ),
        (
            "transparent_block_target_untyped",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               vmap(do {\n\
                 0i32;\n\
                 fn (v) -> sum(v, 0i32)\n\
               })(t)\n",
            true,
        ),
        (
            "as_bound_tuple_projection_untyped",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               pair = (fn (v) -> sum(v, 0i32), 0i32)\n\
               match pair with {\n\
                 | whole @ (mapped, _) => vmap(whole.0)(t)\n\
               }\n\
             }\n",
            true,
        ),
        (
            "direct_match_result_untyped",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] =\n\
               vmap(match (fn (v) -> sum(v, 0i32)) with {\n\
                 | mapped => mapped\n\
               })(t)\n",
            true,
        ),
        (
            "local_match_result_untyped",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               mapped = match (fn (v) -> sum(v, 0i32)) with {\n\
                 | forwarded => forwarded\n\
               }\n\
               vmap(mapped)(t)\n\
             }\n",
            true,
        ),
        (
            "nested_match_result_all_untyped",
            "def probe(flag: bool, t: tensor[5, 4, 3, f32]) -> tensor[4, 3, f32] = {\n\
               carrier = match flag with {\n\
                 | true => ((fn (v) -> sum(v, 0i32), 0i32), 1i32)\n\
                 | false => ((fn (v) -> sum(v, 0i32), 0i32), 1i32)\n\
               }\n\
               vmap((carrier.0).0)(t)\n\
             }\n",
            true,
        ),
        (
            "module_tuple_projection_alias",
            "def reduce(v: tensor[4, 3, f32]) -> tensor[3, f32] = sum(v, 0i32)\n\
             pair = (reduce, 0i32)\n\
             mapped = pair.0\n\
             out = vmap(mapped)\n",
            true,
        ),
        (
            "direct_tuple_projection_typed",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               pair = (fn (v: tensor[4, 3, f32]) -> sum(v, 0i32), 0i32)\n\
               vmap(pair.0)(t)\n\
             }\n",
            false,
        ),
        (
            "inline_tuple_projection_typed",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               vmap((fn (v: tensor[4, 3, f32]) -> sum(v, 0i32), 0i32).0)(t)\n",
            false,
        ),
        (
            "transparent_let_target_typed",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               vmap({\n\
                 mapped = fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
                 mapped\n\
               })(t)\n",
            false,
        ),
        (
            "transparent_block_target_typed",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               vmap(do {\n\
                 0i32;\n\
                 fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)\n\
               })(t)\n",
            false,
        ),
        (
            "as_bound_tuple_projection_typed",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               pair = (fn (v: tensor[4, 3, f32]) -> sum(v, 0i32), 0i32)\n\
               match pair with {\n\
                 | whole @ (mapped, _) => vmap(whole.0)(t)\n\
               }\n\
             }\n",
            false,
        ),
        (
            "direct_match_result_typed",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               vmap(match (fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)) with {\n\
                 | mapped => mapped\n\
               })(t)\n",
            false,
        ),
        (
            "local_match_result_typed",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               mapped = match (fn (v: tensor[4, 3, f32]) -> sum(v, 0i32)) with {\n\
                 | forwarded => forwarded\n\
               }\n\
               vmap(mapped)(t)\n\
             }\n",
            false,
        ),
        (
            "nested_match_result_constrained_by_typed_arm",
            "def probe(flag: bool, t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] = {\n\
               carrier = match flag with {\n\
                 | true => ((fn (v) -> sum(v, 0i32), 0i32), 1i32)\n\
                 | false => ((fn (v: tensor[4, 3, f32]) -> sum(v, 0i32), 0i32), 1i32)\n\
               }\n\
               vmap((carrier.0).0)(t)\n\
             }\n",
            false,
        ),
        (
            "module_tuple_projection_typed_lambda",
            "pair = (\n\
               fn (v: tensor[4, 3, f32]) -> sum(v, 0i32),\n\
               0i32\n\
             )\n\
             mapped = pair.0\n\
             def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               vmap(mapped)(t)\n",
            false,
        ),
    ];

    let dir = tempdir().expect("tempdir");
    for (stem, source, must_fence) in cases {
        let surf_path = dir.path().join(format!("{stem}.ch"));
        let deep_path = dir.path().join(format!("{stem}.dp"));
        write_file(&surf_path, source);
        let deep = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["deep", surf_path.to_str().unwrap()])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        fs::write(&deep_path, deep).expect("write desugared Deep forwarding case");

        for path in [&surf_path, &deep_path] {
            let json = run_json_check(path);
            let errors = json["errors"].as_array().expect("errors array");
            let owned_fence_errors = errors
                .iter()
                .filter(|error| {
                    error["kind"] == "TypeMismatch"
                        && error["message"].as_str().is_some_and(|message| {
                            message.contains("core transform fragment") && message.contains("vmap")
                        })
                })
                .count();
            if stem.ends_with("_untyped") {
                assert!(
                    json["score"].as_f64().is_some_and(|score| score < 1.0),
                    "{stem} ({path:?}) must reject its false mapped result: {json}"
                );
                assert_eq!(
                    owned_fence_errors, 0,
                    "{stem} ({path:?}) must be decided by row typing: {json}"
                );
            } else if must_fence {
                assert!(
                    json["score"].as_f64().is_some_and(|score| score < 1.0),
                    "{stem} ({path:?}) must not receive a perfect score: {json}"
                );
                assert_eq!(
                    owned_fence_errors, 1,
                    "{stem} ({path:?}) needs exactly one owned vmap fence: {json}"
                );
            } else {
                assert_eq!(
                    json["score"].as_f64(),
                    Some(1.0),
                    "{stem} ({path:?}) must remain supported: {json}"
                );
                assert!(
                    errors.is_empty(),
                    "{stem} ({path:?}) must not inherit fence provenance: {json}"
                );
            }
        }
    }
}

#[test]
fn check_classifies_generated_deep_tensor_holes_by_mapped_structure() {
    let dir = tempdir().expect("tempdir");
    let cases = [
        (
            "dimension_hole",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               vmap(fn (v: tensor[d, 3, f32]) -> sum(v, 0i32))(t)\n",
            "(d-var {} d)",
            "(d-var {} _)",
            false,
        ),
        (
            "precision_hole",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 3, f32] =\n\
               vmap(fn (v: tensor[4, 3, f32]) -> sum(v, 0i32))(t)\n",
            "(t-tensor {} (d-lit {} 4) (d-lit {} 3) (t-prim {} f32))",
            "(t-tensor {} (d-lit {} 4) (d-lit {} 3) (t-var {} _))",
            false,
        ),
        (
            "rank_hole",
            "def probe(t: tensor[5, 4, 3, f32]) -> tensor[5, 4, 3, f32] =\n\
               vmap(fn (v: tensor[..r, f32]) -> relu(v))(t)\n",
            "(d-rank {} r)",
            "(d-rank {} _)",
            false,
        ),
    ];

    for (stem, source, needle, replacement, must_fence) in cases {
        let surf_path = dir.path().join(format!("{stem}.ch"));
        let deep_path = dir.path().join(format!("{stem}.dp"));
        write_file(&surf_path, source);
        let generated = Command::cargo_bin("chelis")
            .expect("binary")
            .env("CHELIS_STYLE_GATE_DISABLE", "1")
            .args(["deep", surf_path.to_str().unwrap()])
            .assert()
            .success()
            .get_output()
            .stdout
            .clone();
        let generated = String::from_utf8(generated).expect("generated Deep is UTF-8");
        assert!(
            generated.contains(needle),
            "{stem} fixture must carry `{needle}` before mutation: {generated}"
        );
        let holed = generated.replacen(needle, replacement, 1);
        fs::write(&deep_path, holed).expect("write generated Deep hole case");

        let json = run_json_check(&deep_path);
        let errors = json["errors"].as_array().expect("errors array");
        let owned_fence_errors = errors
            .iter()
            .filter(|error| {
                error["kind"] == "TypeMismatch"
                    && error["message"].as_str().is_some_and(|message| {
                        message.contains("core transform fragment") && message.contains("vmap")
                    })
            })
            .count();
        if must_fence {
            assert!(
                json["score"].as_f64().is_some_and(|score| score < 1.0),
                "{stem} can absorb transform-relevant shape and must be fenced: {json}"
            );
            assert_eq!(
                owned_fence_errors, 1,
                "{stem} needs one owned vmap fence diagnostic: {json}"
            );
        } else {
            assert_eq!(
                json["score"].as_f64(),
                Some(1.0),
                "{stem} preserves explicit mapped tensor structure: {json}"
            );
            assert!(
                errors.is_empty(),
                "{stem} must remain a supported generated-Deep control: {json}"
            );
        }
    }
}

/// Negative parity for `realize`: exercising it in the host lane with
/// no inner expression must produce a clean error rather than a panic.
/// Synthesized programs with bad shape are caught at type-check; this
/// test pins that the runtime path doesn't regress to "host runtime
/// does not support `realize`" once the binding case starts hitting
/// the new arm.
#[test]
fn eval_realize_does_not_regress_to_host_runtime_unsupported() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("realize_no_unsupported.ch");
    write_file(&path, "result = realize(to_tensor([cast(1.0, f32)]))\n");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("host runtime does not support `realize`")
                .not()
                .and(predicate::str::contains("tensor(shape=[1], data=[1.0])")),
        );
}

/// Negative parity for `grad`: the previously-emitted "host runtime does
/// not support `grad`" string must no longer appear on a passing program.
#[test]
fn eval_grad_does_not_regress_to_host_runtime_unsupported() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_no_unsupported.ch");
    write_file(
        &path,
        "def f(x: f32) -> f32 = mul(x, x)\n\
         result = grad(f)(cast(2.0, f32))\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("host runtime does not support `grad`")
                .not()
                // chelis#732 P1 ([05-OBS-4]): the rank-0 gradient renders bare.
                .and(predicate::str::contains("4.0")),
        );
}

/// Negative parity for `vmap`: same closure check.
#[test]
fn eval_vmap_does_not_regress_to_host_runtime_unsupported() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("vmap_no_unsupported.ch");
    write_file(
        &path,
        "def f(x: tensor[f32]) -> tensor[f32] = mul(copy(x), copy(x))\n\
         xs = to_tensor([cast(2.0, f32), cast(3.0, f32)])\n\
         result = vmap(f)(xs)\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .stdout(
            predicate::str::contains("host runtime does not support `vmap`")
                .not()
                .and(predicate::str::contains("data=[4.0, 9.0]")),
        );
}

// ----- Keyed draws through the C backend ---------------------------------
//
// Pre-Bucket-5, `chelis build --target c|hip` rejected any program that
// contained a seeded draw anywhere in the deeply-walked AST, and the gate was
// project-wide: one such file blocked `chelis build` of every sibling. With
// explicit keys (chelis#2413) a draw's key is an ordinary value; the tests
// below pin, in key form:
//
//   1. A keyed `uniform_like` builds, runs, and prints the draw the
//      [05-RNG-2] reference gives for its key, so a key silently defaulted
//      or dropped in the generated C fails.
//   2. Same key gives the same bytes across runs; a different key gives
//      different bytes.
//   3. A sibling program in a directory where another file draws builds
//      to C (the gate cannot be re-introduced).
//
// Expected draws come from `common::key_ref`, pinned against `key_ref.py`
// by `dropout_fixed_stream_cli::worked_values_match_key_ref_py`.

fn write_keyed_uniform(path: &Path, seed: i64) {
    let contents = format!(
        r#"template = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
sampled = uniform_like(key_from_seed({seed}i64), copy(template), cast(0.0, f32), cast(1.0, f32))
"#
    );
    write_file(path, &contents);
}

/// The f32 draws of `uniform_like(key_from_seed(seed), template, 0, 1)` over
/// four elements.
fn reference_uniform_unit(seed: i64) -> Vec<f32> {
    common::key_ref::uniform_f32(common::key_ref::key_from_seed(seed), 4, 0.0, 1.0)
}

/// A printed f32 tensor root's elements; the printed decimals round-trip at
/// binary32.
fn printed_f32(stdout: &str, root: &str) -> Vec<f32> {
    common::parse_tensor_data(stdout, root)
        .into_iter()
        .map(|value| value as f32)
        .collect()
}

/// Build `src` to C under `out_dir`, link it as `binary`, run it once and
/// return its stdout.
fn build_link_run(src: &Path, out_dir: &Path, binary: &str) -> String {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = gcc_link_generated(out_dir, &format!("{binary}.c"), binary);
    assert!(status.success(), "gcc compile/link of generated C failed");
    let run = StdCommand::new(out_dir.join(binary))
        .output()
        .expect("compiled binary must run");
    assert!(
        run.status.success(),
        "{binary} exited non-zero: {}",
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8(run.stdout).expect("utf-8 stdout")
}

#[test]
fn build_c_keyed_uniform_like_prints_the_reference_draw() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("seeded.ch");
    write_keyed_uniform(&src, 7);
    let stdout = build_link_run(&src, &dir.path().join("out"), "seeded");
    assert_eq!(
        printed_f32(&stdout, "sampled"),
        reference_uniform_unit(7),
        "the C draw must read key_from_seed(7), not a defaulted key:\n{stdout}"
    );
}

#[test]
fn build_c_keyed_uniform_like_is_deterministic_and_key_sensitive() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("seeded.ch");
    let out_dir = dir.path().join("out");
    write_keyed_uniform(&src, 7);
    let first = build_link_run(&src, &out_dir, "seeded");
    let run = || {
        let output = StdCommand::new(out_dir.join("seeded"))
            .output()
            .expect("compiled binary must run");
        assert!(output.status.success(), "seeded binary exited non-zero");
        String::from_utf8(output.stdout).expect("utf-8 stdout")
    };
    let second = run();
    let third = run();
    assert_eq!(
        first, second,
        "keyed draw determinism: run 1 vs run 2 differ"
    );
    assert_eq!(
        second, third,
        "keyed draw determinism: run 2 vs run 3 differ"
    );

    // A different key produces different bytes, each the reference's.
    let other_src = dir.path().join("seeded_other.ch");
    write_keyed_uniform(&other_src, 42);
    let other = build_link_run(&other_src, &dir.path().join("out_other"), "seeded_other");
    assert_ne!(
        printed_f32(&first, "sampled"),
        printed_f32(&other, "sampled"),
        "key_from_seed(7) and key_from_seed(42) must produce different bytes"
    );
    assert_eq!(printed_f32(&other, "sampled"), reference_uniform_unit(42));
}

#[test]
fn build_c_keyed_draw_does_not_block_a_sibling_build() {
    // Pre-Bucket-5, a project-wide gate aborted the build of every sibling
    // of a file that drew. A sibling `.ch` file that does not draw builds
    // cleanly even when a file in the same directory does; the test pins
    // that such a gate cannot be re-introduced without breaking it.
    let dir = tempdir().expect("tempdir");
    let keyed_path = dir.path().join("uses_key.ch");
    let plain_path = dir.path().join("plain_sibling.ch");
    write_keyed_uniform(&keyed_path, 7);
    write_file(
        &plain_path,
        "xs = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])\n",
    );

    let out_dir = dir.path().join("out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            "--emit-c",
            plain_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert!(
        out_dir.join("plain_sibling.c").exists(),
        "plain sibling without a draw must build to C"
    );

    // And the drawing file builds standalone too.
    let keyed_out = dir.path().join("keyed-out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            "--emit-c",
            keyed_path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            keyed_out.to_str().unwrap(),
        ])
        .assert()
        .success();
    assert!(
        keyed_out.join("uses_key.c").exists(),
        "the drawing file must build to C"
    );
}

/// A draw inside a helper reads the key its caller passes: equal keys give
/// equal draws and a different key a different draw, each the reference's.
#[test]
fn cross_function_key_wrapper_draws_from_its_callers_key_in_c_backend() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("cross_function_key.ch");
    write_file(
        &src,
        r#"template = to_tensor([cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)])
def sample(k: key, t: tensor[4, f32]) -> tensor[4, f32] =
  uniform_like(k, copy(t), 0.0, 1.0)
seven = sample(key_from_seed(7i64), copy(template))
seven_again = sample(key_from_seed(7i64), copy(template))
forty_two = sample(key_from_seed(42i64), copy(template))
"#,
    );
    let stdout = build_link_run(&src, &dir.path().join("out"), "cross_function_key");
    let seven = printed_f32(&stdout, "seven");
    assert_eq!(seven, reference_uniform_unit(7), "{stdout}");
    assert_eq!(
        printed_f32(&stdout, "seven_again"),
        seven,
        "equal keys through a wrapper call must give the same draw"
    );
    assert_eq!(
        printed_f32(&stdout, "forty_two"),
        reference_uniform_unit(42),
        "{stdout}"
    );
    assert_ne!(reference_uniform_unit(7), reference_uniform_unit(42));
}

/// #1872, [05-OP-37]/[05-RNG-1]: a source-fixed entry retains its sealed
/// execution plan independently of unrelated declarations or source spelling.
#[test]
fn fixed_control_c_entry_is_independent_of_host_siblings() {
    let dir = tempdir().unwrap();
    for (stem, sibling, deep, pure) in [
        ("bare", "", false, false),
        ("with_host", "def status() -> i64 = 7i64\n", false, false),
        ("deep_entry", "", true, false),
        ("seeded_helper", "", false, false),
        ("local_key", "", false, false),
        ("pure_entry", "", false, true),
    ] {
        let surf = dir.path().join(format!("{stem}.ch"));
        write_file(
            &surf,
            &format!(
                "{}{sibling}",
                if pure {
                    "def sample(x: tensor[4, f32]) -> tensor[4, f32] = add(x, x)\n"
                } else if stem == "seeded_helper" {
                    "def keep(k: key, x: tensor[4, f32]) -> tensor[4, f32] = dropout(k, x, 0.5f32)\ndef sample(x: tensor[4, f32]) -> tensor[4, f32] = keep(key_from_seed(1i64), x)\n"
                } else if stem == "local_key" {
                    "def sample(x: tensor[4, f32]) -> tensor[4, f32] = {\n  k = key_from_seed(1i64)\n  dropout(k, x, 0.5f32)\n}\n"
                } else {
                    include_str!("../../../examples/dropout_entry.ch")
                }
            ),
        );
        for args in [
            vec!["fmt", "--inplace"],
            vec!["lint", "--check"],
            vec!["check"],
        ] {
            Command::cargo_bin("chelis")
                .unwrap()
                .args(args)
                .arg(&surf)
                .assert()
                .success();
        }
        let source = if deep {
            let output = Command::cargo_bin("chelis")
                .unwrap()
                .arg("deep")
                .arg(&surf)
                .output()
                .unwrap();
            assert!(output.status.success());
            let path = dir.path().join(format!("{stem}.dp"));
            fs::write(&path, output.stdout).unwrap();
            Command::cargo_bin("chelis")
                .unwrap()
                .arg("check")
                .arg(&path)
                .assert()
                .success();
            path
        } else {
            surf
        };
        let out = dir.path().join(stem);
        Command::cargo_bin("chelis")
            .unwrap()
            .arg("build")
            .arg("--emit-c")
            .arg(&source)
            .args(["--target", "c", "--output"])
            .arg(&out)
            .assert()
            .success();
        let invocation = if pure {
            "chelis_tensor *outputs[1]; pure_entry(&x, 1, outputs, 1); chelis_tensor *y = outputs[0];"
        } else {
            &format!("chelis_tensor *y = {}(x);", authored_c_symbol("sample"))
        };
        let expected = if pure {
            "0x40000000u, 0x40800000u, 0x40c00000u, 0x41000000u"
        } else {
            // key_from_seed(1) at rate 0.5 drops elements 0 and 3 (key_ref.py).
            "0u, 0x40800000u, 0x40c00000u, 0u"
        };
        let driver = format!(
            r#"
#define main generated_main
#include "{stem}.c"
#undef main
#include <assert.h>
#include <string.h>
int main(void) {{
    int64_t n = 4;
    chelis_tensor *x = chelis_alloc(1, &n, CHELIS_DTYPE_F32);
    const uint32_t original[] = {{0x3f800000u, 0x40000000u, 0x40400000u, 0x40800000u}};
    chelis_tensor_write *guard = chelis_tensor_begin_write(x);
    memcpy(chelis_tensor_write_view(guard).data, original, sizeof(original));
    chelis_tensor_end_write(guard);
    const uint32_t expected[] = {{{expected}}};
    for (int repeat = 0; repeat < 3; ++repeat) {{
        {invocation}
        assert(chelis_tensor_rank(y) == 1 && chelis_tensor_shape(y, 0) == n);
        chelis_read_view view = chelis_tensor_read_view(y);
        assert(view.dtype == CHELIS_DTYPE_F32 && view.count == n);
        assert(memcmp(view.data, expected, sizeof(expected)) == 0);
        assert(memcmp(chelis_tensor_read_view(x).data, original, sizeof(original)) == 0);
        chelis_tensor_release(y);
    }}
    chelis_tensor_release(x);
    return 0;
}}
"#
        );
        write_file(&out.join("driver.c"), &driver);
        assert!(gcc_link_generated(&out, "driver.c", "driver").success());
        assert!(
            StdCommand::new(out.join("driver"))
                .status()
                .unwrap()
                .success()
        );
    }
}

#[test]
fn concrete_static_rate_local_helper_executes_eval_and_native_c() {
    let dir = tempdir().unwrap();
    let source = dir.path().join("static_rate.ch");
    write_file(
        &source,
        include_str!("../../../examples/dropout_static_rate.ch"),
    );
    for args in [
        vec!["fmt", "--inplace"],
        vec!["lint", "--check"],
        vec!["check"],
    ] {
        Command::cargo_bin("chelis")
            .unwrap()
            .args(args)
            .arg(&source)
            .assert()
            .success();
    }
    let expected = "result.0 = tensor(shape=[4], data=[2.0, 0.0, 0.0, 2.0])\nresult.1 = tensor(shape=[4], data=[2.0, 0.0, 0.0, 0.0])\nresult.2 = tensor(shape=[4], data=[2.0, 2.0, 0.0, 0.0])\nresult.3 = tensor(shape=[4], data=[1.0, 1.0, 1.0, 1.0])\n";
    Command::cargo_bin("chelis")
        .unwrap()
        .args(["eval", "--file"])
        .arg(&source)
        .assert()
        .success()
        .stdout(expected);
    let out = dir.path().join("out");
    Command::cargo_bin("chelis")
        .unwrap()
        .arg("build")
        .arg("--emit-c")
        .arg(&source)
        .arg("--output")
        .arg(&out)
        .assert()
        .success();
    assert!(gcc_link_generated(&out, "static_rate.c", "static_rate").success());
    let result = StdCommand::new(out.join("static_rate")).output().unwrap();
    assert!(
        result.status.success(),
        "{}",
        String::from_utf8_lossy(&result.stderr)
    );
    assert_eq!(String::from_utf8(result.stdout).unwrap(), expected);

    // [05-OP-37]: a runtime rate is an ordinary operand (chelis#2411). This
    // program has no export list, so every top-level def is public (§2 Surf).
    // A keyless public entry is an arity error, refused at build time instead
    // of drawing from a defaulted key; the entry that takes its key builds.
    write_file(
        &source,
        "def keep(x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(x, rate)\n",
    );
    let runtime_rate = dir.path().join("runtime_rate");
    let refused = Command::cargo_bin("chelis")
        .unwrap()
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .arg("build")
        .arg("--emit-c")
        .arg(&source)
        .arg("--output")
        .arg(&runtime_rate)
        .assert()
        .failure();
    let stderr = String::from_utf8_lossy(&refused.get_output().stderr).into_owned();
    assert!(
        stderr.contains("`dropout(x, rate)` is the retired counter-stream spelling"),
        "{stderr}"
    );
    write_file(
        &source,
        "def keep(k: key, x: tensor[4, f32], rate: f32) -> tensor[4, f32] = dropout(k, x, rate)\n",
    );
    Command::cargo_bin("chelis")
        .unwrap()
        .args(["fmt", "--inplace"])
        .arg(&source)
        .assert()
        .success();
    Command::cargo_bin("chelis")
        .unwrap()
        .arg("build")
        .arg("--emit-c")
        .arg(&source)
        .arg("--output")
        .arg(&runtime_rate)
        .assert()
        .success();
}

/// [05-OP-37]/[05-RNG-2]: a concrete call of a dtype-generic static-rate
/// helper draws from the key it is given through the CLI host build. The
/// key is a concrete `key` parameter beside the `[p: Float]` binder
/// ([04-LIN-10]).
#[test]
fn build_c_runs_generic_static_rate_dropout_host_helper() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("generic_dropout.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &src,
        r#"
def keep[p: Float](k: key, x: tensor[4, p]) -> tensor[4, p] = dropout(k, x, cast(0.5, p))
result = keep(key_from_seed(7i64), to_tensor([1.0f32, 1.0f32, 1.0f32, 1.0f32]))
"#,
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let status = gcc_link_generated(&out_dir, "generic_dropout.c", "generic_dropout");
    assert!(status.success(), "gcc compile/link of generated C failed");
    let run = StdCommand::new(out_dir.join("generic_dropout"))
        .output()
        .expect("compiled binary must run");
    assert!(run.status.success(), "compiled binary exited non-zero");
    // `dropout(key_from_seed(7), ones, 0.5)` from `common::key_ref`.
    let expected = common::key_ref::dropout_f32(common::key_ref::key_from_seed(7), &[1.0; 4], 0.5);
    assert_eq!(expected, [0.0, 2.0, 2.0, 2.0]);
    assert_eq!(
        String::from_utf8(run.stdout).unwrap(),
        "result = tensor(shape=[4], data=[0.0, 2.0, 2.0, 2.0])\n"
    );
}

#[test]
fn build_c_mnist_loss_tail_tensor_pipeline_compiles_object() {
    let dir = tempdir().expect("tempdir");
    let src = dir.path().join("mnist_loss_tail.ch");
    let out_dir = dir.path().join("out");
    write_file(
        &src,
        r#"def loss_tail(logits: tensor[2, 3, f32], labels: tensor[2, 3, f32]) -> tensor[f32] = softmax(logits, 1) |> log |> mul(labels) |> sum(1) |> neg |> mean(0)
logits = to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(0.5, f32), cast(1.5, f32), cast(2.5, f32)]])
labels = to_tensor([[cast(0.0, f32), cast(0.0, f32), cast(1.0, f32)], [cast(1.0, f32), cast(0.0, f32), cast(0.0, f32)]])
loss_value = loss_tail(logits, labels)
"#,
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    let c_src = fs::read_to_string(out_dir.join("mnist_loss_tail.c")).expect("read generated C");
    assert!(
        !c_src.contains("/* unsupported builtin sum */")
            && !c_src.contains("/* unsupported builtin mean */")
            && !c_src.contains("-(__arg0_")
            && !c_src.contains("void* __arg0_"),
        "MNIST loss tail must stay on tensor-helper codegen, got:\n{c_src}"
    );
    let status = gcc_compile_generated_object(&out_dir, "mnist_loss_tail.c");
    assert!(status.success(), "gcc object compile of generated C failed");
    let status = gcc_link_generated(&out_dir, "mnist_loss_tail.c", "mnist_loss_tail");
    assert!(status.success(), "gcc link of generated C failed");
    let run = StdCommand::new(out_dir.join("mnist_loss_tail"))
        .output()
        .expect("compiled binary must run");
    assert!(run.status.success(), "compiled binary exited non-zero");
    let stdout = String::from_utf8(run.stdout).expect("utf-8 stdout");
    // [05-OBS-4] (chelis#732 Phase 2): the rank-0 result renders as the
    // BARE scalar - the shape=[] wrapper is not an exit form.
    let loss_line = stdout
        .lines()
        .find(|l| l.starts_with("loss_value = "))
        .unwrap_or_else(|| panic!("no loss_value root in:\n{stdout}"));
    assert!(
        !loss_line.contains("tensor("),
        "MNIST loss tail tensor[f32] rank-0 result must render bare ([05-OBS-4]), got:\n{stdout}"
    );
    loss_line["loss_value = ".len()..]
        .trim()
        .parse::<f64>()
        .unwrap_or_else(|e| panic!("bare rank-0 payload must parse as a float: {e}\n{stdout}"));
}

/// Bucket 4b regression: `chelis check`, `chelis eval --file`, and
/// `chelis build --target c` (compiled + run) must all agree on
/// `to_tensor([[...], [...]])` for 2-D nested-list literals.
///
/// Previously the typer rejected the form with `to_tensor expects
/// numeric or bool List elements, got List f32`, so the case never made
/// it past `chelis check`. The fix extends the typer to recurse through
/// nested `List<...>` wrappers and reports rank = nesting depth, with
/// matching support in the host runtime
/// (`nested_list_to_tensor_data`) and the C runtime
/// (`chelis_tensor_from_value_list`).
#[test]
fn build_c_to_tensor_2d_nested_literal_matches_eval_output() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("to_tensor_2d.ch");
    let out_dir = dir.path().join("to-tensor-2d-out");
    write_file(
        &path,
        "result = to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]])\n",
    );

    // `chelis check` accepts the rank-2 form.
    let check_output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let check_stdout = String::from_utf8(check_output).expect("check stdout utf8");
    let check_json: serde_json::Value =
        serde_json::from_str(&check_stdout).expect("check stdout is JSON");
    assert_eq!(
        check_json["score"].as_f64(),
        Some(1.0),
        "typer must accept nested-list to_tensor: {check_stdout}",
    );

    // `chelis build --target c` lowers and the generated source is
    // valid C.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Compile + run the generated C and compare its stdout to
    // `chelis eval --file`. With the runtime support in place, both
    // paths must print the same shape and data.
    let eval_stdout = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let eval_text = String::from_utf8(eval_stdout).expect("eval stdout utf8");
    assert!(
        eval_text.contains("shape=[2, 2]"),
        "eval should report rank-2 shape: {eval_text}",
    );

    let status = gcc_link_generated(&out_dir, "to_tensor_2d.c", "to_tensor_2d");
    assert!(status.success(), "gcc failed with status {status}");
    let run_output = StdCommand::new(out_dir.join("to_tensor_2d"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status,
    );
    let run_text = String::from_utf8(run_output.stdout).expect("run stdout utf8");
    // Eval prints `<tensor>\n`, the compiled binary prints
    // `<binding-name> = <tensor>\n`. Match the existing
    // `build_c_runs_top_level_tensor_add_and_matches_eval_output`
    // Issue #912 [05-OBS-6]: both lanes now label, compare directly.
    assert_eq!(
        run_text, eval_text,
        "compiled C binary stdout must equal eval stdout for nested-list to_tensor",
    );
}

/// Bucket 4a regression: `chelis check` and `chelis build --target c`
/// must agree on the shape of `expand(b: tensor[1, f32], 0, count)`.
///
/// The typer is canonical and accepts `[count, 1]` (INSERT semantics) for
/// the linreg-style bias broadcast. The IR evaluator agrees (it consults
/// the IR node's output type). Previously one host function served both
/// spellings and silently picked the same-rank "replicate-singleton"
/// branch when `in_shape[axis] == 1`, producing rank-1 `[count]` instead
/// of the rank-2 `[count, 1]` the typer accepted, the divergence
/// reproduced from the `examples/linreg.ch` shape
/// (`insert(b, 0, 64i64)` over a rank-1 bias). With one result shape per
/// operation the two names route to two host functions and neither
/// guesses (spec/04-type-system.md section 4.7.2).
#[test]
fn build_c_linreg_insert_singleton_bias_keeps_rank2_shape() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("linreg_expand_bias.ch");
    let out_dir = dir.path().join("linreg-expand-bias-out");
    // A rank-1 [1] bias with an axis added at 0 and count 4 must produce
    // rank-2 [4, 1] output. This is the exact shape pattern the
    // `examples/linreg.ch` predict/loss helpers rely on
    // (`insert(b, 0, 64i64)` where `b: tensor[1, f32]`).
    write_file(
        &path,
        "def broadcast_bias(b: tensor[1, f32]) -> tensor[4, 1, f32] = insert(b, 0, 4i64)\n\
         result = broadcast_bias(to_tensor([cast(7.0, f32)]))\n",
    );

    // `chelis check` must accept the rank-2 annotation as canonical.
    let check_output = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    let check_stdout = String::from_utf8(check_output).expect("check stdout utf8");
    let check_json: serde_json::Value =
        serde_json::from_str(&check_stdout).expect("check stdout is JSON");
    assert_eq!(
        check_json["score"].as_f64(),
        Some(1.0),
        "typer must accept the rank-2 expand annotation as canonical: {check_stdout}",
    );

    // `chelis build --target c` must lower without error.
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let generated = fs::read_to_string(out_dir.join("linreg_expand_bias.c")).expect("generated c");
    // Output ndim=2 and a [4, 1] shape literal must appear in the
    // generated allocation; previously the runtime divergence caused
    // the C emit to render the wrong rank.
    assert!(
        generated.contains("(int64_t[]){ 4, 1 }"),
        "expected generated C to allocate rank-2 [4, 1] for the expand result; got:\n{generated}",
    );

    // `chelis test`/`chelis eval` must produce the same shape as the
    // typer (rank-2 [4, 1] with all entries equal to the singleton
    // value). The host-runtime evaluator path is exercised by the
    // companion test `host_runtime_insert_singleton_input_adds_an_axis`
    // in `chelis-compiler-api`; this CLI test pins the typer + C emit
    // legs of the agreement.
    let status = gcc_compile_generated(&out_dir, "linreg_expand_bias.c");
    assert!(
        status.success(),
        "gcc compile of generated C must succeed; status {status}",
    );
}

/// Bucket 4c regression: top-level tensor bindings whose result type
/// carries a polymorphic dim (e.g. `tensor[n, f32]`) must declare every
/// referenced dim in the generated C. Previously a fresh dim variable
/// (`d36`-style autogenerated name) could leak into a
/// `chelis_alloc_view(1, (int64_t[]){ d36 }, ...)` call without a
/// corresponding `int d36 = inputs[k]->shape[axis];` declaration, so
/// the generated C failed to compile with `error: 'd36' undeclared`.
///
/// Since chelis#665 the C emitter declares every such name from its
/// resolved extent ORIGIN (`chelis_ir::axis_sources::dim_extent_origins`)
/// rather than from a search for a `Load` carrying a matching string, and a
/// name that resolves to no origin is a typed receipt. The earlier repair,
/// a sibling sweep in the occurrence walk that panicked on an unrecoverable
/// dim, went with that walk.
#[test]
fn build_c_polymorphic_top_level_tensor_dims_are_declared() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("poly_top_dim.ch");
    let out_dir = dir.path().join("poly-top-dim-out");
    write_file(
        &path,
        "def quadratic[n](theta: tensor[n, f32]) -> tensor[f32] = sum(mul(copy(theta), theta), 0)\n\
         g = grad(quadratic, wrt=theta)(to_tensor([1.0, 2.0, 3.0]))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("poly_top_dim.c")).expect("generated c");
    // Every dim that appears in a `(int64_t[]){ <name>` literal must also
    // appear as an `int64_t <name> = chelis_tensor_shape(inputs[...], <axis>);`
    // declaration. Walk both sets and assert containment.
    let mut used: chelis_unord::UnordSet<String> = chelis_unord::UnordSet::new();
    for line in source.lines() {
        if let Some(after) = line.split("(int64_t[]){ ").nth(1) {
            let name: String = after
                .chars()
                .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
                .collect();
            if !name.is_empty() && !name.chars().all(|ch| ch.is_ascii_digit()) {
                used.insert(name);
            }
        }
    }
    assert!(
        !used.is_empty(),
        "the fixture must retain a symbolic generated-C extent; generated source:\n{source}"
    );
    let mut declared: chelis_unord::UnordSet<String> = chelis_unord::UnordSet::new();
    for line in source.lines() {
        if let Some(idx) = line.find("int64_t ")
            && let Some(rest) = line.get(idx + 8..)
            && rest.contains(" = chelis_tensor_shape(inputs[")
        {
            let name: String = rest
                .chars()
                .take_while(|ch| ch.is_alphanumeric() || *ch == '_')
                .collect();
            if !name.is_empty() {
                declared.insert(name);
            }
        }
    }
    for name in used.to_sorted() {
        assert!(
            declared.contains(name),
            "dim `{name}` used in `(int64_t[]){{ {name} }}` but never declared as \
             `int64_t {name} = chelis_tensor_shape(inputs[...], ...);` -- Bucket 4c symbolic-dim \
             leakage. Generated source:\n{source}",
        );
    }

    // The generated C must link with gcc.
    let status = gcc_link_generated(&out_dir, "poly_top_dim.c", "poly_top_dim");
    assert!(
        status.success(),
        "gcc link failed with status {status}; the polymorphic-dim sweep \
         left an undeclared identifier in the C source.",
    );
}

/// Bucket 4d regression: a higher-order def whose non-callable params
/// and return type are scalar `f32` (e.g. `(model: f32 -> f32, x: f32)
/// -> f32` referencing `model(x)`) must emit its host wrapper
/// definition in the generated C. Previously the predicate at
/// `chelis_ir::host::lower_host_program` blocked the wrapper for any
/// fn with callable params, and the DAG-only path can't represent
/// scalar fn params -- so the def was silently dropped, leaving `gcc`
/// to fail with `implicit declaration of function 'apply'`. The same
/// shape with `tensor[n, f32]` had been working because the
/// non-callable params were tensors and a different code path emitted
/// a tensor-helper wrapper.
#[test]
fn build_c_higher_order_scalar_fn_param_emits_wrapper() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scalar_fn_param.ch");
    let out_dir = dir.path().join("scalar-fn-param-out");
    write_file(
        &path,
        "def apply(model: f32 -> f32, x: f32) -> f32 = model(x)\n\
         def square(y: f32) -> f32 = mul(y, y)\n\
         result = apply(square, cast(3.0, f32))\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("scalar_fn_param.c")).expect("generated c");
    // The wrapper definition must be emitted, not just the prototype.
    // WS-4: every param here is declared `f32`, so the emitted C uses the
    // 4-byte `float` type. Before the precision fix the host lane collapsed
    // these to `double`, silently widening the declared `f32` signature.
    assert!(
        source.contains(&format!(
            "float {}(float (*model)(float), float x) {{",
            authored_c_symbol("apply")
        )) && !source.contains(&format!(
            "static inline float {}(",
            authored_c_symbol("apply")
        )),
        "expected externally linked `apply` wrapper definition in the C source; only a \
         forward declaration would leave gcc with `implicit declaration`. Source:\n{source}",
    );
    // Parity with the tensor case: the same shape with `tensor[n, f32]`
    // already emits the wrapper. Make sure both shapes succeed in this
    // test by also linking + running the binary.
    let status = gcc_link_generated(&out_dir, "scalar_fn_param.c", "scalar_fn_param");
    assert!(status.success(), "gcc link failed with status {status}");
    let run_output = StdCommand::new(out_dir.join("scalar_fn_param"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status,
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    // `square(3) = 9.0`: integral floats keep one fractional digit in the
    // frozen grammar ([05-OBS-2], chelis#732 Phase 2), byte-identical to
    // eval.
    assert_eq!(
        stdout.trim_end(),
        "result = 9.0",
        "compiled binary stdout for `apply(square, 3.0)` must equal `9.0`; got: {stdout:?}",
    );
}

/// Bucket 4e regression: a tensor-valued pipe expression (`xs |> step
/// |> step` for a user-defined `step`, or `softmax(...) |> log |>
/// mul(labels) |> sum(...) |> neg |> mean(...)` from `examples/mnist.ch`)
/// must lower in the C lane to the same value as the equivalent
/// nested-call form. Previously a top-level binding to a pipe whose
/// stages included user-defined fns produced unit-typed C output
/// (`out = ()`) because `lower_host_expr_kind` had no `pipe` arm and
/// fell through to `HostExpr::new(HostExprKind::Unit)`. The fix adds a
/// host-side pipe handler that beta-reduces lambda stages and rewrites
/// var stages into nested-app form, plus an IR-side `lower_pipe`
/// fallthrough fix so unknown vars are routed through
/// `resolve_callable_expr` rather than silently no-op'ing.
#[test]
fn build_c_pipe_into_user_defined_unary_tensor_fn_matches_nested_call() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pipe_user_unary.ch");
    let out_dir = dir.path().join("pipe-user-unary-out");
    write_file(
        &path,
        "def step(x: tensor[3, f32]) -> tensor[3, f32] = mul(copy(x), x)\n\
         out = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]) |> step |> step\n",
    );

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let source = fs::read_to_string(out_dir.join("pipe_user_unary.c")).expect("generated c");
    // The pipe must NOT degrade to a unit-typed binding. Before the
    // fix the generated C contained `int out = __binding_0_value;`
    // and `printf("()")`. After: `chelis_tensor* out = ...` and a
    // `chelis_print_tensor_stdout(out)` call.
    assert!(
        !source.contains("int out = __binding_0_value;") && !source.contains(r#"printf("()");"#),
        "pipe must not produce a unit-typed top-level binding; source:\n{source}",
    );

    let status = gcc_link_generated(&out_dir, "pipe_user_unary.c", "pipe_user_unary");
    assert!(status.success(), "gcc link failed with status {status}");
    let run_output = StdCommand::new(out_dir.join("pipe_user_unary"))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {}",
        run_output.status,
    );
    let stdout = String::from_utf8(run_output.stdout).expect("utf-8 stdout");
    // `step([1, 2, 3]) = [1, 4, 9]`, `step([1, 4, 9]) = [1, 16, 81]`.
    assert_eq!(
        stdout.trim_end(),
        "out = tensor(shape=[3], data=[1.0, 16.0, 81.0])",
        "compiled binary stdout for `xs |> step |> step` must match the \
         nested-call form's value; got: {stdout:?}",
    );
}

// issue #1095 counter-oracle (rt-1103): a higher-order def whose body does
// NOT use its function parameter still owns a real root, and must keep it.
//
// This is the case a signature-shaped rule gets wrong. `ho_a` declares
// `f: tensor[3, f32] -> f32` and never applies it, so lowering never
// reaches the function-parameter placeholder and the body lowers to a
// perfectly good `mul` root. Excluding it on the strength of its type
// deletes real computation: the module is a pure-DAG kernel, so the root
// set IS the kernel's inputs/outputs, and dropping it emits a 0-in/0-out
// stub that aborts at runtime with "expected 0 inputs, got 1" instead of
// computing anything.
//
// The build exits 0 either way, so this drives the emitted kernel and
// pins the value: x * x over [1, 2, 3] must be [1, 4, 9].
#[test]
#[cfg(unix)]
fn build_c_higher_order_def_with_unused_fn_param_keeps_its_kernel() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("only_ho.ch");
    write_file(
        &source,
        "module OnlyHo
         def ho_a(f: tensor[3, f32] -> f32, x: tensor[3, f32]) -> tensor[3, f32] =          mul(copy(x), x)
",
    );

    let out_dir = dir.path().join("only-ho-build");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    // A pure-DAG module emits no `main`, so drive the kernel directly.
    write_file(
        &out_dir.join("driver.c"),
        "#include <stdio.h>\n         #include <string.h>\n         #include \"chelis_runtime.h\"\n         void only_ho(chelis_tensor **inputs, int n_in, chelis_tensor **outputs, int n_out);\n         int main(void) {\n         \x20   int64_t shape[1] = {3};\n         \x20   chelis_tensor* x = chelis_alloc(1, shape, CHELIS_DTYPE_F32);\n         \x20   float xd[3] = {1.0f, 2.0f, 3.0f};\n         \x20   chelis_tensor_write* x_guard = chelis_tensor_begin_write(x);\n         \x20   chelis_write_view x_view = chelis_tensor_write_view(x_guard);\n         \x20   memcpy(x_view.data, xd, sizeof(xd));\n         \x20   chelis_tensor_end_write(x_guard);\n         \x20   chelis_tensor* ins[1] = { x };\n         \x20   chelis_tensor* outs[1] = { NULL };\n         \x20   only_ho(ins, 1, outs, 1);\n         \x20   chelis_read_view out_view = chelis_tensor_read_view(outs[0]);\n         \x20   for (int i = 0; i < 3; i++) printf(\"%.1f\\n\", ((const float *)out_view.data)[i]);\n         \x20   chelis_tensor_release(outs[0]);\n         \x20   chelis_tensor_release(x);\n         \x20   return 0;\n         }\n",
    );

    let status = gcc_link_sources(&out_dir, &["driver.c", "only_ho.c"], "only_ho_driver");
    assert!(status.success(), "gcc link failed for the only_ho driver");
    let run = StdCommand::new("./only_ho_driver")
        .current_dir(&out_dir)
        .output()
        .expect("driver should run");
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&run.stderr).into_owned();
    assert!(
        run.status.success(),
        "only_ho kernel aborted (a dropped root emits a 0-in/0-out stub); \
         status {} stderr: {stderr}",
        run.status,
    );
    assert_eq!(
        stdout.trim_end().lines().collect::<Vec<_>>(),
        ["1.0", "4.0", "9.0"],
        "an unused fn-typed parameter must not cost the def its kernel; got {stdout:?}",
    );
}

// issue #1095, mixed module (rt-1103): the three def shapes must coexist —
// a first-order def, a higher-order def that keeps its root, and a
// higher-order `grad` def that owns none. Getting the subtraction right for
// one shape in isolation is not enough; the declared names are aligned
// against `dag.roots()` POSITIONALLY, so dropping the wrong one still
// balances the count and silently mislabels every later root.
#[test]
#[cfg(unix)]
fn build_c_mixed_module_keeps_working_roots_and_drops_only_the_rootless_grad() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("mixed.ch");
    write_file(
        &source,
        "module Mixed
         def sumsq(theta: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, theta), 0))
         def ho_ignores(f: tensor[3, f32] -> f32, x: tensor[3, f32]) -> tensor[3, f32] =          mul(copy(x), x)
         def grad_sumsq(model: tensor[3, f32] -> f32, theta: tensor[3, f32]) -> tensor[3, f32] = {
         \x20 target = fn (theta_local: tensor[3, f32]) -> model(theta_local)
         \x20 grad(target, wrt=theta_local)(theta)
         }
         theta = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])
         squared = ho_ignores(sumsq, theta)
         dsumsq = grad_sumsq(sumsq, theta)
",
    );

    let out_dir = dir.path().join("mixed-build");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let status = gcc_link_generated(&out_dir, "mixed.c", "mixed");
    assert!(status.success(), "gcc link failed for mixed.c");
    let run = StdCommand::new("./mixed")
        .current_dir(&out_dir)
        .output()
        .expect("compiled binary should run");
    assert!(run.status.success(), "mixed binary failed: {}", run.status);
    let stdout = String::from_utf8(run.stdout).expect("utf-8 stdout");
    assert_eq!(
        stdout.trim_end(),
        "theta = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n\
         squared = tensor(shape=[3], data=[1.0, 4.0, 9.0])\n\
         dsumsq = tensor(shape=[3], data=[2.0, 4.0, 6.0])",
        "the unused-fn-param def must keep computing x * x while the grad \
         def is fixed; got: {stdout:?}",
    );
}

// issue #1095 regression oracle: the grad_quadratic program must lower
// with both of its named roots intact.
//
// This is the same source the #406 leak oracle below builds, but with no
// valgrind dependency. That matters: #406 self-skips when valgrind is
// absent, and CI has no valgrind, so when #1013 routed `build --target c`
// through the pipeline's exact root alignment the resulting
// `lowered root count mismatch: expected 2 named roots, got 1` ran in no
// continuous job. The declared roots here are the two `def`s (`sumsq` and
// `grad_sumsq`), not the `theta` / `dsumsq` host bindings the program
// prints. `grad_sumsq`'s `grad` has no adjoint to build (its callee is a
// function parameter, supplied only at the call site), so it lowers to an
// empty value and owns no root; the lowerer reports that and the declared
// root names subtract it. See `LoweredValue::contributes_no_root`.
//
// Asserting the build succeeds is not enough on its own — a root can be
// dropped and still produce compilable C — so this also runs the linked
// binary and pins the gradient value. d/dtheta sum(theta * theta) is
// 2 * theta, so [1, 2, 3] must yield [2, 4, 6]; a dropped or zeroed
// gradient root shows up here as [0, 0, 0] or a missing line.
#[test]
#[cfg(unix)]
fn build_c_grad_program_keeps_both_named_roots() {
    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("grad_quadratic.ch");
    write_file(
        &source,
        "module GradQuadratic\n\
         def sumsq(theta: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, theta), 0))\n\
         def grad_sumsq(model: tensor[3, f32] -> f32, theta: tensor[3, f32]) -> tensor[3, f32] = {\n\
         \x20 target = fn (theta_local: tensor[3, f32]) -> model(theta_local)\n\
         \x20 grad(target, wrt=theta_local)(theta)\n\
         }\n\
         theta = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])\n\
         dsumsq = grad_sumsq(sumsq, theta)\n",
    );

    let out_dir = dir.path().join("grad-build");
    let assert = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert();
    let output = assert.get_output();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    assert!(
        !stderr.contains("root count mismatch"),
        "grad_quadratic must lower with every named root intact; got: {stderr}",
    );
    assert.success();

    let status = gcc_link_generated(&out_dir, "grad_quadratic.c", "grad_quadratic");
    assert!(status.success(), "gcc link failed for grad_quadratic.c");
    let run = StdCommand::new("./grad_quadratic")
        .current_dir(&out_dir)
        .output()
        .expect("compiled binary should run");
    assert!(
        run.status.success(),
        "compiled binary failed with status {}",
        run.status,
    );
    let stdout = String::from_utf8(run.stdout).expect("utf-8 stdout");
    assert_eq!(
        stdout.trim_end(),
        "theta = tensor(shape=[3], data=[1.0, 2.0, 3.0])\n\
         dsumsq = tensor(shape=[3], data=[2.0, 4.0, 6.0])",
        "grad_quadratic must print both roots with the exact gradient \
         2 * theta; got: {stdout:?}",
    );
}

// issue #406 acceptance oracle: a `chelis build --target c` program
// must not leak the list / tensor temporaries its `main` allocates.
//
// On `origin/main` this program leaked two "definitely lost" blocks
// under valgrind (`chelis_list_from_values` -> the `to_tensor` list, and
// `chelis_alloc` <- `chelis_contiguous` -> the result tensor), both
// reachable from `main`. The fix frees `main`'s owned heap allocations
// at scope exit (see `chelis_backend_c::host_emit::emit_scope_releases`).
//
// This is the authoritative completion oracle for the leak fix: it
// builds the grad_quadratic reproducer to C, gcc-links it against the
// runtime static archive, runs it under
// `valgrind --leak-check=full`, and asserts the
// `definitely lost: 0 bytes` line with NO suppressions.
//
// The OpenMP thread-pool TLS that libgomp allocates via `GOMP_parallel`
// is reported as "possibly lost" by valgrind regardless of chelis code;
// that is a documented libgomp false positive (see issue #406), so this
// oracle asserts on `definitely lost` specifically rather than on
// valgrind's overall exit code.
//
// The C is compiled with `-mavx2` rather than the toolchain default
// `-march=native`: on AVX-512 hosts `-march=native` emits EVEX-encoded
// instructions that valgrind's memcheck core cannot decode, raising a
// spurious SIGILL before the program runs. `-mavx2` keeps the codegen
// vectorized while staying inside valgrind's supported instruction set.
//
// Runs by default where `valgrind` and `gcc` are installed; cleanly
// skips (printing why) otherwise, so a toolchain without valgrind stays
// green. Manual gate to force the leak check locally:
//   cargo test -p chelis-cli --test cli \
//     build_c_grad_program_has_zero_definitely_lost_under_valgrind \
//     -- --nocapture
// Expected success condition: the printed valgrind output contains
// "definitely lost: 0 bytes in 0 blocks".
#[test]
#[cfg(unix)]
fn build_c_grad_program_has_zero_definitely_lost_under_valgrind() {
    fn tool_available(tool: &str) -> bool {
        StdCommand::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    if !tool_available("valgrind") {
        eprintln!("SKIP: valgrind not installed; cannot run the #406 leak oracle");
        return;
    }
    if !tool_available("gcc") {
        eprintln!("SKIP: gcc not installed; cannot link the #406 leak oracle");
        return;
    }

    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("grad_quadratic.ch");
    // grad_quadratic reproducer from issue #406: a tensor `grad` over a
    // function param, with the wrt-tensor built from a `to_tensor([...])`
    // list literal. Exercises both leak frames the issue reported.
    write_file(
        &source,
        "module GradQuadratic\n\
         def sumsq(theta: tensor[3, f32]) -> f32 = tensor_to_scalar(sum(mul(theta, theta), 0))\n\
         def grad_sumsq(model: tensor[3, f32] -> f32, theta: tensor[3, f32]) -> tensor[3, f32] = {\n\
         \x20 target = fn (theta_local: tensor[3, f32]) -> model(theta_local)\n\
         \x20 grad(target, wrt=theta_local)(theta)\n\
         }\n\
         theta = to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)])\n\
         dsumsq = grad_sumsq(sumsq, theta)\n",
    );

    let out_dir = dir.path().join("grad-build");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    // Link directly (not via gcc_link_generated): force -mavx2 in place
    // of the toolchain's default -march=native so valgrind can decode
    // every instruction. -fopenmp matches the runtime archive's OpenMP
    // dependency; without it the link fails on unresolved GOMP symbols.
    let bin = out_dir.join("grad_quadratic");
    let link = StdCommand::new("gcc")
        .current_dir(&out_dir)
        .args([
            "-O2",
            "-mavx2",
            "-fopenmp",
            "grad_quadratic.c",
            "libchelis_runtime.a",
            "-lm",
            "-lpthread",
            "-ldl",
            "-o",
            bin.to_str().unwrap(),
        ])
        .output()
        .expect("gcc should run");
    assert!(
        link.status.success(),
        "gcc link failed:\n{}",
        String::from_utf8_lossy(&link.stderr)
    );

    let valgrind = StdCommand::new("valgrind")
        .current_dir(&out_dir)
        .args([
            "--leak-check=full",
            "--errors-for-leak-kinds=definite",
            bin.to_str().unwrap(),
        ])
        .output()
        .expect("valgrind should run");

    let vg_stdout = String::from_utf8_lossy(&valgrind.stdout);
    let vg_stderr = String::from_utf8_lossy(&valgrind.stderr);
    println!("valgrind stdout:\n{vg_stdout}");
    println!("valgrind stderr:\n{vg_stderr}");

    // The program itself must produce the correct gradient before we
    // care about leaks.
    assert!(
        vg_stdout.contains("dsumsq = tensor(shape=[3], data=[2.0, 4.0, 6.0])"),
        "grad_quadratic produced wrong output under valgrind:\n{vg_stdout}"
    );

    // The acceptance contract: zero bytes definitely lost, no
    // suppressions. valgrind prints the leak summary to stderr.
    assert!(
        vg_stderr.contains("definitely lost: 0 bytes in 0 blocks"),
        "chelis-built C program must have zero definitely-lost bytes \
         under valgrind (issue #406); valgrind reported:\n{vg_stderr}"
    );
    assert!(
        vg_stderr.contains("suppressed: 0 bytes in 0 blocks"),
        "the #406 leak oracle must run with NO suppressions; \
         valgrind reported:\n{vg_stderr}"
    );
}

// issue #943: the emitted `map`/`filter` accumulation loops used to
// rebuild the result list per element (`target = chelis_list_append(
// target, …)`) without releasing the predecessor — Θ(n²) allocation,
// with every intermediate generation unreachable at exit ("definitely
// lost"). The fix accumulates in place (`chelis_list_with_capacity` +
// `chelis_list_push_moved`), so a combinator pipeline must now run leak-free.
// Elements are floats. Heap-payload elements are covered by the
// ownership-ledger corpus in chelis-compiler-api's
// `issue_2508_list_step_ownership`: the linear leak once attributed to
// `chelis_list_index` was the loop step cloning an item it had been
// moved (chelis#2332, chelis#2508).
//
// Same toolchain gating and no-suppression contract as the #406 oracle
// above. Registered in `docs/manual_gates.md`; manual gate:
//   cargo test -p chelis-cli --test cli \
//     build_c_list_combinator_program_has_zero_definitely_lost_under_valgrind \
//     -- --ignored --nocapture
//
// `#[ignore]` rather than the #406 sibling's bare self-skip: valgrind is
// installed in no CI job (`grep -rniE valgrind .github/workflows/` is
// empty) and is unavailable on macOS arm64, so on every machine that
// currently runs the suite the tool-availability `return` below made
// nextest report **PASS in ~0.008s** for a leak oracle that never ran.
// A green result for an unexecuted check is worse than an honest skip,
// especially for the standing guard on this change set's central claim.
// `docs/manual_gates.md` states the contract this now satisfies: "If a
// test is `#[ignore]`'d, it must appear here with its full command and
// prerequisite."
//
// The tool-availability check is kept as a second line of defence so the
// documented `--ignored` command still explains itself on a box without
// valgrind rather than failing obscurely. The #406 sibling
// (`cli.rs`, `build_c_leak_program_has_zero_definitely_lost_under_valgrind`)
// and the other five #406-era oracles still self-skip and report a false
// PASS; converting them is out of scope here and tracked separately.
#[test]
#[ignore = "requires valgrind + gcc; registered in docs/manual_gates.md (chelis#2333)"]
#[cfg(unix)]
fn build_c_list_combinator_program_has_zero_definitely_lost_under_valgrind() {
    fn tool_available(tool: &str) -> bool {
        StdCommand::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    if !tool_available("valgrind") {
        eprintln!("SKIP: valgrind not installed; cannot run the #943 leak oracle");
        return;
    }
    if !tool_available("gcc") {
        eprintln!("SKIP: gcc not installed; cannot link the #943 leak oracle");
        return;
    }

    let dir = tempdir().expect("tempdir");
    let source = dir.path().join("combinators.ch");
    write_file(
        &source,
        "module Combinators\n\
         vals = map(fn (x) -> mul(x, 2.0f64), [1.0f64, 2.0f64, 3.0f64, 4.0f64])\n\
         kept = filter(fn (x) -> gt(x, 3.0f64), vals)\n\
         total = fold(fn (a, x) -> add(a, x), 0.0f64, kept)\n",
    );

    let out_dir = dir.path().join("combinators-build");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            source.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let bin = out_dir.join("combinators");
    let link = StdCommand::new("gcc")
        .current_dir(&out_dir)
        .args([
            "-O2",
            "-mavx2",
            "-fopenmp",
            "combinators.c",
            "libchelis_runtime.a",
            "-lm",
            "-lpthread",
            "-ldl",
            "-o",
            bin.to_str().unwrap(),
        ])
        .output()
        .expect("gcc should run");
    assert!(
        link.status.success(),
        "gcc link failed:\n{}",
        String::from_utf8_lossy(&link.stderr)
    );

    let valgrind = StdCommand::new("valgrind")
        .current_dir(&out_dir)
        .args([
            "--leak-check=full",
            "--errors-for-leak-kinds=definite",
            bin.to_str().unwrap(),
        ])
        .output()
        .expect("valgrind should run");

    let vg_stdout = String::from_utf8_lossy(&valgrind.stdout);
    let vg_stderr = String::from_utf8_lossy(&valgrind.stderr);
    println!("valgrind stdout:\n{vg_stdout}");
    println!("valgrind stderr:\n{vg_stderr}");

    assert!(
        vg_stdout.contains("total = 18.0"),
        "combinator pipeline produced wrong output under valgrind:\n{vg_stdout}"
    );
    let no_definitely_lost = vg_stderr.contains("definitely lost: 0 bytes in 0 blocks")
        || vg_stderr.contains("All heap blocks were freed -- no leaks are possible");
    assert!(
        no_definitely_lost,
        "chelis-built map/filter program must have zero definitely-lost \
         bytes under valgrind (issue #943); valgrind reported:\n{vg_stderr}"
    );
    let no_suppressed_errors = vg_stderr.contains("suppressed: 0 bytes in 0 blocks")
        || vg_stderr.contains("(suppressed: 0 from 0)");
    assert!(
        no_suppressed_errors,
        "the #943 leak oracle must run with NO suppressions; \
         valgrind reported:\n{vg_stderr}"
    );
}

// chelis#928: the Std.Io serializers must stay compiled-lane compatible.
// The chelis-std self-test corpus runs under `chelis test` (the eval lane
// only), so a std source change that type-checks and evals but cannot
// LOWER — the [05-UNS-1]/chelis#730 class; e.g. Option-in-List composites,
// which the C host lane cannot resolve — ships green unless something
// builds a program importing the new surface. This test is that something:
// it stages a reef project that writes a CSV and a JSON document through
// Std.Io, `chelis build`s it, links with the system toolchain, runs the
// binary, and asserts both emitted documents byte-exactly (including a
// 17-significant-digit f64 that must survive shortest-round-trip).
#[test]
#[cfg(unix)]
fn build_c_program_using_std_io_serializers_emits_exact_documents() {
    let dir = tempdir().expect("tempdir");
    let proj = dir.path().join("proj");
    fs::create_dir_all(proj.join("src")).expect("mkdir proj/src");
    fs::write(
        proj.join("reef.toml"),
        format!(
            "[package]\nname = \"t\"\nversion = \"0.1.0\"\ncompiler = \"={}\"\n\
             module_prefix = \"T\"\n\n[dependencies]\nchelis-std = {{ version = \"0.4.0\" }}\n",
            env!("CARGO_PKG_VERSION")
        ),
    )
    .expect("write reef.toml");

    let csv_path = dir.path().join("out.csv");
    let json_path = dir.path().join("out.json");
    fs::write(
        proj.join("src/main.ch"),
        format!(
            "module T.Main\n\
             import Std.Io.Csv (write_csv, read_csv)\n\
             import Std.Io.Json (Json, JsonFloat, JsonString, JsonInt, JsonObject, write_json)\n\
             rows = [dict_of([(\"k\", \"a,b\"), (\"price\", \"7773.015187\")]), dict_of([(\"k\", \"he said \\\"hi\\\"\"), (\"price\", \"0.15110743269565682\")])]\n\
             done_csv = write_csv(\"{csv}\", rows)\n\
             doc = JsonObject(dict_of([(\"cap_price\", JsonFloat(0.15110743269565682f64, \"0.15110743269565682\")), (\"name\", JsonString(\"a\\\"b\\\\c\")), (\"n\", JsonInt(cast(3, i64)))]))\n\
             done_json = write_json(\"{json}\", doc)\n\
             back = read_csv(\"{csv}\")\n\
             n = len(back)\n",
            csv = csv_path.display(),
            json = json_path.display()
        ),
    )
    .expect("write main.ch");

    let out_dir = dir.path().join("ser-build");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", dir.path().join("reef-home"))
        .current_dir(&proj)
        .args([
            "build",
            "--emit-c",
            "src/main.ch",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let status = gcc_link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "linking the serializer program failed");

    let run = StdCommand::new(out_dir.join("main"))
        .output()
        .expect("serializer program should run");
    assert!(run.status.success(), "serializer program exited nonzero");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("n = 2"),
        "compiled read-back of the written CSV must see 2 rows:\n{stdout}"
    );

    let json_text = fs::read_to_string(&json_path).expect("out.json written");
    assert_eq!(
        json_text, "{\"cap_price\":0.15110743269565682,\"n\":3,\"name\":\"a\\\"b\\\\c\"}",
        "compiled write_json output must be byte-exact"
    );
    let csv_text = fs::read_to_string(&csv_path).expect("out.csv written");
    assert_eq!(
        csv_text, "k,price\n\"a,b\",7773.015187\n\"he said \"\"hi\"\"\",0.15110743269565682\n",
        "compiled write_csv output must be byte-exact"
    );
}

// issue #406 (complete): the sibling leaks #412 did not reach. #412 freed
// the heap temporaries the program-root `main` allocates, but two more
// "definitely lost" classes of the same shape survived, each via a
// `chelis_*_from_values` allocation `main` never released transitively:
//
//   1. A heap host value (tuple / list / dict / adt / string) built as an
//      *intermediate* inside a compiled function body — e.g. a `let`-bound
//      `p = (a, b)` consumed by the body — was never released at the
//      function's block exit. #412 only enabled scope-release tracking for
//      `emit_main`, whose flat-scope guard deliberately skips the deeper
//      indents a function body's `{ ... }` block introduces.
//
//   2. A tuple field that is itself a heap container (a *nested* tuple /
//      list / adt) leaked at the labeled-root printer: `chelis_tuple_get`
//      retains the boxed element it returns, and the printer only read it.
//
// The fix releases a compiled `let` block's heap bindings at block close,
// retaining any binding the block result aliases so the caller keeps one
// reference (`chelis_backend_c::host_emit`'s `LetReleaseScope` /
// `retain_transferred_result`), and releases the retained tuple-field
// handle after the labeled-root printer reads it.
//
// Both reproducers below leaked under valgrind on the post-#412 tree and
// are clean after the fix. They share the #412 oracle's harness contract:
// build to C, gcc-link `-mavx2` against the runtime archive, run under
// `valgrind --leak-check=full` with NO suppressions, and assert
// `definitely lost: 0 bytes`. (`-mavx2` keeps codegen inside valgrind's
// decodable instruction set on AVX-512 hosts; the libgomp `GOMP_parallel`
// TLS is a documented "possibly lost" false positive, so the contract is
// on `definitely lost` specifically.) Run by default where `valgrind` and
// `gcc` exist; cleanly skip (printing why) otherwise. Manual gate:
//   cargo test -p chelis-cli --test cli \
//     build_c_function_body_heap_temp_has_zero_definitely_lost_under_valgrind \
//     -- --nocapture
//   cargo test -p chelis-cli --test cli \
//     build_c_nested_tuple_print_has_zero_definitely_lost_under_valgrind \
//     -- --nocapture
// Expected success condition: the printed valgrind output contains
// "definitely lost: 0 bytes in 0 blocks".
#[cfg(unix)]
fn assert_built_c_has_zero_definitely_lost(name: &str, source: &str, expected_stdout: &[&str]) {
    fn tool_available(tool: &str) -> bool {
        StdCommand::new(tool)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    }

    if !tool_available("valgrind") {
        eprintln!("SKIP: valgrind not installed; cannot run the #406 leak oracle");
        return;
    }
    if !tool_available("gcc") {
        eprintln!("SKIP: gcc not installed; cannot link the #406 leak oracle");
        return;
    }

    let dir = tempdir().expect("tempdir");
    let src = dir.path().join(format!("{name}.ch"));
    write_file(&src, source);

    let out_dir = dir.path().join(format!("{name}-build"));
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let bin = out_dir.join(name);
    let link = StdCommand::new("gcc")
        .current_dir(&out_dir)
        .args([
            "-O2",
            "-mavx2",
            "-fopenmp",
            &format!("{name}.c"),
            "libchelis_runtime.a",
            "-lm",
            "-lpthread",
            "-ldl",
            "-o",
            bin.to_str().unwrap(),
        ])
        .output()
        .expect("gcc should run");
    assert!(
        link.status.success(),
        "gcc link failed:\n{}",
        String::from_utf8_lossy(&link.stderr)
    );

    let valgrind = StdCommand::new("valgrind")
        .current_dir(&out_dir)
        .args([
            "--leak-check=full",
            "--errors-for-leak-kinds=definite",
            bin.to_str().unwrap(),
        ])
        .output()
        .expect("valgrind should run");

    let vg_stdout = String::from_utf8_lossy(&valgrind.stdout);
    let vg_stderr = String::from_utf8_lossy(&valgrind.stderr);
    println!("valgrind stdout:\n{vg_stdout}");
    println!("valgrind stderr:\n{vg_stderr}");

    // The program must produce its correct output before leaks matter.
    for expected in expected_stdout {
        assert!(
            vg_stdout.contains(expected),
            "{name} produced wrong output under valgrind; \
             expected to contain {expected:?}:\n{vg_stdout}"
        );
    }

    assert!(
        vg_stderr.contains("definitely lost: 0 bytes in 0 blocks"),
        "chelis-built C program {name} must have zero definitely-lost \
         bytes under valgrind (issue #406); valgrind reported:\n{vg_stderr}"
    );
    assert!(
        vg_stderr.contains("suppressed: 0 bytes in 0 blocks"),
        "the #406 leak oracle must run with NO suppressions; \
         valgrind reported:\n{vg_stderr}"
    );
}

#[test]
#[cfg(unix)]
fn build_c_function_body_heap_temp_has_zero_definitely_lost_under_valgrind() {
    // A tuple built as an intermediate inside a compiled function body and
    // consumed by the body (not returned): leaked the tuple via
    // `chelis_tuple_from_values` on the post-#412 tree.
    assert_built_c_has_zero_definitely_lost(
        "fn_body_tuple_temp",
        "def mk(x: f32, y: f32) -> f32 = {\n\
         \x20 p = (mul(x, 2.0), add(y, 1.0))\n\
         \x20 add(p.0, p.1)\n\
         }\n\
         out = mk(3.0, 4.0)\n",
        &["out = 11"],
    );
}

#[test]
#[cfg(unix)]
fn build_c_function_body_list_temp_has_zero_definitely_lost_under_valgrind() {
    // The list sibling of the tuple intermediate: a `let`-bound list built
    // and consumed inside a function body leaked via
    // `chelis_list_from_values`. Confirms the block-scope release covers
    // every refcounted host-value type, not just tuples.
    assert_built_c_has_zero_definitely_lost(
        "fn_body_list_temp",
        "def mk(n: i64) -> i64 = {\n\
         \x20 xs = [n, mul(n, cast(2, i64)), mul(n, cast(3, i64))]\n\
         \x20 len(xs)\n\
         }\n\
         out = mk(cast(5, i64))\n",
        &["out = 3"],
    );
}

#[test]
#[cfg(unix)]
fn build_c_function_body_tuple_transfer_has_zero_definitely_lost_under_valgrind() {
    // Negative-direction parity: the block result *escapes* by aliasing a
    // heap binding through an `if`/`else` (the unchosen branch's tuple was
    // leaking, and a naive release of both arms would use-after-free the
    // chosen one). Asserts both correct output and zero definitely-lost,
    // so a regression that double-frees or leaks here is caught.
    assert_built_c_has_zero_definitely_lost(
        "fn_body_tuple_transfer",
        "def mk(x: f32, y: f32) -> (f32, f32) = {\n\
         \x20 p = (mul(x, 2.0), add(y, 1.0))\n\
         \x20 q = (add(x, 1.0), mul(y, 2.0))\n\
         \x20 if gt(x, 0.0) then p else q\n\
         }\n\
         out = mk(3.0, 4.0)\n",
        &["out.0 = 6", "out.1 = 5"],
    );
}

#[test]
#[cfg(unix)]
fn build_c_nested_tuple_print_has_zero_definitely_lost_under_valgrind() {
    // A nested tuple returned and printed: the labeled-root printer's
    // `chelis_tuple_get` retained each nested-container field and never
    // released it, leaking the inner tuples for the process lifetime.
    assert_built_c_has_zero_definitely_lost(
        "nested_tuple_print",
        "def mk(x: f32, y: f32) -> ((f32, f32), (f32, f32)) = {\n\
         \x20 p = (mul(x, 2.0), add(y, 1.0))\n\
         \x20 q = (add(x, 1.0), mul(y, 2.0))\n\
         \x20 (p, q)\n\
         }\n\
         out = mk(3.0, 4.0)\n",
        &["out.0.0 = 6", "out.1.1 = 8"],
    );
}

#[test]
#[cfg(unix)]
fn build_c_call_return_tuple_escape_has_zero_definitely_lost_under_valgrind() {
    // Issue #406 call-escape: a block-frame heap binding that escapes the
    // block *through a function-call return* (not a bare `__result = p`)
    // was freed by the block release while the returned alias still lived
    // -- a use-after-free that crashed a correct program (`expected
    // float64 value`, exit 1; under valgrind: invalid reads + a 56-byte
    // definitely-lost block from the prematurely-dropped tuple).
    //
    // `id_pair` returns its bare argument, so the analysis
    // (`analyze_returns_arg`) flags arg 0 as escaping; the emit site
    // retains the call result before the block releases `p`, leaving
    // exactly one live reference. Asserts both correct output and zero
    // definitely-lost: a regression reintroduces the UAF (wrong output /
    // crash) or, if the retain is unbalanced, a leak.
    //
    // origin/main has no function-body block release at all (the #406 fix
    // lives only on this branch), so this oracle is meaningful as a GREEN
    // assertion on the branch; there is no red-on-main counterpart.
    assert_built_c_has_zero_definitely_lost(
        "call_return_tuple_escape",
        "def id_pair(p: (f32, f32)) -> (f32, f32) = p\n\
         def mk(x: f32) -> (f32, f32) = {\n\
         \x20 p = (mul(x, 2.0), add(x, 1.0))\n\
         \x20 id_pair(p)\n\
         }\n\
         out = mk(3.0)\n",
        &["out.0 = 6", "out.1 = 4"],
    );
}

#[test]
#[cfg(unix)]
fn build_c_call_return_list_escape_has_zero_definitely_lost_under_valgrind() {
    // The list sibling of the call-return tuple escape: a `let`-bound list
    // returned through `id2(xs)` SEGFAULTED on the branch before the fix
    // (exit 139) because the block released `xs` while the returned alias
    // still pointed at it. The same `analyze_returns_arg` + retain-on-
    // call-escape path covers every refcounted host-value type, so the
    // list case must be correct and definitely-lost-free too.
    assert_built_c_has_zero_definitely_lost(
        "call_return_list_escape",
        "def id2(a: List[i64]) -> List[i64] = a\n\
         def mk(n: i64) -> List[i64] = {\n\
         \x20 xs = [n, mul(n, cast(2, i64)), mul(n, cast(3, i64))]\n\
         \x20 id2(xs)\n\
         }\n\
         out = mk(cast(5, i64))\n",
        &["out = [5, 10, 15]"],
    );
}

#[test]
#[cfg(unix)]
fn build_c_call_fresh_result_does_not_over_retain_under_valgrind() {
    // Precision guard for the #406 call-escape retain: a call whose callee
    // builds a *fresh* result (does not return its argument) must NOT
    // retain the block result, or the block-frame binding it was built
    // from leaks. `dup` constructs a new list, so `analyze_returns_arg`
    // reports it returns none of its parameters and the emit site retains
    // nothing -- the block release of `xs` then balances its construction
    // and the program is definitely-lost-free. A naive
    // conservative-retain-everything fallback would leak the `xs`
    // allocation here; this oracle locks the precise interprocedural
    // behavior so a later regression to over-retain is caught as a leak.
    assert_built_c_has_zero_definitely_lost(
        "call_fresh_result_no_over_retain",
        "def dup(a: List[i64]) -> List[i64] = [len(a), len(a)]\n\
         def mk(n: i64) -> List[i64] = {\n\
         \x20 xs = [n, mul(n, cast(2, i64)), mul(n, cast(3, i64))]\n\
         \x20 dup(xs)\n\
         }\n\
         out = mk(cast(5, i64))\n",
        &["out = [3, 3]"],
    );
}

// The RT792 probes for the Phase 1 interim stdout comparator were
// retired with the comparator itself (chelis#732 Phase 2): byte equality
// reports every one of their divergence classes as a raw diff by
// construction, and the accept-classes (int-vs-float renders, the rank-0
// wrapper) no longer exist because the lanes render identically.
