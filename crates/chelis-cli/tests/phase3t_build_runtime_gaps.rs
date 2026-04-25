//! Phase 3t.A1 follow-up — build-path coverage for Std modules whose host
//! evaluator path is incomplete (#38).
//!
//! Several `Std.Nn` / `Std.Loss` modules use ops the host runtime does not
//! support: `matmul`, `expand`, `softmax`, `sum` reductions, etc. Those
//! modules type-check and lower to IR cleanly, and the C-build target emits
//! correct kernels for them, but `chelis eval` (the host runtime) returns an
//! error before the test bodies run. As a consequence Std self-tests for
//! those modules can only assert importability + typecheck, not behavior.
//!
//! As of v0.2.4 + N2 fix (Nautilus upstream report), the host runtime gained
//! `matmul`, `permute`, and `sum` evaluators. The remaining gaps for these
//! fixtures are:
//!
//!   * `Linear.forward` — uses `expand` (host runtime: unsupported)
//!   * `Attention.scaled_dot_product_attention` — uses `softmax`
//!   * `CrossEntropy.loss` — uses `softmax`
//!   * `RmsNorm.forward` — uses `to_list` + `map` (already eval-clean)
//!
//! Closing the remaining `expand` / `softmax` gaps would let these run under
//! `chelis eval` end-to-end. Self-tests for the matmul/permute/sum surface
//! live in `packages/chelis-std/tests/runtime/`.
//!
//! These integration tests close that gap on the build path: stage
//! chelis-std into a tempdir reef home, write a `main.ch` that calls the
//! module with concrete inputs, run `chelis build --target c`, compile the
//! generated C with `gcc`, run the binary, and assert on stdout. If the
//! eval lane later gains support for the missing ops, the same fixtures
//! can be reused to cross-check eval vs. build outputs.
//!
//! Pattern mirrors `phase3i_std::reef_std_generate_builds_and_runs_compiled_program`.

use assert_cmd::Command;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn package_std() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../packages/chelis-std")
        .canonicalize()
        .expect("chelis-std package must exist")
}

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn copy_dir_recursive(src: &Path, dst: &Path) {
    fs::create_dir_all(dst).expect("create dir");
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

fn gcc_link_generated(out_dir: &Path, source: &str, binary: &str) -> std::process::ExitStatus {
    let needs_blas = fs::read_to_string(out_dir.join(source))
        .map(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
        .unwrap_or(false);
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas,
        },
    );
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

fn make_app(dir_name: &str) -> (tempfile::TempDir, PathBuf, PathBuf) {
    let dir = tempdir().expect("tempdir");
    let reef_home = dir.path().join("reef-home");
    let std_pkg = dir.path().join("chelis-std");
    let app_pkg = dir.path().join(dir_name);
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
        &format!(
            r#"[package]
name = "{dir_name}"
version = "0.1.0"
compiler = "=0.2.5"
module_prefix = "Demo"

[dependencies]
chelis-std = {{ version = "0.1.0" }}
"#
        ),
    );
    (dir, reef_home, app_pkg)
}

/// Run `chelis build --target c`, compile with gcc, run the binary, and
/// return its stdout as a String. Asserts each step succeeded.
fn build_and_run(reef_home: &Path, app_pkg: &Path) -> String {
    let out_dir = app_pkg.join("out");
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_REEF_HOME", reef_home)
        .current_dir(app_pkg)
        .args([
            "build",
            app_pkg.join("src/main.ch").to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .assert()
        .success();

    let status = gcc_link_generated(&out_dir, "main.c", "main");
    assert!(status.success(), "gcc failed with status {status}");

    let run_output = StdCommand::new(out_dir.join("main"))
        .current_dir(app_pkg)
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "compiled binary failed with status {} (stderr: {})",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr)
    );
    String::from_utf8(run_output.stdout).expect("compiled stdout should be utf-8")
}

#[test]
fn reef_std_linear_forward_builds_and_produces_expected_output() {
    // Linear.forward(x, w, b) = matmul(x, w) + expand(b, batch).
    // Pick a 1×2 input, 2×2 identity weights, and a 2-vector bias so the
    // hand-computable output is x + b broadcast across the (single) batch:
    //
    //   x = [[3.0, 5.0]]
    //   w = [[1.0, 0.0], [0.0, 1.0]]
    //   b = [10.0, 100.0]
    //   forward(x, w, b) = [[3+10, 5+100]] = [[13.0, 105.0]]
    //
    // Eval cannot run this — `matmul` and `expand` are not in the host
    // runtime — so the build path is the only way to exercise it from a
    // Rust integration test today.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-linear-build");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Linear (forward)

x = (pad_sequences_to([[cast(3.0, f32), cast(5.0, f32)]], cast(2, int64), cast(0.0, f32)) : tensor[1, 2, f32])
w = (pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)], [cast(0.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32)) : tensor[2, 2, f32])
b = (to_tensor([cast(10.0, f32), cast(100.0, f32)]) : tensor[2, f32])
y = forward(x, w, b)
"#,
    );

    let stdout = build_and_run(&reef_home, &app_pkg);
    // The compiled binary prints each top-level binding. The exact format
    // we rely on is the rendered tensor body; assert presence of the two
    // expected element values to keep the test stable across formatter
    // tweaks.
    assert!(
        stdout.contains("13") && stdout.contains("105"),
        "expected matmul+bias output to contain 13 and 105; got:\n{stdout}"
    );
}

#[test]
fn reef_std_attention_scaled_dot_product_builds_and_produces_expected_output() {
    // scaled_dot_product_attention(q, k, v, scale):
    //   kt      = permute(k, 1, 0)
    //   scores  = matmul(q, kt)
    //   scaled  = mul(scores, scale)
    //   weights = softmax(scaled, -1)
    //   out     = matmul(weights, v)
    //
    // Pin a row-uniform case: q and k both zero, scale all-ones. scores = 0,
    // scaled = 0, softmax(0,0,0,0) = [0.25, 0.25, 0.25, 0.25] per row.
    // With v = [[1,1,1,1], [2,2,2,2], [3,3,3,3], [4,4,4,4]] each output
    // row is the uniform mean of v's rows = [2.5, 2.5, 2.5, 2.5].
    //
    // The body uses matmul / softmax / permute which the host evaluator
    // doesn't support; build path is the only way to exercise this end-to-end.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-attention-build");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.Attention (scaled_dot_product_attention)

zero_row = [cast(0.0, f32), cast(0.0, f32), cast(0.0, f32), cast(0.0, f32)]
ones_row = [cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)]

q = (pad_sequences_to([zero_row, zero_row, zero_row, zero_row], cast(4, int64), cast(0.0, f32)) : tensor[4, 4, f32])
k = (pad_sequences_to([zero_row, zero_row, zero_row, zero_row], cast(4, int64), cast(0.0, f32)) : tensor[4, 4, f32])
v = (pad_sequences_to([
  [cast(1.0, f32), cast(1.0, f32), cast(1.0, f32), cast(1.0, f32)],
  [cast(2.0, f32), cast(2.0, f32), cast(2.0, f32), cast(2.0, f32)],
  [cast(3.0, f32), cast(3.0, f32), cast(3.0, f32), cast(3.0, f32)],
  [cast(4.0, f32), cast(4.0, f32), cast(4.0, f32), cast(4.0, f32)]
], cast(4, int64), cast(0.0, f32)) : tensor[4, 4, f32])
scale = (pad_sequences_to([ones_row, ones_row, ones_row, ones_row], cast(4, int64), cast(0.0, f32)) : tensor[4, 4, f32])
attn = scaled_dot_product_attention(q, k, v, scale)
"#,
    );

    let stdout = build_and_run(&reef_home, &app_pkg);
    // Every output element ≈ 2.5. Match the leading three significant digits.
    assert!(
        stdout.contains("2.5"),
        "expected attention output to contain 2.5 (uniform-mean of v); got:\n{stdout}"
    );
}

#[test]
fn reef_std_crossentropy_loss_builds_and_produces_expected_output() {
    // CrossEntropy.loss(logits, labels) =
    //   softmax(logits, 1) |> log |> mul(labels) |> sum(1) |> neg
    //
    // For batch=1, classes=2:
    //   logits = [[2.0, 1.0]]
    //   softmax = [exp(2)/(exp(2)+exp(1)), exp(1)/(exp(2)+exp(1))]
    //           = [0.7310585, 0.2689414]
    //   log     = [-0.3132617, -1.3132617]
    //   labels  = [[1.0, 0.0]]   (one-hot on class 0)
    //   prod    = [-0.3132617, 0.0]
    //   sum/-   = 0.3132617
    //
    // The body uses `softmax` and `sum` reductions which are not in the host
    // evaluator runtime, so eval would crash. Build path must produce
    // ≈ 0.3132 (i.e. -log(softmax)_correct_class).
    let (_dir, reef_home, app_pkg) = make_app("phase3t-crossentropy-build");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Loss.CrossEntropy (loss)

logits = (pad_sequences_to([[cast(2.0, f32), cast(1.0, f32)]], cast(2, int64), cast(0.0, f32)) : tensor[1, 2, f32])
labels = (pad_sequences_to([[cast(1.0, f32), cast(0.0, f32)]], cast(2, int64), cast(0.0, f32)) : tensor[1, 2, f32])
nll = loss(logits, labels)
"#,
    );

    let stdout = build_and_run(&reef_home, &app_pkg);
    // Expected 0.3132. Match the leading three significant digits.
    assert!(
        stdout.contains("0.313"),
        "expected cross-entropy loss to contain 0.313; got:\n{stdout}"
    );
}

#[test]
fn reef_std_rmsnorm_forward_builds_and_runs() {
    // RmsNorm.forward(x, gain, eps) normalizes by sqrt(mean(x^2) + eps).
    // For x = [3, 4]:
    //   mean(x^2) = (9 + 16) / 2 = 12.5
    //   1 / sqrt(12.5 + 1e-4) ≈ 0.2828
    //   y = x * 0.2828 = [0.8485, 1.1314]
    // gain = [1, 1] keeps the scale unchanged.
    //
    // The body uses `to_list` + `map` which works under eval, but pinning
    // the build path matters: rmsnorm shows up as an op in compiled
    // attention blocks and we want to lock the lowering against silent
    // regressions.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-rmsnorm-build");
    write_file(
        &app_pkg.join("src/main.ch"),
        r#"module Demo.Main

import Std.Nn.RmsNorm (forward)

x = (to_tensor([cast(3.0, f32), cast(4.0, f32)]) : tensor[2, f32])
gain = (to_tensor([cast(1.0, f32), cast(1.0, f32)]) : tensor[2, f32])
y = forward(x, gain, cast(0.0001, f32))
"#,
    );

    let stdout = build_and_run(&reef_home, &app_pkg);
    // Expected ≈ [0.8485, 1.1314]. Match the leading three significant
    // digits so the assert survives float-printing variations across libcs.
    assert!(
        stdout.contains("0.848") && stdout.contains("1.131"),
        "expected rmsnorm output to contain 0.848 and 1.131; got:\n{stdout}"
    );
}
