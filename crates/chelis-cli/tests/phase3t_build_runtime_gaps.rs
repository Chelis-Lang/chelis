//! Phase 3t.A1 follow-up — build-path coverage for Std modules whose host
//! evaluator path is incomplete (#38).
//!
//! Several `Std.Nn` / `Std.Loss` modules use ops the host runtime does not
//! support: `matmul`, `expand`, `softmax`, `sum` reductions, etc. Those
//! modules type-check and lower to IR cleanly, and the C-build target emits
//! correct kernels for them, but `chelis eval` (the host runtime) returned an
//! error before the test bodies could run. As a consequence Std self-tests
//! for those modules could only assert importability + typecheck, not
//! behavior.
//!
//! As of v0.2.5 + N2 fix and the follow-up `expand`/`softmax` host wiring,
//! the host runtime now supports `matmul`, `permute`, `sum`, `expand`, and
//! `softmax`.
//!
//!   * `Linear.forward` — uses `expand` (host runtime now supports the op
//!     itself, but the dim-generic `batch` parameter still cannot be
//!     resolved at host-evaluator runtime, so end-to-end eval still fails
//!     with `unknown runtime name 'batch'`)
//!   * `Attention.scaled_dot_product_attention` — fully eval-clean as of
//!     this commit; cross-checked under `chelis test` in
//!     `packages/chelis-std/tests/runtime/attention_eval_cross.ch`
//!   * `CrossEntropy.loss` — `softmax` works, but the host evaluator's
//!     `log`/`exp` builtins still only accept scalar floats; piping a
//!     tensor through `log` errors with `float op expects float arg`.
//!     Out of scope for this fix.
//!   * `RmsNorm.forward` — uses `to_list` + `map` (already eval-clean)
//!
//! Self-tests for the `matmul` / `permute` / `sum` / `expand` / `softmax`
//! host evaluator surface live in `packages/chelis-std/tests/runtime/`.
//! Closing the remaining `log`-on-tensor and dim-generic-resolution gaps
//! would let `Linear.forward` and `CrossEntropy.loss` run end-to-end under
//! `chelis eval` too. Tracked separately from the N2 / expand-softmax
//! work — those gaps predate this fix.
//!
//! These integration tests close that gap on the build path: stage
//! chelis-std into a tempdir reef home, write a `main.ch` that calls the
//! module with concrete inputs, run `chelis build --target c`, compile the
//! generated C with `gcc`, run the binary, and assert on stdout. The
//! attention fixture now has an eval-lane sibling at
//! `packages/chelis-std/tests/runtime/attention_eval_cross.ch` that
//! confirms eval-vs-build agreement on the uniform case.
//!
//! Pattern mirrors `phase3i_std::reef_std_generate_builds_and_runs_compiled_program`.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

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
    // y = forward(x, w, b) = matmul(x, w) + expand(b, 0, 1)
    //                     = [[3, 5]] + [[10, 100]] = [[13, 105]].
    // Parse the rendered data field rather than substring-matching "13" or
    // "105", which would also match "130", "1305", "-13", etc.
    assert!(
        stdout.contains("y = tensor(shape=[1, 2],"),
        "expected y shape header `[1, 2]`; got:\n{stdout}"
    );
    let elements = parse_tensor_data_for(&stdout, "y")
        .unwrap_or_else(|| panic!("failed to parse y tensor data from:\n{stdout}"));
    assert_eq!(
        elements.len(),
        2,
        "expected 2 elements; got {} ({elements:?})\n{stdout}",
        elements.len()
    );
    assert!(
        (elements[0] - 13.0).abs() < 1e-3,
        "y[0] expected 13.0, got {} (full data: {elements:?})\n{stdout}",
        elements[0]
    );
    assert!(
        (elements[1] - 105.0).abs() < 1e-3,
        "y[1] expected 105.0, got {} (full data: {elements:?})\n{stdout}",
        elements[1]
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
    // Every output element ≈ 2.5. The eval-side cross-check at
    // `packages/chelis-std/tests/runtime/attention_eval_cross.ch` does an
    // exact 16-element assert_close_tensor; the build side now matches
    // that rigor (red-team v0.2.6 MEDIUM raised the renderer cap from 10
    // to 32 elements, so all 16 are now visible).
    assert!(
        stdout.contains("attn = tensor(shape=[4, 4],"),
        "expected attn shape header `[4, 4]`; got:\n{stdout}"
    );
    let elements = parse_tensor_data_for(&stdout, "attn")
        .unwrap_or_else(|| panic!("failed to parse attn tensor data from:\n{stdout}"));
    assert_eq!(
        elements.len(),
        16,
        "expected all 16 elements (4x4) to be printed under the new 32-cap; \
         got {} ({elements:?})\n{stdout}",
        elements.len()
    );
    for (i, &v) in elements.iter().enumerate() {
        assert!(
            (v - 2.5).abs() < 1e-3,
            "element {i} expected 2.5, got {v} (full data: {elements:?})\nstdout=\n{stdout}"
        );
    }
}

/// Parse the `data=[...]` slice of a top-level `<name> = tensor(...)`
/// rendering produced by the compiled binary's stdout. Returns the list of
/// f32 elements, or `None` if the binding line or data field can't be
/// located. The format the runtime emits is, e.g.,
/// `attn = tensor(shape=[4, 4], data=[2.5, 2.5, ..., 2.5])`.
fn parse_tensor_data_for(stdout: &str, binding: &str) -> Option<Vec<f32>> {
    let prefix = format!("{binding} = tensor(");
    let line = stdout.lines().find(|l| l.contains(&prefix))?;
    let after_data = line.split("data=[").nth(1)?;
    let inside = after_data.split(']').next()?;
    inside
        .split(',')
        .map(str::trim)
        // The renderer emits a trailing "..." token when the tensor has more
        // elements than the print cap (currently 32). Skip it so the parser
        // returns just the visible numeric prefix.
        .filter(|s| !s.is_empty() && *s != "...")
        .map(|s| s.parse::<f32>().ok())
        .collect()
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
    // Expected 0.3132 (single-element tensor[1, f32]). Parse the data field
    // and compare to the closed-form value within tolerance, instead of
    // substring-matching "0.313" which would also match "1.3130", "0.3135"
    // and similar near-but-wrong outputs.
    let elements = parse_tensor_data_for(&stdout, "nll")
        .unwrap_or_else(|| panic!("failed to parse nll tensor data from:\n{stdout}"));
    assert_eq!(
        elements.len(),
        1,
        "expected 1-element loss tensor; got {} ({elements:?})\n{stdout}",
        elements.len()
    );
    assert!(
        (elements[0] - 0.3132617).abs() < 1e-3,
        "nll expected ≈ 0.3132617, got {} (full data: {elements:?})\n{stdout}",
        elements[0]
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
    // Expected y ≈ [0.8485, 1.1314]. Parse and compare per-element with
    // tolerance instead of substring-matching, so we don't accept e.g.
    // 0.84850001 vs 0.84890000 indistinguishably.
    let elements = parse_tensor_data_for(&stdout, "y")
        .unwrap_or_else(|| panic!("failed to parse y tensor data from:\n{stdout}"));
    assert_eq!(
        elements.len(),
        2,
        "expected 2 elements; got {} ({elements:?})\n{stdout}",
        elements.len()
    );
    assert!(
        (elements[0] - 0.84852).abs() < 1e-3,
        "y[0] expected ≈ 0.84852, got {} (full data: {elements:?})\n{stdout}",
        elements[0]
    );
    assert!(
        (elements[1] - 1.13137).abs() < 1e-3,
        "y[1] expected ≈ 1.13137, got {} (full data: {elements:?})\n{stdout}",
        elements[1]
    );
}
