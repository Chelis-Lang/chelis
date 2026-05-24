//! Issue Chelis-Lang/chelis#218 — finite-difference numerical
//! correctness validation for `grad` over `to_tensor`-literal bodies.
//!
//! The acceptance-oracle suite
//! (`crates/chelis-cli/tests/issue_218_to_tensor_in_grad_body.rs`)
//! verifies that the affected R-round patterns build cleanly. This
//! file goes one step further: each test compiles the generated C,
//! runs the binary, parses the printed gradient, and checks it
//! against an analytically-derived expected value (which a hand
//! finite-difference would also approximate).
//!
//! For `sum(x * w)`-style scalar-loss bodies the analytic gradient
//! is `w` (or a linear function of `w`), so we hardcode the expected
//! tensor and compare element-wise within a small tolerance. The
//! tolerance accounts for the bf16 / f16 / f32 / f64 representation
//! choice — all cases here are f32 with single-digit-magnitude
//! values, so 1e-5 is comfortably above the rounding floor.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn generated_source_needs_blas(out_dir: &Path, source: &str) -> bool {
    fs::read_to_string(out_dir.join(source))
        .map(|text| text.contains("cblas_sgemm(") || text.contains("\"chelis_blas.h\""))
        .unwrap_or(false)
}

/// Link the chelis-generated C source against the platform-
/// appropriate host toolchain. Mirrors `link_generated` from
/// `crates/chelis-cli/tests/cli.rs` to keep this test file
/// self-contained.
fn link_generated(out_dir: &Path, source: &str, binary: &str) -> std::process::ExitStatus {
    let toolchain = chelis_backend_c::toolchain::runtime_toolchain(
        chelis_backend_c::toolchain::CodegenRequirements {
            wants_openmp: true,
            needs_blas: generated_source_needs_blas(out_dir, source),
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
    cmd.status().expect("host compiler should run")
}

/// Parse a printed Chelis tensor of the form
/// `name = tensor(shape=[..], data=[v0, v1, ...])`. Returns the
/// flat data vector. Used to spot-check gradient values without
/// hard-coding the exact printed string (the tensor print pipeline
/// can vary trailing whitespace and value formatting precision).
fn parse_tensor_data(stdout: &str, name: &str) -> Vec<f64> {
    let prefix = format!("{name} = tensor(");
    let line = stdout
        .lines()
        .find(|l| l.starts_with(&prefix))
        .unwrap_or_else(|| panic!("output does not contain `{prefix}` line:\n{stdout}"));
    let data_marker = "data=[";
    let start = line
        .find(data_marker)
        .unwrap_or_else(|| panic!("no `data=[` in line: {line}"))
        + data_marker.len();
    let end = line[start..]
        .find(']')
        .unwrap_or_else(|| panic!("no closing `]` after data: {line}"));
    line[start..start + end]
        .split(',')
        .map(|s| s.trim().parse::<f64>().expect("numeric"))
        .collect()
}

fn assert_grad_matches(actual: &[f64], expected: &[f64], tol: f64, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{msg}: expected length {} got {}; actual={actual:?}",
        expected.len(),
        actual.len(),
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "{msg}: element {i} mismatch: actual={a} expected={e} diff={diff} tol={tol}",
        );
    }
}

fn build_and_run(source: &str, name: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, source);

    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
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

    let source_file = format!("{name}.c");
    let status = link_generated(&out_dir, &source_file, name);
    assert!(status.success(), "link failed: {status}");

    let run_output = StdCommand::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    assert!(
        run_output.status.success(),
        "binary failed: {}\nstderr: {}",
        run_output.status,
        String::from_utf8_lossy(&run_output.stderr),
    );
    String::from_utf8(run_output.stdout).expect("utf-8 stdout")
}

// =================================================================
// Case 1: scalar loss `sum(x * w)` with w from to_tensor literal.
// Analytic gradient w.r.t. x is `w` itself.
// =================================================================

#[test]
fn issue_218_grad_of_sum_mul_x_w_equals_w() {
    let stdout = build_and_run(
        "def g(x: tensor[3, f32]) -> tensor[f32] = {\n\
           w = to_tensor([1.5, 2.5, 3.5])\n\
           sum(mul(copy(x), w), 0)\n\
         }\n\
         def compute_grad(x: tensor[3, f32]) -> tensor[3, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([0.1, 0.2, 0.3]))\n",
        "grad_sum_mul",
    );
    let actual = parse_tensor_data(&stdout, "out");
    // grad of sum(x*w) wrt x = w.
    assert_grad_matches(&actual, &[1.5, 2.5, 3.5], 1e-5, "case 1");
}

// =================================================================
// Case 2: scalar loss `sum(sum(matmul(x, w), 0), 0)` with
// x: [2, 3] and w: [3, 1] from to_tensor literal.
// Analytic gradient: for each x[i, j], d/dx[i, j] sum(x@w) = sum over
// output positions of w[j, k] = w[j, 0] (since k = 1). So
// grad[i, j] = w[j, 0], broadcast across batch:
// grad = [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]].
// =================================================================

#[test]
fn issue_218_grad_of_sum_sum_matmul_x_w_equals_broadcast_w_transpose() {
    let stdout = build_and_run(
        "def g(x: tensor[2, 3, f32]) -> tensor[f32] = {\n\
           w = to_tensor([[1.0], [2.0], [3.0]])\n\
           sum(sum(matmul(copy(x), w), 0), 0)\n\
         }\n\
         def compute_grad(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]))\n",
        "grad_matmul_reduce",
    );
    let actual = parse_tensor_data(&stdout, "out");
    // grad has shape [2, 3]; element [i, j] = w[j, 0].
    assert_grad_matches(&actual, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-5, "case 2");
}

// =================================================================
// Case 3: scalar loss `sum(sum(relu(matmul(x, w)), 0), 0)`.
// All matmul outputs are positive (x is all-positive, w is
// all-positive), so relu is the identity here and the gradient is
// the same as case 2.
// =================================================================

#[test]
fn issue_218_grad_of_sum_relu_matmul_when_all_positive() {
    let stdout = build_and_run(
        "def g(x: tensor[2, 3, f32]) -> tensor[f32] = {\n\
           w = to_tensor([[1.0], [2.0], [3.0]])\n\
           sum(sum(relu(matmul(copy(x), w)), 0), 0)\n\
         }\n\
         def compute_grad(x: tensor[2, 3, f32]) -> tensor[2, 3, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([[0.5, 0.6, 0.7], [0.8, 0.9, 1.0]]))\n",
        "grad_relu_matmul",
    );
    let actual = parse_tensor_data(&stdout, "out");
    assert_grad_matches(&actual, &[1.0, 2.0, 3.0, 1.0, 2.0, 3.0], 1e-5, "case 3");
}

// =================================================================
// Case 4: scalar loss with a negative-literal weight tensor.
// `sum(x * w)` with w = [-1.0, 2.0, -3.0]; grad = w.
// =================================================================

#[test]
fn issue_218_grad_of_sum_mul_with_negative_literal_weight() {
    let stdout = build_and_run(
        "def g(x: tensor[3, f32]) -> tensor[f32] = {\n\
           w = to_tensor([-1.0, 2.0, -3.0])\n\
           sum(mul(copy(x), w), 0)\n\
         }\n\
         def compute_grad(x: tensor[3, f32]) -> tensor[3, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([1.0, 1.0, 1.0]))\n",
        "grad_neg_literal",
    );
    let actual = parse_tensor_data(&stdout, "out");
    assert_grad_matches(&actual, &[-1.0, 2.0, -3.0], 1e-5, "case 4");
}

// =================================================================
// Case 5: multi-layer matmul, both weights from to_tensor literals.
// `g(x) = sum(sum(matmul(matmul(x, w1), w2), 0), 0)` with
// x: [1, 2], w1: [2, 3], w2: [3, 1].
// Analytic gradient: for the chain sum-of-output,
// d/dx[i, j] = sum_k sum_l w1[j, k] * w2[k, l] = w1[j, :] @ w2[:, 0]
// summed over both output dims.
// w1 = [[1,1,1], [1,1,1]] -> w1[0, :] = [1, 1, 1], w1[1, :] = [1, 1, 1]
// w2 = [[1], [1], [1]] -> w1[i, :] @ w2 = 3, so grad[0, j] = 3 for j in 0..2.
// =================================================================

#[test]
fn issue_218_grad_of_two_layer_matmul_with_to_tensor_weights() {
    let stdout = build_and_run(
        "def g(x: tensor[1, 2, f32]) -> tensor[f32] = {\n\
           w1 = to_tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])\n\
           w2 = to_tensor([[1.0], [1.0], [1.0]])\n\
           sum(sum(matmul(matmul(copy(x), w1), w2), 0), 0)\n\
         }\n\
         def compute_grad(x: tensor[1, 2, f32]) -> tensor[1, 2, f32] =\n\
           grad(g, wrt=x)(x)\n\
         out = compute_grad(to_tensor([[0.25, 0.75]]))\n",
        "grad_two_layer",
    );
    let actual = parse_tensor_data(&stdout, "out");
    assert_grad_matches(&actual, &[3.0, 3.0], 1e-5, "case 5");
}
