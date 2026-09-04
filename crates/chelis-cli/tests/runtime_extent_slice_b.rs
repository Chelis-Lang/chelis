//! Public CLI acceptance rows for runtime-extent Slice B (chelis#1277).

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::TempDir;

#[path = "common/mod.rs"]
mod common;

/// Chelis#1482's remaining reproducer: a runtime-bound `shrink` under a
/// symbolic signature, consumed by a composite elementwise lowering that
/// still synthesizes anonymous-extent constants.
const RUNTIME_BOUND_SHRINK_SIGMOID: &str = "module Repro.M1\n\
sig f: tensor[n, f32] -> tensor[u, f32]\n\
def f(x) = {\n  \
k = shape(x, cast(0, int32))\n  \
z = cast((k - k), int64)\n  \
s = shrink(x, [[z, k]])\n  \
sigmoid(s)\n\
}\n\
out = f(to_tensor([cast(1.0, f32), cast(-2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";

/// The original #1482 spelling. Chelis#1313's structural ReLU identity no
/// longer synthesizes a tensor zero, so this exact program now executes.
const RUNTIME_BOUND_SHRINK_RELU: &str = "module Repro.M1\n\
sig f: tensor[n, f32] -> tensor[u, f32]\n\
def f(x) = {\n  \
k = shape(x, cast(0, int32))\n  \
z = cast((k - k), int64)\n  \
s = shrink(x, [[z, k]])\n  \
relu(s)\n\
}\n\
out = f(to_tensor([cast(1.0, f32), cast(-2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";

/// The same elementwise shape over a plain symbolic tensor. This builds on
/// `main` and must keep building: its zero fill carries the declared
/// dimension `n`, not an anonymous extent.
const SYMBOLIC_RELU: &str = "module Probe.SymRelu\n\
sig f: tensor[n, f32] -> tensor[n, f32]\n\
def f(x) = relu(x)\n\
out = f(to_tensor([cast(1.0, f32), cast(-2.0, f32)]))\n";

fn fixture(dir: &TempDir, name: &str, source: &str) -> std::path::PathBuf {
    let path = dir.path().join(name);
    fs::write(&path, source).expect("fixture");
    path
}

fn build_c(path: &Path, out_dir: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("build")
}

fn eval(path: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "eval",
            "--allow-style-violations",
            "--file",
            path.to_str().unwrap(),
        ])
        .output()
        .expect("eval")
}

/// Chelis#1482 / `runtime_extents.md` C4.3: sigmoid's synthesized constants
/// still have a missing extent source and therefore produce the registered
/// typed receipt, not the occurrence pass's internal compiler error or an
/// input extent silently substituted for the real one.
///
/// Eval and the plain-symbolic control remain negative parity: the gap is the
/// source-less synthesized constant under a runtime-bound result, not sigmoid
/// or symbolic elementwise execution generally.
#[test]
fn runtime_bound_shrink_consumed_elementwise_reports_a_typed_receipt() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "repro.ch", RUNTIME_BOUND_SHRINK_SIGMOID);

    let build = build_c(&path, &dir.path().join("repro-out"));
    let stderr = String::from_utf8_lossy(&build.stderr).to_string();
    assert!(
        !build.status.success(),
        "the build must refuse rather than emit undeclared C: {stderr}"
    );
    assert!(
        !stderr.contains("internal compiler error"),
        "the ICE must be replaced by the receipt, not accompanied by it: {stderr}"
    );
    assert_eq!(
        build.status.code(),
        Some(1),
        "a typed refusal exits 1, where the panic exited 101: {stderr}"
    );
    assert!(
        stderr.contains("unsupported: "),
        "the receipt carries the section C2 brand: {stderr}"
    );
    assert!(
        stderr.contains("unimplemented chelis#1482"),
        "the receipt cites the issue that owns the gap: {stderr}"
    );
    assert!(
        stderr.contains("extent source(s) for"),
        "the receipt names the axis that has no source: {stderr}"
    );
    assert!(
        stderr.contains("(codegen:c)"),
        "the receipt names the lane that refused: {stderr}"
    );

    // The eval lane accepts this program on `main` and still does: it
    // computes shapes from values and never sees the anonymous extent.
    let evaluated = eval(&path);
    let stdout = String::from_utf8_lossy(&evaluated.stdout).to_string();
    assert!(
        evaluated.status.success(),
        "eval acceptance is unchanged: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let values = common::parse_tensor_data(&stdout, "out");
    let expected = [
        0.731_058_6_f64,
        0.119_202_92_f64,
        0.952_574_13_f64,
        0.982_013_76_f64,
    ];
    assert_eq!(values.len(), expected.len(), "{stdout}");
    for (actual, expected) in values.iter().zip(expected) {
        assert!((actual - expected).abs() < 1e-6, "{actual} vs {expected}");
    }

    // The control that decides how narrow the rule had to be: an
    // elementwise fill under a DECLARED dimension is not a sourceless axis.
    let symbolic = fixture(&dir, "sym_relu.ch", SYMBOLIC_RELU);
    let build = build_c(&symbolic, &dir.path().join("sym-out"));
    assert!(
        build.status.success(),
        "a declared dimension on an elementwise fill still builds: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let evaluated = eval(&symbolic);
    assert!(
        String::from_utf8_lossy(&evaluated.stdout).contains("[1.0, 0.0]"),
        "{}",
        String::from_utf8_lossy(&evaluated.stdout)
    );
}

/// Part of chelis#1482 via chelis#1313: the original ReLU spelling is now a
/// positive build-link-run/eval regression. The dedicated identity removes
/// exactly the anonymous tensor-zero operand that triggered the source walk;
/// this does not claim the broader synthesized-constant class is closed.
#[test]
fn runtime_bound_shrink_relu_builds_and_matches_eval_exactly() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = fixture(&dir, "relu_repro.ch", RUNTIME_BOUND_SHRINK_RELU);
    let out_dir = dir.path().join("relu-repro-out");

    let build = build_c(&path, &out_dir);
    assert!(
        build.status.success(),
        "dedicated ReLU must build without a synthesized Const receipt: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let linked = common::link_generated(&out_dir, "relu_repro.c", "relu_repro");
    assert!(linked.success(), "generated ReLU program must link");
    let compiled = std::process::Command::new(out_dir.join("relu_repro"))
        .output()
        .expect("compiled ReLU program");
    assert!(
        compiled.status.success(),
        "compiled ReLU program failed: {}",
        String::from_utf8_lossy(&compiled.stderr)
    );

    let evaluated = eval(&path);
    assert!(
        evaluated.status.success(),
        "eval ReLU program failed: {}",
        String::from_utf8_lossy(&evaluated.stderr)
    );
    let compiled_stdout = String::from_utf8_lossy(&compiled.stdout);
    let evaluated_stdout = String::from_utf8_lossy(&evaluated.stdout);
    assert_eq!(
        compiled_stdout, evaluated_stdout,
        "compiled C and eval must agree byte-for-byte"
    );
    assert!(
        compiled_stdout.contains("shape=[4]")
            && compiled_stdout.contains("data=[1.0, 0.0, 3.0, 4.0]"),
        "ReLU must retain the exact original result: {compiled_stdout}"
    );
}
