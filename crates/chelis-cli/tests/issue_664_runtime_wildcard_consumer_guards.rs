//! chelis#664: C-lane runtime guards for CONSUMERS of runtime-wildcard
//! extents — elementwise operand-shape agreement and reshape numel over
//! non-static shapes.
//!
//! A movement op with runtime bounds produces a wildcard extent the
//! checker accepts permissively (spec §4.5); before chelis#664 the C
//! backend emitted NO operand-agreement guard on elementwise ops and its
//! reshape numel guard fired only for Node-valued targets, so a consumer
//! mixing that wildcard with a differently-sized sibling read out of
//! bounds and the binary exited 0 with wrong values while `chelis eval`
//! rejected the same program — the silent-divergence class. These pins
//! assert loud error parity in both lanes (and no false aborts on the
//! matching-shape twins). Pre-existing #616-era gap, surfaced by the
//! PR #663 red team; the shrink controls prove the class predates the
//! chelis#632 checker change.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

fn f32_literal(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("cast({v:?}, f32)"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn run_eval(source: &str, stem: &str) -> std::process::Output {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write");
    Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run chelis eval")
}

fn parse_tensor_data(stdout: &str) -> Vec<f64> {
    let line = stdout
        .lines()
        .find(|l| l.contains("data=["))
        .unwrap_or_else(|| panic!("no tensor in output: {stdout}"));
    line.split_once("data=[")
        .and_then(|(_, r)| r.split_once(']'))
        .map(|(s, _)| {
            s.split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| t.trim().parse::<f64>().expect("tensor element"))
                .collect()
        })
        .unwrap()
}

fn assert_close(label: &str, got: &[f64], want: &[f64]) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch got {got:?} want {want:?}"
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < 1e-3,
            "{label}: elem {i}: got {g}, want {w} (full got={got:?} want={want:?})"
        );
    }
}

fn build_c(source: &str, stem: &str) -> (TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
    fs::write(&src_path, source).expect("write .ch source");
    let build_dir = dir.path().join("build");
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            build_dir.to_str().unwrap(),
        ])
        .assert()
        .success();
    (dir, build_dir)
}

fn gcc(build_dir: &Path, stem: &str, bin_name: &str) -> std::path::PathBuf {
    let kernel = build_dir.join(format!("{stem}.c"));
    let runtime = build_dir.join("libchelis_runtime.a");
    let bin = build_dir.join(bin_name);
    let compile = StdCommand::new("gcc")
        .args([
            "-O0",
            "-std=c11",
            "-I",
            build_dir.to_str().unwrap(),
            kernel.to_str().unwrap(),
            "-o",
            bin.to_str().unwrap(),
            runtime.to_str().unwrap(),
            "-lm",
            "-lpthread",
            "-ldl",
        ])
        .output()
        .expect("invoke gcc");
    assert!(
        compile.status.success(),
        "gcc compile failed: stderr={}",
        String::from_utf8_lossy(&compile.stderr)
    );
    bin
}

fn six() -> String {
    f32_literal(&[1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
}

/// Both lanes must reject `source` loudly: eval with `eval_needle` on
/// stderr, the compiled C binary with a non-zero exit and `c_needle` on
/// stderr — never exit 0 with values.
fn assert_error_parity(source: &str, stem: &str, eval_needle: &str, c_needle: &str) {
    let eval_out = run_eval(source, stem);
    assert!(
        !eval_out.status.success(),
        "{stem}: eval must reject; stdout={}",
        String::from_utf8_lossy(&eval_out.stdout)
    );
    let eval_err = String::from_utf8_lossy(&eval_out.stderr).into_owned();
    assert!(
        eval_err.contains(eval_needle),
        "{stem}: eval must name the mismatch ({eval_needle}); stderr={eval_err}"
    );

    let (_dir, build_dir) = build_c(source, stem);
    let bin = gcc(&build_dir, stem, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        !run.status.success(),
        "{stem}: C binary must abort, never compute over mismatched shapes; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains(c_needle),
        "{stem}: C abort must name the guard ({c_needle}); stderr={stderr}"
    );
}

/// Both lanes must run `source` to success and print the same values —
/// the guards must not false-abort matching shapes.
fn assert_value_parity(source: &str, stem: &str, want: &[f64]) {
    let eval_out = run_eval(source, stem);
    assert!(
        eval_out.status.success(),
        "{stem}: eval leg failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close(&format!("{stem} eval"), &eval_values, want);

    let (_dir, build_dir) = build_c(source, stem);
    let bin = gcc(&build_dir, stem, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "{stem}: C binary must run to success (no false abort); stderr={}",
        String::from_utf8_lossy(&run.stderr)
    );
    let c_values = parse_tensor_data(&String::from_utf8_lossy(&run.stdout));
    assert_close(&format!("{stem} C"), &c_values, &eval_values);
}

/// The chelis#664 headline: a runtime-strided operand beside the full
/// input. eval rejects `[3]` vs `[6]`; pre-fix C exited 0 printing
/// `[2, 5, 8]` (the strided view added to x's first three elements).
#[test]
fn issue_664_elementwise_stride_operand_mismatch_errs_in_both_lanes() {
    let source = format!(
        "module Repro.ElemStride\nsig f: tensor[n, f32] -> tensor[n, f32]\ndef f(x) = {{\n  s = stride(x, cast(2, int64))\n  add(s, x)\n}}\nout = f(to_tensor([{}]))\n",
        six()
    );
    assert_error_parity(
        &source,
        "elemstride",
        "tensor shapes must match for elementwise op",
        "elementwise operand shape mismatch",
    );
}

/// Pad variant: `[4]` vs `[3]` — pre-fix the C binary's last element was
/// an out-of-bounds read of `x`.
#[test]
fn issue_664_elementwise_pad_operand_mismatch_errs_in_both_lanes() {
    let source = format!(
        "module Repro.ElemPad\nsig f: tensor[n, f32] -> tensor[n, f32]\ndef f(x) = {{\n  p = pad(x, [[cast(1, int64), cast(0, int64)]], cast(0.0, f32))\n  add(p, x)\n}}\nout = f(to_tensor([{}]))\n",
        f32_literal(&[1.0, 2.0, 3.0])
    );
    assert_error_parity(
        &source,
        "elempad",
        "tensor shapes must match for elementwise op",
        "elementwise operand shape mismatch",
    );
}

/// Shrink control: the identical divergence through a RUNTIME-bounded
/// `shrink` (untouched by chelis#632) — proves the guard covers the
/// pre-existing #616-era path, not just the chelis#632-widened one. (A
/// literal bound would pin the sig's `n` at check time; the runtime
/// bound `n - 3` keeps the wildcard route: `[3]` vs `[6]` at run time.)
#[test]
fn issue_664_elementwise_shrink_control_errs_in_both_lanes() {
    let source = format!(
        "module Repro.ElemShrink\nsig f: tensor[n, f32] -> tensor[n, f32]\ndef f(x) = {{\n  k = cast(sub(cast(shape(x, cast(0, int32)), int64), cast(3, int64)), int64)\n  s = shrink(x, [[cast(0, int64), k]])\n  add(s, x)\n}}\nout = f(to_tensor([{}]))\n",
        six()
    );
    assert_error_parity(
        &source,
        "elemshrink",
        "tensor shapes must match for elementwise op",
        "elementwise operand shape mismatch",
    );
}

/// Reshape whose target folds to a SYM (the shape read of the ORIGINAL
/// input, 6) over the strided view (3 elements). The pre-fix numel guard
/// fired only for Node-valued targets, so the binary exited 0 printing
/// `[1, 3, 5, 0, 0, 0]`.
#[test]
fn issue_664_reshape_sym_target_numel_mismatch_errs_in_both_lanes() {
    let source = format!(
        "module Repro.ReshapeSym\nsig f: tensor[n, f32] -> tensor[u, f32]\ndef f(x) = {{\n  s = stride(x, cast(2, int64))\n  reshape(s, [cast(shape(x, cast(0, int32)), int64)])\n}}\nout = f(to_tensor([{}]))\n",
        six()
    );
    assert_error_parity(
        &source,
        "reshapesym",
        "reshape expects 6 elements but tensor has 3",
        "reshape numel mismatch",
    );
}

/// Reshape with a fully STATIC target over a runtime-sized input: the
/// target claims 6 elements, the strided view has 3. No runtime target,
/// so pre-fix NO guard was emitted at all.
#[test]
fn issue_664_reshape_static_target_over_runtime_input_errs_in_both_lanes() {
    let source = format!(
        "module Repro.ReshapeStatic\nsig f: tensor[n, f32] -> tensor[u, f32]\ndef f(x) = {{\n  s = stride(x, cast(2, int64))\n  reshape(s, [cast(6, int64)])\n}}\nout = f(to_tensor([{}]))\n",
        six()
    );
    assert_error_parity(
        &source,
        "reshapestatic",
        "reshape expects 6 elements but tensor has 3",
        "reshape numel mismatch",
    );
}

/// No false abort: two runtime-strided operands with MATCHING shapes
/// must still run in both lanes (`[2, 6, 10]`).
#[test]
fn issue_664_matching_runtime_operands_still_run() {
    let source = format!(
        "module Repro.ElemMatch\nsig f: tensor[n, f32] -> tensor[u, f32]\ndef f(x) = {{\n  a = stride(x, cast(2, int64))\n  b = stride(x, cast(2, int64))\n  add(a, b)\n}}\nout = f(to_tensor([{}]))\n",
        six()
    );
    assert_value_parity(&source, "elemmatch", &[2.0, 6.0, 10.0]);
}

/// No false abort: an identity reshape whose Sym-resolved target equals
/// the input numel must still run in both lanes. (The target resolves to
/// the Load symbol `n`, so the honest return claim is `tensor[n, f32]` —
/// a distinct `u` would trip the §4.4.1 rigidity guard.)
#[test]
fn issue_664_identity_reshape_sym_target_still_runs() {
    let source = format!(
        "module Repro.ReshapeIdent\nsig f: tensor[n, f32] -> tensor[n, f32]\ndef f(x) = reshape(x, [cast(shape(x, cast(0, int32)), int64)])\nout = f(to_tensor([{}]))\n",
        six()
    );
    assert_value_parity(&source, "reshapeident", &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}
