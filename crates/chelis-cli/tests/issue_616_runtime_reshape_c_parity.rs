//! chelis#616 step 3b: forward eval-vs-C parity for runtime (node-valued)
//! RESHAPE target extents — the windowed `reshape(stride(shrink(x, ...)),
//! [1i64, m])` form of the avgpool oracle's `window_row`, with the window count
//! `m` computed from `shape()` arithmetic at run time.
//!
//! The reshape target `m` lowers to a rank-0 integer scalar node the op
//! references as `chelis_ir::dag::RtDim::Node` (exactly like a movement bound); the C lane
//! declares `int m = <scalar>` at the reshape behind negativity + numel
//! abort guards, and the eval lane resolves the scalar and enforces the
//! numel invariant with a clean error. One compiled binary handles every
//! input length.

mod common;

use common::authored_c_symbol;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

/// Single-window verb: `m = (n - 2) / 2 + 1` windows of stride 2 starting
/// at 0, reshaped to `[1, m]`. For `x = [1, 2, ..., n]` the window keeps
/// `[1, 3, 5, ...]` (`m` odd values).
const WINDOW_BODY: &str = "\
  m = add(floor_div(sub(cast(shape(x, cast(0, i32)), i64), cast(2, i64)), cast(2, i64)), cast(1, i64))\n\
  extent = cast(add(mul(sub(m, cast(1, i64)), cast(2, i64)), cast(1, i64)), i64)\n\
  reshape(stride(shrink(x, [[cast(0, i64), extent]]), cast(2, i64)), [cast(1, i64), m])";

fn window_source(out_line: &str) -> String {
    format!(
        "module Repro.RtWindow\nsig window[n, m]: tensor[n, f32] -> tensor[1, m, f32]\ndef window(x) = {{\n{WINDOW_BODY}\n}}\n{out_line}\n"
    )
}

fn f32_literal(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("cast({v:?}, f32)"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// The window values for input `[1, 2, ..., n]`: `m = (n-2)/2 + 1` odd
/// values `1, 3, 5, ...`.
fn expected_window(n: usize) -> Vec<f64> {
    let m = (n - 2) / 2 + 1;
    (0..m).map(|i| (1 + 2 * i) as f64).collect()
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

fn gcc(
    build_dir: &Path,
    stem: &str,
    extra_src: Option<&Path>,
    bin_name: &str,
) -> std::path::PathBuf {
    let kernel = build_dir.join(format!("{stem}.c"));
    let runtime = build_dir.join("libchelis_runtime.a");
    let bin = build_dir.join(bin_name);
    let mut args = vec![
        "-O0".to_string(),
        "-std=c11".to_string(),
        "-I".to_string(),
        build_dir.to_str().unwrap().to_string(),
        kernel.to_str().unwrap().to_string(),
    ];
    if let Some(extra) = extra_src {
        args.push(extra.to_str().unwrap().to_string());
    }
    args.extend([
        "-o".to_string(),
        bin.to_str().unwrap().to_string(),
        runtime.to_str().unwrap().to_string(),
        "-lm".to_string(),
        "-lpthread".to_string(),
        "-ldl".to_string(),
    ]);
    let compile = StdCommand::new("gcc")
        .args(&args)
        .output()
        .expect("invoke gcc");
    assert!(
        compile.status.success(),
        "gcc compile failed: stderr={}",
        String::from_utf8_lossy(&compile.stderr)
    );
    bin
}

/// eval-vs-C forward parity at n = 6 (m = 3): both lanes compute the window
/// `[1, 3, 5]` by resolving the shrink extent AND the reshape target `m`
/// from runtime scalars.
#[test]
fn issue_616_runtime_reshape_window_forward_eval_matches_c() {
    let input: Vec<f64> = (1..=6).map(|v| v as f64).collect();
    let source = window_source(&format!(
        "out = window(to_tensor([{}]))",
        f32_literal(&input)
    ));

    let eval_out = run_eval(&source, "rtwindow");
    assert!(
        eval_out.status.success(),
        "forward eval failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close("eval window n=6", &eval_values, &expected_window(6));

    let (_dir, build_dir) = build_c(&source, "rtwindow");
    let bin = gcc(&build_dir, "rtwindow", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "emitted program exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let c_values = parse_tensor_data(&String::from_utf8_lossy(&run.stdout));
    assert_close("eval-vs-C window n=6", &c_values, &eval_values);
}

/// ONE compiled binary (the exported `def out(x)`), fed n = 4, 6, 9, sizes
/// the reshape target `m` from the runtime scalar each call — the window
/// count is never codegen-baked.
#[test]
fn issue_616_runtime_reshape_c_binary_handles_multiple_lengths() {
    let source = format!(
        "module Repro.RtWindowBare\nsig out[n, m]: tensor[n, f32] -> tensor[1, m, f32]\ndef out(x) = {{\n{WINDOW_BODY}\n}}\n"
    );
    let (_dir, build_dir) = build_c(&source, "rtwindowbare");
    let out_symbol = authored_c_symbol("out");

    let lengths = [4usize, 6, 9];
    let runs = lengths
        .iter()
        .map(|n| {
            format!(
                "    {{ int64_t shape[1] = {{{n}}}; chelis_tensor* x = chelis_alloc(1, shape, CHELIS_DTYPE_F32); \
                 chelis_tensor_write* x_guard = chelis_tensor_begin_write(x); chelis_write_view x_view = chelis_tensor_write_view(x_guard); \
                 for (int i = 0; i < {n}; i++) ((float *)x_view.data)[i] = (float)(i + 1); chelis_tensor_end_write(x_guard); \
                 chelis_tensor* w = {out_symbol}(x); chelis_read_view w_view = chelis_tensor_read_view(w); \
                 for (int64_t i = 0; i < w_view.count; i++) printf(\"%.6f\\n\", ((const float *)w_view.data)[i]); \
                 printf(\"---\\n\"); chelis_tensor_release(w); chelis_tensor_release(x); }}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let driver_src = format!(
        r#"
#include <stdio.h>
#include "chelis_runtime.h"
extern chelis_tensor* {out_symbol}(chelis_tensor* arg0);
static chelis_tensor* out(chelis_tensor* arg0) {{ chelis_tensor_retain(arg0); return arg0; }}
int main(void) {{
{runs}
    return 0;
}}
"#
    );
    let driver = build_dir.join("driver.c");
    fs::write(&driver, driver_src).expect("write driver.c");
    let bin = gcc(&build_dir, "rtwindowbare", Some(&driver), "driver_bin");
    let run = StdCommand::new(&bin).output().expect("run driver binary");
    assert!(
        run.status.success(),
        "driver binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let blocks: Vec<Vec<f64>> = String::from_utf8_lossy(&run.stdout)
        .split("---")
        .filter(|blk| !blk.trim().is_empty())
        .map(|blk| {
            blk.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().parse::<f64>().expect("window element"))
                .collect()
        })
        .collect();
    assert_eq!(blocks.len(), lengths.len(), "one window per input length");
    for (n, got) in lengths.iter().zip(blocks.iter()) {
        assert_close(&format!("C window n={n}"), got, &expected_window(*n));
    }
}

/// GRADIENT eval-vs-C parity for the full runtime window chain: the loss
/// `sum(sum(w))` over the windowed `[1, m]` reshape reads elements
/// 0, 2, ... of the input, so the gradient is the upsample mask
/// `[1, 0, 1, 0]` — computed by the runtime movement adjoints (Shape-read
/// trim bounds, runtime merge extents) identically in both lanes.
#[test]
fn issue_616_runtime_window_grad_eval_matches_c() {
    let source = "module Repro.RtWindowGrad\nsig f: tensor[4, f32] -> f32\ndef f(x) = {\n\
  m = add(floor_div(sub(cast(shape(x, cast(0, i32)), i64), cast(2, i64)), cast(2, i64)), cast(1, i64))\n\
  extent = cast(add(mul(sub(m, cast(1, i64)), cast(2, i64)), cast(1, i64)), i64)\n\
  w = reshape(stride(shrink(x, [[cast(0, i64), extent]]), cast(2, i64)), [cast(1, i64), m])\n\
  sum(sum(w, cast(0, i32)), cast(0, i32)) |> tensor_to_scalar\n\
}\nout = grad(f)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";

    let eval_out = run_eval(source, "rtwindowgrad");
    assert!(
        eval_out.status.success(),
        "runtime window grad eval failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_grad = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close("eval window grad", &eval_grad, &[1.0, 0.0, 1.0, 0.0]);

    let (_dir, build_dir) = build_c(source, "rtwindowgrad");
    let bin = gcc(&build_dir, "rtwindowgrad", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "emitted grad program exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    let c_grad = parse_tensor_data(&String::from_utf8_lossy(&run.stdout));
    assert_close("eval-vs-C window grad", &c_grad, &eval_grad);
}

/// Error-path parity (soundness): a runtime reshape target that resolves to
/// a NEGATIVE extent must fail LOUDLY in both lanes — the eval lane rejects
/// the negative scalar, the C lane aborts at the emitted negativity guard —
/// never a silently mis-sized allocation.
#[test]
fn issue_616_runtime_reshape_negative_extent_errs_in_both_lanes() {
    // m = n - 10 < 0 for the 4-element input.
    let body = "\
  m = sub(cast(shape(x, cast(0, i32)), i64), cast(10, i64))\n\
  reshape(x, [cast(1, i64), m])";
    let input = f32_literal(&[1.0, 2.0, 3.0, 4.0]);
    let source = format!(
        "module Repro.RtNegDim\nsig f[n, m]: tensor[n, f32] -> tensor[1, m, f32]\ndef f(x) = {{\n{body}\n}}\nout = f(to_tensor([{input}]))\n"
    );

    let eval_out = run_eval(&source, "rtnegdim");
    assert!(
        !eval_out.status.success(),
        "eval must reject the negative runtime reshape extent; stdout={}",
        String::from_utf8_lossy(&eval_out.stdout)
    );

    let (_dir, build_dir) = build_c(&source, "rtnegdim");
    let bin = gcc(&build_dir, "rtnegdim", None, "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        !run.status.success(),
        "C binary must abort on the negative runtime reshape extent; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("must be non-negative") || stderr.contains("numel mismatch"),
        "C abort must name the reshape runtime guard; stderr={stderr}"
    );
}
