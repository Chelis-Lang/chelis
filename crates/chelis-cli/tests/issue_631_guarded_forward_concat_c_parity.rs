//! chelis#631: guarded (`if`/`fail`) FORWARD builds through the
//! host-program lane — eval-vs-C parity for the avgpool oracle's forward
//! program.
//!
//! Before chelis#631, the checker typed a list `concat` as its element
//! type (an axis-0 concat of two `[1, m]` rows stayed `[1, m]`), the host
//! lane baked that wrong static extent into its tensor-helper signatures,
//! and the compiled binary aborted at the chelis#616 runtime-dim guard
//! (`runtime dim `m` mismatch`) while `chelis eval` computed the correct
//! values. With concat result typing (spec/04-type-system.md §4.5.4) the
//! static type is honest (`[2, m]`) and the guarded forward program has
//! full build-and-run parity. The gradient twin of this program is the
//! chelis#616 completion oracle in `issue_368_grad_concat_windows.rs`.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

/// The chelis#616 avgpool oracle's forward program, verbatim from the
/// chelis#631 report: window 2, stride 2, runtime window count `m`,
/// behind an `if`/`fail` kernel-length guard.
const AVGPOOL: &str = "sig avgpool1d: tensor[n, f32] -> tensor[m, f32]\n\
def avgpool1d(x) = {\n\
  n = cast(shape(x, cast(0, int32)), int64)\n\
  if gt(cast(2, int64), n) then fail(\"kernel exceeds input length\") else {\n\
    m = add(floor_div(sub(n, cast(2, int64)), cast(2, int64)), cast(1, int64))\n\
    rows = [window_row(&x, m, cast(0, int64)), window_row(&x, m, cast(1, int64))]\n\
    mean(concat(rows, cast(0, int32)), cast(0, int32))\n\
  }\n\
}\n\
def window_row[n](x: &tensor[n, f32], m: int64, k: int64) -> tensor[u, m, f32] = {\n\
  start = cast(k, int64)\n\
  extent = cast(add(add(k, mul(sub(m, cast(1, int64)), cast(2, int64))), cast(1, int64)), int64)\n\
  reshape(stride(shrink(x, [[start, extent]]), cast(2, int64)), [cast(1, int64), m])\n\
}";

fn avgpool_source(values: &[f64]) -> String {
    format!(
        "module Repro.AvgPoolFwd\n{AVGPOOL}\nout = avgpool1d(to_tensor([{}]))\n",
        f32_literal(values)
    )
}

fn f32_literal(values: &[f64]) -> String {
    values
        .iter()
        .map(|v| format!("cast({v:?}, f32)"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// avgpool1d([1, 2, ..., n], window 2, stride 2): m = (n-2)/2 + 1 window
/// means; window i averages elements [2i+1, 2i+2] -> 2i + 1.5.
fn expected_avgpool(n: usize) -> Vec<f64> {
    let m = (n - 2) / 2 + 1;
    (0..m).map(|i| (2 * i) as f64 + 1.5).collect()
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

/// THE chelis#631 acceptance oracle: the guarded forward avgpool program
/// builds with `--target c`, compiles, RUNS TO SUCCESS, and prints the
/// same pooled means as `chelis eval`. Before the fix the binary aborted
/// at the runtime-dim guard (`runtime dim `m` mismatch`) because the
/// concat's static type carried the element extent 1 where the runtime
/// value has 2 rows.
#[test]
fn issue_631_guarded_forward_avgpool_eval_matches_c() {
    let input: Vec<f64> = (1..=4).map(|v| v as f64).collect();
    let source = avgpool_source(&input);

    let eval_out = run_eval(&source, "avgpoolfwd");
    assert!(
        eval_out.status.success(),
        "eval leg failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close("eval avgpool n=4", &eval_values, &expected_avgpool(4));

    let (_dir, build_dir) = build_c(&source, "avgpoolfwd");
    let bin = gcc(&build_dir, "avgpoolfwd", "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "C binary must run the guarded forward program to success; stderr={}",
        String::from_utf8_lossy(&run.stderr)
    );
    let c_values = parse_tensor_data(&String::from_utf8_lossy(&run.stdout));
    assert_close("C avgpool n=4", &c_values, &eval_values);
}

/// Length generality: the same program shape at n = 6 (m = 3 windows)
/// keeps eval-vs-C parity — the runtime window count is genuinely
/// resolved at run time, not baked for one length.
#[test]
fn issue_631_guarded_forward_avgpool_matches_c_at_n6() {
    let input: Vec<f64> = (1..=6).map(|v| v as f64).collect();
    let source = avgpool_source(&input);

    let eval_out = run_eval(&source, "avgpoolfwd6");
    assert!(
        eval_out.status.success(),
        "eval leg failed: {}",
        String::from_utf8_lossy(&eval_out.stderr)
    );
    let eval_values = parse_tensor_data(&String::from_utf8_lossy(&eval_out.stdout));
    assert_close("eval avgpool n=6", &eval_values, &expected_avgpool(6));

    let (_dir, build_dir) = build_c(&source, "avgpoolfwd6");
    let bin = gcc(&build_dir, "avgpoolfwd6", "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "C binary must run the n=6 guarded forward program; stderr={}",
        String::from_utf8_lossy(&run.stderr)
    );
    let c_values = parse_tensor_data(&String::from_utf8_lossy(&run.stdout));
    assert_close("C avgpool n=6", &c_values, &eval_values);
}

/// Soundness pin (found while fixing chelis#631): a guarded program
/// whose `fail` branch fires while the OTHER branch computes cleanly
/// must abort in the C lane too. The DAG lane lowers `fail` to a
/// mask-selected zero placeholder (grad-lane semantics); before the
/// `expr_reaches_forward_fail` host-lane gate, this program's compiled binary
/// exited 0 printing ZEROS where eval aborts with the user's message —
/// silent-wrong, not just a message mismatch. Fail-reaching bodies must
/// stay in the host lane, whose `if`/`fail` are real control flow.
#[test]
fn issue_631_data_dependent_fail_branch_aborts_in_c() {
    let source = "module Repro.FailClean\n\
sig f: tensor[n, f32] -> tensor[n, f32]\n\
def f(x) = {\n\
  total = tensor_to_scalar(sum(x, cast(0, int32)))\n\
  if gt(total, cast(0.0, f32)) then fail(\"positive sum\") else neg(x)\n\
}\n\
out = f(to_tensor([cast(1.0, f32), cast(2.0, f32)]))\n";

    let eval_out = run_eval(source, "failclean");
    assert!(
        !eval_out.status.success(),
        "eval must take the fail branch; stdout={}",
        String::from_utf8_lossy(&eval_out.stdout)
    );
    let eval_err = String::from_utf8_lossy(&eval_out.stderr).into_owned();
    assert!(
        eval_err.contains("positive sum"),
        "eval must surface the fail message; stderr={eval_err}"
    );

    let (_dir, build_dir) = build_c(source, "failclean");
    let bin = gcc(&build_dir, "failclean", "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        !run.status.success(),
        "C binary must abort on the taken fail branch, never return zeros; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("positive sum"),
        "C abort must carry the fail message; stderr={stderr}"
    );
}

/// Negative parity: an input shorter than the kernel takes the `fail`
/// branch in BOTH lanes — the guard survives the typing fix (loud, with
/// the user's message; never a mis-sized result).
#[test]
fn issue_631_guard_fail_branch_errs_in_both_lanes() {
    let source = avgpool_source(&[1.0]);

    let eval_out = run_eval(&source, "avgpoolguard");
    assert!(
        !eval_out.status.success(),
        "eval must take the fail branch for n=1; stdout={}",
        String::from_utf8_lossy(&eval_out.stdout)
    );
    let eval_err = String::from_utf8_lossy(&eval_out.stderr).into_owned();
    assert!(
        eval_err.contains("kernel exceeds input length"),
        "eval must surface the fail message; stderr={eval_err}"
    );

    let (_dir, build_dir) = build_c(&source, "avgpoolguard");
    let bin = gcc(&build_dir, "avgpoolguard", "self_bin");
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        !run.status.success(),
        "C binary must take the fail branch for n=1; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("kernel exceeds input length"),
        "C abort must carry the fail message; stderr={stderr}"
    );
}
