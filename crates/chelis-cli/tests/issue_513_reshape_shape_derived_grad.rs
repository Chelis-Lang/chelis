//! Issue #513 gap 1 (Load-traceable declaration of runtime-derived dims),
//! `reshape_shape_derived_grad` witness — the minimal reproducer of the
//! symbolic-dim-machinery gap called out in the issue thread:
//!
//!   `grad` through a `reshape` whose TARGET dim is `shape()`-derived (a
//!   runtime value) over a symbolic-rank input `tensor[n, f32]`. The
//!   sum/reshape backward `Expand`/`Sum` inherited a `Sym("k")` that no Load
//!   declared, so `symbolic_occurrences` ICE'd ("symbolic dim `k` ... no Load
//!   input declares it"). No `mean` divisor and no windowing — the narrowest
//!   form of the runtime-derived-dim declaration gap.
//!
//! Fix (`extract_reshape_dim_list` in `chelis-ir/src/lower.rs`): a
//! `shape(operand, axis)`-derived reshape target dim (directly or through the
//! `let k = shape(x, 0); reshape(&x, [k, 1])` indirection) is resolved to the
//! operand's declaring source dim and recorded as a `shape_dep` — folding to a
//! concrete `Lit` when the operand axis is static and to a Load-carried
//! `Named` when symbolic. Either way the dim traces to a declaring input, so
//! the reduce/reshape backward `Expand`/`Sum` no longer carries an orphan
//! symbol.
//!
//! This is the FD + eval-vs-C-backend acceptance oracle for that witness.
//! The BROADER runtime-symbolic-window `grad` (avgpool1d with `shape()`-
//! derived `mean` divisor + runtime `shrink`/`stride` bounds + `if`/`fail`
//! `*`-wildcard adjoints) remains an expected-to-fail residual, pinned in
//! `issue_368_grad_concat_windows.rs::issue_368_runtime_symbolic_window_grad_is_tracked_residual`
//! (it needs scalar `shape()` reads + int arithmetic lowered into the DAG,
//! chelis#513 gaps 2/3 — a broader rewrite).

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

/// Run `chelis eval` on a forward program returning a scalar `out`, and parse
/// the printed scalar. Used to build a central-difference finite-difference
/// gradient from the FORWARD loss.
fn eval_scalar(forward_body: &str, input_literal: &str) -> f64 {
    let source = format!(
        "module Repro.Fwd\n\
         sig f: tensor[n, f32] -> f32\n\
         def f(x) = {{\n{forward_body}\n}}\n\
         out = f(to_tensor([{input_literal}]))\n"
    );
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("fwd.ch");
    fs::write(&path, source).expect("write");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run");
    assert!(
        output.status.success(),
        "forward eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    stdout
        .trim()
        .lines()
        .last()
        .and_then(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no scalar in forward output: {stdout}"))
}

/// Run `chelis eval` on `grad(f)(input)` and parse the gradient tensor.
fn eval_grad(forward_body: &str, input_literal: &str) -> Vec<f64> {
    let source = format!(
        "module Repro.Grad\n\
         sig f: tensor[n, f32] -> f32\n\
         def f(x) = {{\n{forward_body}\n}}\n\
         out = grad(f)(to_tensor([{input_literal}]))\n"
    );
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad.ch");
    fs::write(&path, source).expect("write");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run");
    assert!(
        output.status.success(),
        "grad eval failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );
    let stdout = String::from_utf8_lossy(&output.stdout);
    let line = stdout
        .lines()
        .find(|l| l.contains("data=["))
        .unwrap_or_else(|| panic!("no tensor in grad output: {stdout}"));
    line.split_once("data=[")
        .and_then(|(_, r)| r.split_once(']'))
        .map(|(s, _)| {
            s.split(',')
                .map(|t| t.trim().parse::<f64>().unwrap())
                .collect()
        })
        .unwrap()
}

fn fmt_f32_list(xs: &[f64]) -> String {
    xs.iter()
        .map(|x| format!("cast({x:?}, f32)"))
        .collect::<Vec<_>>()
        .join(", ")
}

/// Central-difference finite-difference gradient of the forward loss.
fn finite_difference(forward_body: &str, base: &[f64]) -> Vec<f64> {
    let h = 1e-2;
    let mut fd = Vec::with_capacity(base.len());
    for i in 0..base.len() {
        let mut xp = base.to_vec();
        let mut xm = base.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let lp = eval_scalar(forward_body, &fmt_f32_list(&xp));
        let lm = eval_scalar(forward_body, &fmt_f32_list(&xm));
        fd.push((lp - lm) / (2.0 * h));
    }
    fd
}

// Linear loss: sum(reshape(x, [shape(x,0), 1])) = sum(x). grad == 1 everywhere.
const LINEAR_BODY: &str = "  k = cast(shape(x, cast(0, int32)), int64)\n\
  r = reshape(&x, [k, cast(1, int64)])\n\
  sum(sum(r, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar";

// Nonlinear loss: sum(square(reshape(x, [shape(x,0), 1]))) = sum(x^2).
// grad == 2 x.
const NONLINEAR_BODY: &str = "  k = cast(shape(x, cast(0, int32)), int64)\n\
  r = reshape(&x, [k, cast(1, int64)])\n\
  sq = mul(r, r)\n\
  sum(sum(sq, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar";

/// FD oracle: the runtime-`shape()`-derived reshape target grad must match the
/// central-difference gradient of the same forward loss. Pre-fix this ICE'd in
/// `symbolic_occurrences`; the fix resolves the reshape dim to its declaring
/// Load.
#[test]
fn issue_513_reshape_shape_derived_grad_linear_matches_fd() {
    let base = vec![1.0, 2.0, 3.0, 4.0];
    let grad = eval_grad(LINEAR_BODY, &fmt_f32_list(&base));
    assert_eq!(grad.len(), 4, "grad shape");
    for (i, g) in grad.iter().enumerate() {
        assert!(
            (g - 1.0).abs() < 1e-3,
            "linear grad elem {i}: got {g}, want 1"
        );
    }
    let fd = finite_difference(LINEAR_BODY, &base);
    for (i, (g, f)) in grad.iter().zip(fd.iter()).enumerate() {
        assert!(
            (g - f).abs() < 5e-2,
            "linear grad elem {i}: analytic {g} vs finite-difference {f}"
        );
    }
}

/// FD oracle, nonlinear: grad of sum(square(reshape)) == 2 x, FD-validated.
#[test]
fn issue_513_reshape_shape_derived_grad_nonlinear_matches_fd() {
    let base = vec![1.0, 2.0, 3.0, 4.0];
    let grad = eval_grad(NONLINEAR_BODY, &fmt_f32_list(&base));
    assert_eq!(grad.len(), 4, "grad shape");
    for (i, g) in grad.iter().enumerate() {
        assert!(
            (g - 2.0 * base[i]).abs() < 1e-2,
            "nonlinear grad elem {i}: got {g}, want {}",
            2.0 * base[i]
        );
    }
    let fd = finite_difference(NONLINEAR_BODY, &base);
    for (i, (g, f)) in grad.iter().zip(fd.iter()).enumerate() {
        assert!(
            (g - f).abs() < 5e-2,
            "nonlinear grad elem {i}: analytic {g} vs finite-difference {f}"
        );
    }
}

// --- eval-vs-C-backend agreement ---------------------------------------------

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

fn compile_and_run(build_dir: &Path, stem: &str, driver_src: &str) -> String {
    let driver = build_dir.join("driver.c");
    fs::write(&driver, driver_src).expect("write driver.c");
    let kernel = build_dir.join(format!("{stem}.c"));
    let runtime = build_dir.join("libchelis_runtime.a");
    let bin = build_dir.join("test_bin");
    let compile = StdCommand::new("gcc")
        .args([
            "-O0",
            "-std=c11",
            "-I",
            build_dir.to_str().unwrap(),
            kernel.to_str().unwrap(),
            driver.to_str().unwrap(),
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
    let run = StdCommand::new(&bin).output().expect("run test binary");
    assert!(
        run.status.success(),
        "binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

/// eval-vs-C-backend agreement: the symbolic-`n` C build of the same
/// nonlinear reshape-shape-derived grad must compile, run, and produce the
/// same gradient the eval lane does (2 x). Pre-fix `chelis build --target c`
/// ICE'd in `symbolic_occurrences` on the same orphan `Sym("k")`.
#[test]
fn issue_513_reshape_shape_derived_grad_c_backend_agrees() {
    let source = "module Repro.ReshapeBuild\n\
sig f: tensor[n, f32] -> f32\n\
def f(x) = {\n\
  k = cast(shape(x, cast(0, int32)), int64)\n\
  r = reshape(&x, [k, cast(1, int64)])\n\
  sq = mul(r, r)\n\
  sum(sum(sq, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)\n";
    let (_dir, build_dir) = build_c(source, "reshape513");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {
    int shape[1] = {4};
    chelis_tensor* x = chelis_alloc(1, shape, CHELIS_F32);
    float xd[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    memcpy(x->data, xd, sizeof(xd));
    chelis_tensor* g = out(x);
    if (g->size != 4) { printf("FAIL_SIZE %d\n", g->size); return 1; }
    for (int i = 0; i < 4; i++) printf("%.6f\n", g->data[i]);
    return 0;
}
"#;
    let stdout = compile_and_run(&build_dir, "reshape513", driver);
    let c_grad: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect();
    // Eval lane on the same nonlinear body (2 x = [2, 4, 6, 8]).
    let eval_g = eval_grad(NONLINEAR_BODY, &fmt_f32_list(&[1.0, 2.0, 3.0, 4.0]));
    assert_eq!(c_grad.len(), eval_g.len(), "grad length");
    for (i, (c, e)) in c_grad.iter().zip(eval_g.iter()).enumerate() {
        assert!((c - e).abs() < 1e-3, "elem {i}: C backend {c} != eval {e}");
        assert!(
            (c - 2.0 * (i as f64 + 1.0)).abs() < 1e-3,
            "elem {i}: C backend {c} != analytic {}",
            2.0 * (i as f64 + 1.0)
        );
    }
}
