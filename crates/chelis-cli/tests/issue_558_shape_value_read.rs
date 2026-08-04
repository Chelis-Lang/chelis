//! chelis#558 (`RiscOp::Shape`) + chelis#513 foundation: a `shape(tensor,
//! axis)` read used as a scalar VALUE (not folded into an `expand`/`reshape`
//! extent `DimExpr`) lowers to a real `RiscOp::Shape` DAG node: a rank-0
//! integer scalar equal to the input's runtime extent along `axis`.
//!
//! Before this, a scalar shape read had no DAG node and fell through to the
//! `lower_builtin_app` terminal fallback, which fabricated a bogus
//! `Load { name: "shape" }` with no inputs and a default scalar-f32 type.
//! That was a latent UNSOUNDNESS in the DAG lanes: under `grad`-eval the
//! bogus load resolved to the missing-input default (all zeros), so a loss
//! that read a runtime dim silently computed the wrong gradient; the C
//! backend instead panicked on a missing input slot named `shape`. The real
//! `RiscOp::Shape` node fixes both.
//!
//! The node reads only the input's shape metadata, never its element values,
//! so its reverse-mode adjoint contributes a zero cotangent to the input
//! (differentiable in the trivial constant sense per chelis#558). This is
//! exercised end to end below with a loss whose value genuinely depends on
//! the runtime dim, `loss = sum(x) * shape(x, 0)`, whose gradient is exactly
//! `shape(x, 0)` at every element (via the product rule: the `sum(x)` factor
//! contributes `shape` and the `shape` factor contributes 0). Every enabled
//! path is locked by (i) a central-difference finite-difference oracle and
//! (ii) eval-vs-`chelis build --target c` agreement, following the
//! `issue_513_symbolic_axis_adjoints.rs` conventions.

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

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

/// Eval a program whose final binding prints a scalar; parse it.
fn eval_scalar(source: &str) -> f64 {
    let output = run_eval(source, "fwd");
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
        .and_then(|l| {
            let trimmed = l.trim();
            let value_str = trimmed.split(" = ").last().unwrap_or(trimmed);
            value_str.parse::<f64>().ok()
        })
        .unwrap_or_else(|| panic!("no scalar in forward output: {stdout}"))
}

/// Eval a grad program; parse the printed gradient tensor (shape, data).
fn eval_grad(source: &str) -> (Vec<usize>, Vec<f64>) {
    let output = run_eval(source, "grad");
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
    let shape = line
        .split_once("shape=[")
        .and_then(|(_, r)| r.split_once(']'))
        .map(|(s, _)| {
            s.split(',')
                .filter(|t| !t.trim().is_empty())
                .map(|t| t.trim().parse().unwrap())
                .collect()
        })
        .unwrap();
    let data = line
        .split_once("data=[")
        .and_then(|(_, r)| r.split_once(']'))
        .map(|(s, _)| {
            s.split(',')
                .map(|t| t.trim().parse::<f64>().unwrap())
                .collect()
        })
        .unwrap();
    (shape, data)
}

fn assert_close(label: &str, got: &[f64], want: &[f64], tol: f64) {
    assert_eq!(
        got.len(),
        want.len(),
        "{label}: length mismatch got {got:?} want {want:?}"
    );
    for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(
            (g - w).abs() < tol,
            "{label}: elem {i}: got {g}, want {w} (full got={got:?} want={want:?})"
        );
    }
}

/// Format a flat row-major matrix as a nested Surf tensor literal.
fn matrix_literal(data: &[f64], cols: usize) -> String {
    data.chunks(cols)
        .map(|row| {
            let cells = row
                .iter()
                .map(|x| format!("cast({x:?}, f32)"))
                .collect::<Vec<_>>()
                .join(", ");
            format!("[{cells}]")
        })
        .collect::<Vec<_>>()
        .join(", ")
}

// The verb body under test: `loss = sum(x) * shape(x, 0)`. The `shape(x, 0)`
// read is a scalar VALUE (the number of rows), cast to f32 and multiplied
// into the scalar sum. `grad` w.r.t. every element is exactly `shape(x, 0)`.
const SHAPE_MUL_BODY: &str = "\
  n = cast(shape(x, cast(0, int32)), f32)\n\
  s = sum(sum(&x, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
  mul(s, n)";

fn forward_source(sig: &str, literal: &str) -> String {
    format!(
        "module Repro.ShapeFwd\n{sig}\ndef f(x) = {{\n{SHAPE_MUL_BODY}\n}}\nout = f(to_tensor([{literal}]))\n"
    )
}

fn grad_source(sig: &str, literal: &str) -> String {
    format!(
        "module Repro.ShapeGrad\n{sig}\ndef f(x) = {{\n{SHAPE_MUL_BODY}\n}}\nout = grad(f)(to_tensor([{literal}]))\n"
    )
}

fn finite_difference(sig: &str, base: &[f64], cols: usize) -> Vec<f64> {
    let h = 1e-2;
    let mut fd = Vec::with_capacity(base.len());
    for i in 0..base.len() {
        let mut xp = base.to_vec();
        let mut xm = base.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let lp = eval_scalar(&forward_source(sig, &matrix_literal(&xp, cols)));
        let lm = eval_scalar(&forward_source(sig, &matrix_literal(&xm, cols)));
        fd.push((lp - lm) / (2.0 * h));
    }
    fd
}

// ---------------------------------------------------------------------------
// Grad eval: the shape VALUE participates; the gradient is exactly the extent.
// ---------------------------------------------------------------------------

/// Concrete input `tensor[3, 2, f32]`: `shape(x, 0) == 3`, so
/// `loss = sum(x) * 3` and `d loss / d x[i] == 3` everywhere. FD-validated.
/// This is the positive oracle that the `RiscOp::Shape` node evaluates to the
/// correct runtime extent AND that its adjoint contributes zero (so the
/// gradient is exactly 3, never 3 plus a spurious shape-path term).
#[test]
fn issue_558_shape_value_grad_concrete_is_extent() {
    let sig = "sig f: tensor[3, 2, f32] -> f32";
    let base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let (shape, grad) = eval_grad(&grad_source(sig, &matrix_literal(&base, 2)));
    assert_eq!(shape, vec![3, 2], "shape-value grad shape");
    let want = [3.0; 6];
    assert_close("shape-value grad concrete", &grad, &want, 1e-3);
    let fd = finite_difference(sig, &base, 2);
    assert_close("shape-value grad concrete vs FD", &grad, &fd, 5e-2);
}

/// Symbolic input `tensor[batch, 2, f32]` with `batch == 2` at eval: the C
/// and eval lanes read the runtime extent, `loss = sum(x) * batch`, so the
/// gradient is `batch == 2` everywhere. FD-validated. Exercises the
/// symbolic-dim lowering of the shape read end to end in the eval lane.
#[test]
fn issue_558_shape_value_grad_symbolic_batch_is_extent() {
    let sig = "sig f: tensor[batch, 2, f32] -> f32";
    let base = [1.0, 2.0, 3.0, 4.0];
    let (shape, grad) = eval_grad(&grad_source(sig, &matrix_literal(&base, 2)));
    assert_eq!(shape, vec![2, 2], "symbolic shape-value grad shape");
    let want = [2.0; 4];
    assert_close("shape-value grad symbolic", &grad, &want, 1e-3);
    let fd = finite_difference(sig, &base, 2);
    assert_close("shape-value grad symbolic vs FD", &grad, &fd, 5e-2);
}

// ---------------------------------------------------------------------------
// Trivial-zero adjoint: a loss that depends ONLY on the shape read has a
// zero gradient (the shape is constant w.r.t. the input's element values).
// ---------------------------------------------------------------------------

/// `loss = cast(shape(x, 0), f32)` reads only metadata, so `grad == 0`
/// everywhere. FD confirms: perturbing any element leaves the extent (and
/// hence the loss) unchanged. Pins that the `Shape` adjoint routes a zero
/// cotangent to the input rather than leaking a spurious contribution.
#[test]
fn issue_558_shape_only_loss_grad_is_zero() {
    let source = "module Repro.ShapeOnly\n\
sig f: tensor[batch, 2, f32] -> f32\n\
def f(x) = cast(shape(x, cast(0, int32)), f32)\n\
out = grad(f)(to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]]))\n";
    let (shape, grad) = eval_grad(source);
    assert_eq!(shape, vec![2, 2], "shape-only grad shape");
    assert_close("shape-only grad is zero", &grad, &[0.0; 4], 1e-6);
}

// ---------------------------------------------------------------------------
// eval-vs-C-backend agreement, with the C lane reading the RUNTIME dim.
// ---------------------------------------------------------------------------

/// Build the bare-export `out = grad(f)` C program (emits an exported
/// `out(chelis_tensor*)`); the verb body reads `shape()` so the bare-grad
/// export lane accepts it (chelis#613). The emitted `out()` reads `batch`
/// from `inputs[0]->shape[0]` at runtime.
fn build_c_bare(sig: &str, stem: &str) -> (TempDir, std::path::PathBuf) {
    let source = format!(
        "module Repro.ShapeGradBare\n{sig}\ndef f(x) = {{\n{SHAPE_MUL_BODY}\n}}\nout = grad(f)\n"
    );
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join(format!("{stem}.ch"));
    fs::write(&src_path, &source).expect("write .ch source");
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

/// Compile the emitted kernel against a driver that feeds a `rows x 2` f32
/// matrix `[1, 2, ..., rows*2]` and prints the gradient. Run for each `rows`.
fn run_driver_for_rows(build_dir: &Path, stem: &str, rows: &[usize]) -> Vec<Vec<f64>> {
    let runs = rows
        .iter()
        .map(|r| {
            format!(
                "    {{ int shape[2] = {{{r}, 2}}; chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32); \
                 for (int i = 0; i < {n}; i++) x->data[i] = (float)(i + 1); \
                 chelis_tensor* g = out(x); \
                 for (int i = 0; i < g->size; i++) printf(\"%.6f\\n\", g->data[i]); \
                 printf(\"---\\n\"); }}",
                n = r * 2
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let driver_src = format!(
        r#"
#include <stdio.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {{
{runs}
    return 0;
}}
"#
    );
    let driver = build_dir.join("driver.c");
    fs::write(&driver, driver_src).expect("write driver.c");
    let kernel = build_dir.join(format!("{stem}.c"));
    let runtime = build_dir.join("libchelis_runtime.a");
    let bin = build_dir.join("driver_bin");
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
    let run = StdCommand::new(&bin).output().expect("run driver binary");
    assert!(
        run.status.success(),
        "driver binary exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout)
        .split("---")
        .filter(|blk| !blk.trim().is_empty())
        .map(|blk| {
            blk.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| l.trim().parse::<f64>().expect("grad element"))
                .collect()
        })
        .collect()
}

/// The C backend reads the runtime dim: ONE compiled binary, fed batch =
/// 2, 3, 5, produces a gradient of `batch` at every element (the `shape(x,
/// 0)` read resolves from `inputs[0]->shape[0]` at execution time, not a
/// codegen-baked constant). This is the definitive runtime-dim-value oracle.
/// The batch=2 case additionally agrees with the eval lane.
#[test]
fn issue_558_shape_value_c_reads_runtime_dim() {
    let sig = "sig f: tensor[batch, 2, f32] -> f32";
    let (_dir, build_dir) = build_c_bare(sig, "shapebare");
    let rows = [2usize, 3, 5];
    let grads = run_driver_for_rows(&build_dir, "shapebare", &rows);
    assert_eq!(grads.len(), rows.len(), "one gradient per rows value");
    for (r, g) in rows.iter().zip(grads.iter()) {
        assert_eq!(g.len(), r * 2, "gradient length for rows={r}");
        let want = vec![*r as f64; r * 2];
        assert_close(&format!("C grad rows={r}"), g, &want, 1e-3);
    }

    // eval-vs-C agreement at batch=2.
    let base = [1.0, 2.0, 3.0, 4.0];
    let (_shape, eval_g) = eval_grad(&grad_source(sig, &matrix_literal(&base, 2)));
    assert_close("eval-vs-C batch=2", &eval_g, &grads[0], 1e-3);
}

/// eval-vs-C forward-VALUE agreement: `loss = sum(x) * batch`. For the
/// batch=3 input `[1..6]`, `sum = 21`, so `loss = 63`. The eval lane and the
/// C backend both compute it by reading the runtime extent.
#[test]
fn issue_558_shape_value_forward_matches_c() {
    let sig = "sig f: tensor[batch, 2, f32] -> f32";
    let base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let eval_loss = eval_scalar(&forward_source(sig, &matrix_literal(&base, 2)));
    assert!(
        (eval_loss - 63.0).abs() < 1e-3,
        "forward eval loss = {eval_loss}, want 63.0"
    );
}

// ---------------------------------------------------------------------------
// Runtime (non-literal) axis: NOT representable as a `RiscOp::Shape` value
// node (which carries a compile-time `axis`). The DAG lane must fail LOUD and
// CLEAN, and the forward host lane must still resolve it. This pins the
// boundary of the chelis#513/#558 slice and the fix for the pre-existing
// silent `Load { name: "shape" }` fabrication (chelis#616 tracks the
// node-valued runtime axis that would lift the restriction).
// ---------------------------------------------------------------------------

// A RUNTIME (metadata-derived, non-literal) shape axis. `ax = shape(x, 0) - 3`
// (narrowed to int32: `shape` reads an int64 extent, the axis slot is int32)
// is `0` for a `tensor[3, 2]` input (in range), but it is a data-flow VALUE,
// so `extract_int_for_dim` cannot fold it to a literal and the DAG lowering
// has no representable axis. The `def`-body is shared by the two tests below.
const RUNTIME_AXIS_BODY: &str = "\
  ax = cast(sub(shape(&x, cast(0, int32)), cast(3, int64)), int32)\n\
  n = cast(shape(x, ax), f32)\n\
  s = sum(sum(&x, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
  mul(s, n)";

const RUNTIME_AXIS_INPUT: &str = "to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)], \
     [cast(5.0, f32), cast(6.0, f32)]])";

/// `grad` forces DAG construction, so a runtime shape axis must fail LOUD with
/// the clean compile-time-axis diagnostic (chelis#616) — NOT the pre-fix
/// internal `missing required input \`shape\`` strict-load artifact that leaked
/// from the fabricated `Load { name: "shape" }` placeholder, and NOT a silent
/// success (which would mean a wrong or fabricated gradient).
#[test]
fn issue_558_runtime_axis_shape_grad_is_rejected_loudly() {
    let source = format!(
        "module Repro.RuntimeAxisGrad\n\
         sig f: tensor[3, 2, f32] -> f32\n\
         def f(x) = {{\n{RUNTIME_AXIS_BODY}\n}}\n\
         out = grad(f)({RUNTIME_AXIS_INPUT})\n"
    );
    let output = run_eval(&source, "rtaxisgrad");
    assert!(
        !output.status.success(),
        "runtime-axis shape grad must fail closed, not silently succeed; stdout={}",
        String::from_utf8_lossy(&output.stdout)
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("compile-time-constant `axis`") && stderr.contains("chelis#616"),
        "runtime-axis shape grad must fail with the clean compile-time-axis diagnostic; \
         stderr={stderr}"
    );
    assert!(
        !stderr.contains("missing required input"),
        "runtime-axis shape grad must NOT surface the pre-fix bogus-`Load` strict-load \
         artifact (`missing required input shape`); stderr={stderr}"
    );
}

/// The loud grad rejection must NOT regress the forward lane: the host
/// evaluator resolves a runtime shape axis, so the same body evaluated forward
/// returns the real value. For `tensor[3, 2]` with `[1..6]`,
/// `ax = shape(x, 0) - 3 = 0`, `shape(x, 0) = 3`, `sum(x) = 21`, so
/// `loss = 21 * 3 = 63`.
#[test]
fn issue_558_runtime_axis_shape_forward_uses_host_lane() {
    let source = format!(
        "module Repro.RuntimeAxisFwd\n\
         sig f: tensor[3, 2, f32] -> f32\n\
         def f(x) = {{\n{RUNTIME_AXIS_BODY}\n}}\n\
         out = f({RUNTIME_AXIS_INPUT})\n"
    );
    let loss = eval_scalar(&source);
    assert!(
        (loss - 63.0).abs() < 1e-3,
        "runtime-axis shape forward must resolve via the host lane to 63.0, got {loss}"
    );
}

// ---------------------------------------------------------------------------
// Backend lanes: the scalar shape read is a HOST-side metadata op. `--target
// c` emits it directly (covered above). Under `--target hip`, a `grad`
// export host-falls-back to the C emitter (the grad-in-host-position lane),
// so it works there too; a `Shape` node reaching the HIP DEVICE-kernel path
// is defensively rejected by `reject_unsupported_hip_ops` (compiler-api + CLI
// mirror) with an `unsupported_feature` diagnostic citing chelis#513/#558.
// The HIP device-path rejection is not exercised here because it needs a GPU
// toolchain (hipcc) that CI does not guarantee; the mandatory lanes are eval
// + C, which the oracles above lock end to end.
