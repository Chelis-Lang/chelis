//! Issue #513 gap 3 (wildcard / symbolic-input adjoints), the STRUCTURAL
//! slice: movement/reduction adjoints whose construction previously demanded
//! a concrete size (`dim_size`) for EVERY input axis, panicking on a
//! `Named(_, None)` symbolic dim even when the symbolic axis does not
//! participate in the op at all.
//!
//! Three sub-slices are enabled, each exact (no shape() VALUE read needed,
//! the adjoint carries the symbolic dim through structurally):
//!
//!   1. `Stride` adjoint over a symbolic NON-strided axis (`stride` of a
//!      `tensor[batch, k, f32]` along the concrete axis): the upsample
//!      cascade only needs the strided axis's size; symbolic bystander axes
//!      flow through `Reshape`/`Pad` unchanged and the trim `Shrink` uses
//!      the `SHRINK_TO_END` full-axis sentinel.
//!   2. `ProdReduce` adjoint over a symbolic NON-reduced axis: the per-slice
//!      `Shrink` bounds use the sentinel on bystander symbolic axes; only
//!      the reduced axis needs a concrete size (that stays fail-closed).
//!   3. `reshape` whose target dims are integer ARITHMETIC over
//!      statically-sized `shape()` reads (the school im2col witness shape,
//!      `reshape(p, [mul(b_d, a_d), 1i64])`): `extract_reshape_dim_list` now
//!      const-folds the arithmetic to a `Lit` when every leaf is static, so
//!      the backward `Expand`/`Sum` no longer inherits the checker's
//!      `Named("*")` wildcard (the dag.rs symbolic-occurrences ICE).
//!
//! Every enabled path is locked by (i) a central-difference finite-difference
//! oracle and (ii) eval-vs-`chelis build --target c` backend agreement,
//! following `issue_513_reshape_shape_derived_grad.rs`. Paths that would need
//! a runtime shape() VALUE in an ADJOINT construction (symbolic strided
//! axis, symbolic reduced prod axis, runtime shrink bounds on a symbolic
//! axis) REMAIN fail-closed with loud diagnostics, pinned by the negative
//! tests at the bottom.
//!
//! chelis#616 update: a shape()-derived arithmetic reshape target the
//! exactness gate refuses to FOLD (symbolic dim leaf, negative operand,
//! non-positive divisor, or overflow) now LOWERS as a runtime (node-valued)
//! target extent instead of failing at lowering. The written expression is
//! evaluated with true runtime semantics in both lanes, and the reshape
//! numel invariant is enforced at run time — a clean eval error and a C
//! runtime abort (never the pre-hardening silent wildcard acceptance, and
//! never a lowering-time refusal of a valid program). The symbolic-sig
//! im2col form is now a passing grad oracle; the ill-formed [4, 4]-over-8
//! form is a runtime numel-mismatch error in both lanes. A target the fold
//! PROVES negative still fails loud at lowering (proven-invalid program).

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

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

/// Eval a forward program whose final binding prints a scalar; parse it.
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
            // [05-OBS-6]: strip `name = ` prefix if present.
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

/// Assemble the forward-loss program for a `sig f: ... -> f32` verb.
fn forward_source(sig: &str, body: &str, literal: &str) -> String {
    format!("module Repro.Fwd\n{sig}\ndef f(x) = {{\n{body}\n}}\nout = f(to_tensor([{literal}]))\n")
}

/// Assemble the grad program for the same verb.
fn grad_source(sig: &str, body: &str, literal: &str) -> String {
    format!(
        "module Repro.Grad\n{sig}\ndef f(x) = {{\n{body}\n}}\nout = grad(f)(to_tensor([{literal}]))\n"
    )
}

/// Central-difference finite-difference gradient of the forward loss over a
/// flat row-major matrix input.
fn finite_difference(sig: &str, body: &str, base: &[f64], cols: usize) -> Vec<f64> {
    let h = 1e-2;
    let mut fd = Vec::with_capacity(base.len());
    for i in 0..base.len() {
        let mut xp = base.to_vec();
        let mut xm = base.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let lp = eval_scalar(&forward_source(sig, body, &matrix_literal(&xp, cols)));
        let lm = eval_scalar(&forward_source(sig, body, &matrix_literal(&xm, cols)));
        fd.push((lp - lm) / (2.0 * h));
    }
    fd
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

// ---------------------------------------------------------------------------
// Sub-slice 1: Stride adjoint, symbolic batch axis, concrete strided axis.
// stride(x, 1i64, 2i64) over tensor[batch, 4] keeps columns {0, 2}.
// ---------------------------------------------------------------------------

const STRIDE_SIG: &str = "sig f: tensor[batch, 4, f32] -> f32";

const STRIDE_LINEAR_BODY: &str = "  s = stride(&x, cast(1, int64), cast(2, int64))\n\
  sum(sum(s, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar";

const STRIDE_NONLINEAR_BODY: &str = "  s = stride(&x, cast(1, int64), cast(2, int64))\n\
  sq = mul(s, s)\n\
  sum(sum(sq, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar";

const STRIDE_BASE: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

/// FD oracle, linear: loss = sum(stride(x, 1i64, 2i64)), so the gradient is 1 at
/// the kept columns {0, 2} of every (symbolic-batch) row and 0 elsewhere.
/// Pre-fix this failed loud: "cannot determine size for symbolic dimension
/// `batch`" from the stride adjoint's all-axes dim_size sweep.
#[test]
fn issue_513_stride_symbolic_batch_grad_linear_matches_fd() {
    let (shape, grad) = eval_grad(&grad_source(
        STRIDE_SIG,
        STRIDE_LINEAR_BODY,
        &matrix_literal(&STRIDE_BASE, 4),
    ));
    assert_eq!(shape, vec![2, 4], "stride grad shape");
    let want = [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0];
    assert_close("stride linear grad", &grad, &want, 1e-3);
    let fd = finite_difference(STRIDE_SIG, STRIDE_LINEAR_BODY, &STRIDE_BASE, 4);
    assert_close("stride linear grad vs FD", &grad, &fd, 5e-2);
}

/// FD oracle, nonlinear: loss = sum(square(stride(x, 1i64, 2i64))), gradient is 2x
/// at kept columns and 0 elsewhere.
#[test]
fn issue_513_stride_symbolic_batch_grad_nonlinear_matches_fd() {
    let (shape, grad) = eval_grad(&grad_source(
        STRIDE_SIG,
        STRIDE_NONLINEAR_BODY,
        &matrix_literal(&STRIDE_BASE, 4),
    ));
    assert_eq!(shape, vec![2, 4], "stride nonlinear grad shape");
    let want = [2.0, 0.0, 6.0, 0.0, 10.0, 0.0, 14.0, 0.0];
    assert_close("stride nonlinear grad", &grad, &want, 1e-3);
    let fd = finite_difference(STRIDE_SIG, STRIDE_NONLINEAR_BODY, &STRIDE_BASE, 4);
    assert_close("stride nonlinear grad vs FD", &grad, &fd, 5e-2);
}

/// ADVERSARIAL: overshooting stride (step 3 over a size-4 axis, so the
/// upsample cascade's merged axis is 2 * 3 = 6 and the trim discards the
/// trailing 2 slots) with the symbolic bystander axis in play. Locks the
/// `m_a * step > n_a` trailing-trim path next to the sentinel bounds:
/// kept columns are {0, 3}, gradient 2x there and 0 elsewhere.
#[test]
fn issue_513_stride_overshoot_symbolic_batch_grad_matches_fd() {
    let body = "  s = stride(&x, cast(1, int64), cast(3, int64))\n\
  sq = mul(s, s)\n\
  sum(sum(sq, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar";
    let (shape, grad) = eval_grad(&grad_source(
        STRIDE_SIG,
        body,
        &matrix_literal(&STRIDE_BASE, 4),
    ));
    assert_eq!(shape, vec![2, 4], "stride overshoot grad shape");
    let want = [2.0, 0.0, 0.0, 8.0, 10.0, 0.0, 0.0, 16.0];
    assert_close("stride overshoot grad", &grad, &want, 1e-3);
    let fd = finite_difference(STRIDE_SIG, body, &STRIDE_BASE, 4);
    assert_close("stride overshoot grad vs FD", &grad, &fd, 5e-2);
}

// ---------------------------------------------------------------------------
// Sub-slice 2: ProdReduce adjoint, symbolic batch axis, concrete reduced axis.
// ---------------------------------------------------------------------------

const PROD_SIG: &str = "sig f: tensor[batch, 3, f32] -> f32";

const PROD_BODY: &str = "  p = prod_reduce(&x, cast(1, int32))\n\
  sum(p, cast(0, int32)) |> tensor_to_scalar";

const PROD_BASE: [f64; 6] = [0.5, 1.5, 2.0, 1.0, 2.5, 0.5];

/// FD oracle: loss = sum over batch of prod over the concrete axis; the
/// gradient at x[i][j] is the leave-one-out product of row i. Pre-fix this
/// failed loud: "cannot determine size for symbolic dimension `batch`" from
/// the prod_reduce adjoint's bystander-axis dim_size in the slice bounds.
#[test]
fn issue_513_prod_reduce_symbolic_batch_grad_matches_fd() {
    let (shape, grad) = eval_grad(&grad_source(
        PROD_SIG,
        PROD_BODY,
        &matrix_literal(&PROD_BASE, 3),
    ));
    assert_eq!(shape, vec![2, 3], "prod_reduce grad shape");
    // Leave-one-out products: row0 = [1.5*2, 0.5*2, 0.5*1.5],
    // row1 = [2.5*0.5, 1.0*0.5, 1.0*2.5].
    let want = [3.0, 1.0, 0.75, 1.25, 0.5, 2.5];
    assert_close("prod_reduce grad", &grad, &want, 1e-3);
    let fd = finite_difference(PROD_SIG, PROD_BODY, &PROD_BASE, 3);
    assert_close("prod_reduce grad vs FD", &grad, &fd, 5e-2);
}

/// ADVERSARIAL: zero elements in the input on the ENABLED symbolic-bystander
/// path. The naive `g * prod / x_i` adjoint divides by zero here; the
/// prefix*suffix construction must produce the exact finite leave-one-out
/// products through the sentinel-bounded slices. Row 0 = [2, 0, 4] has one
/// zero (gradient nonzero ONLY at the zero position: 2 * 4 = 8); row 1 =
/// [0, 0, 5] has two zeros (every leave-one-out product contains a zero, so
/// the whole row's gradient is 0).
#[test]
fn issue_513_prod_reduce_zero_element_symbolic_batch_grad_matches_fd() {
    let base = [2.0, 0.0, 4.0, 0.0, 0.0, 5.0];
    let (shape, grad) = eval_grad(&grad_source(PROD_SIG, PROD_BODY, &matrix_literal(&base, 3)));
    assert_eq!(shape, vec![2, 3], "prod_reduce zero-element grad shape");
    for v in &grad {
        assert!(v.is_finite(), "prod_reduce grad must be finite; got {v}");
    }
    let want = [0.0, 8.0, 0.0, 0.0, 0.0, 0.0];
    assert_close("prod_reduce zero-element grad", &grad, &want, 1e-3);
    let fd = finite_difference(PROD_SIG, PROD_BODY, &base, 3);
    assert_close("prod_reduce zero-element grad vs FD", &grad, &fd, 5e-2);
}

// ---------------------------------------------------------------------------
// Sub-slice 3: reshape target dims that are static integer ARITHMETIC over
// shape() reads (school im2col witness: permute + reshape([mul(b_d, a_d), 1])).
// ---------------------------------------------------------------------------

/// Concrete-input verb whose reshape target is `[mul(b_d, a_d), 1]`, an
/// arithmetic expression over shape() reads. No `sig` line: the def carries
/// the concrete annotation directly.
fn reshape_arith_source(nonlinear: bool, literal: &str, grad: bool) -> String {
    let sq = if nonlinear {
        "  sq = mul(r, r)\n  sum(sum(sq, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar"
    } else {
        "  sum(sum(r, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar"
    };
    let call = if grad {
        format!("out = grad(f)(to_tensor([{literal}]))")
    } else {
        format!("out = f(to_tensor([{literal}]))")
    };
    format!(
        "module Repro.ReshapeArith\n\
         def f(x: tensor[2, 2, f32]) -> f32 = {{\n\
         \x20 a_d = cast(shape(x, cast(0, int32)), int64)\n\
         \x20 b_d = cast(shape(x, cast(1, int32)), int64)\n\
         \x20 p = permute(&x, cast(1, int32), cast(0, int32))\n\
         \x20 r = reshape(p, [mul(b_d, a_d), cast(1, int64)])\n\
         {sq}\n\
         }}\n\
         {call}\n"
    )
}

const RESHAPE_BASE: [f64; 4] = [1.0, 2.0, 3.0, 4.0];

/// FD oracle, linear: loss = sum(reshape(permute(x), [b*a, 1i64])) = sum(x),
/// gradient all ones. Pre-fix the unresolvable `mul(...)` target fell back to
/// the checker's `Named("*")` wildcard dims and the backward Expand ICEd in
/// `symbolic_occurrences` ("symbolic dim `*` ... no Load input declares it").
#[test]
fn issue_513_reshape_arith_target_grad_linear_matches_fd() {
    let (shape, grad) = eval_grad(&reshape_arith_source(
        false,
        &matrix_literal(&RESHAPE_BASE, 2),
        true,
    ));
    assert_eq!(shape, vec![2, 2], "reshape-arith grad shape");
    assert_close("reshape-arith linear grad", &grad, &[1.0; 4], 1e-3);
    // FD from the forward loss.
    let h = 1e-2;
    let mut fd = Vec::new();
    for i in 0..RESHAPE_BASE.len() {
        let mut xp = RESHAPE_BASE.to_vec();
        let mut xm = RESHAPE_BASE.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let lp = eval_scalar(&reshape_arith_source(false, &matrix_literal(&xp, 2), false));
        let lm = eval_scalar(&reshape_arith_source(false, &matrix_literal(&xm, 2), false));
        fd.push((lp - lm) / (2.0 * h));
    }
    assert_close("reshape-arith linear grad vs FD", &grad, &fd, 5e-2);
}

/// FD oracle, nonlinear: loss = sum(square(...)) = sum(x^2), gradient 2x
/// (element order in x's own layout; permute/reshape only reindex).
#[test]
fn issue_513_reshape_arith_target_grad_nonlinear_matches_fd() {
    let (shape, grad) = eval_grad(&reshape_arith_source(
        true,
        &matrix_literal(&RESHAPE_BASE, 2),
        true,
    ));
    assert_eq!(shape, vec![2, 2], "reshape-arith nonlinear grad shape");
    let want: Vec<f64> = RESHAPE_BASE.iter().map(|x| 2.0 * x).collect();
    assert_close("reshape-arith nonlinear grad", &grad, &want, 1e-3);
    let h = 1e-2;
    let mut fd = Vec::new();
    for i in 0..RESHAPE_BASE.len() {
        let mut xp = RESHAPE_BASE.to_vec();
        let mut xm = RESHAPE_BASE.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let lp = eval_scalar(&reshape_arith_source(true, &matrix_literal(&xp, 2), false));
        let lm = eval_scalar(&reshape_arith_source(true, &matrix_literal(&xm, 2), false));
        fd.push((lp - lm) / (2.0 * h));
    }
    assert_close("reshape-arith nonlinear grad vs FD", &grad, &fd, 5e-2);
}

/// General 2x4 verb whose reshape target dims are caller-supplied
/// arithmetic expressions over the shape-read bindings `a_d` (= 2) and
/// `b_d` (= 4). Used by the division/mod fold oracles and the fail-closed
/// pins below.
fn reshape_arith2_source(target: &str, literal: &str, grad: bool) -> String {
    let call = if grad {
        format!("out = grad(f)(to_tensor([{literal}]))")
    } else {
        format!("out = f(to_tensor([{literal}]))")
    };
    format!(
        "module Repro.ReshapeArith2\n\
         def f(x: tensor[2, 4, f32]) -> f32 = {{\n\
         \x20 a_d = cast(shape(x, cast(0, int32)), int64)\n\
         \x20 b_d = cast(shape(x, cast(1, int32)), int64)\n\
         \x20 p = permute(&x, cast(1, int32), cast(0, int32))\n\
         \x20 r = reshape(p, [{target}])\n\
         \x20 sq = mul(r, r)\n\
         \x20 sum(sum(sq, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
         }}\n\
         {call}\n"
    )
}

const RESHAPE2_BASE: [f64; 8] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

/// The gated-domain division/mod arms of the fold: target
/// `[floor_div(mul(b_d, a_d), 4), add(sub(b_d, a_d), mod(a_d, b_d))]`
/// = `[8 / 4, (4 - 2) + (2 mod 4)]` = `[2, 4]` (all operands non-negative,
/// all divisors positive, so floor, trunc, and euclidean semantics agree
/// and the fold is exact). Loss = sum of squares, gradient 2x. Pre-fold
/// these arms had no positive coverage at all.
const DIV_MOD_TARGET: &str = "floor_div(mul(b_d, a_d), cast(4, int64)), \
     add(sub(b_d, a_d), mod(a_d, b_d))";

#[test]
fn issue_513_reshape_div_mod_target_grad_matches_fd() {
    let (shape, grad) = eval_grad(&reshape_arith2_source(
        DIV_MOD_TARGET,
        &matrix_literal(&RESHAPE2_BASE, 4),
        true,
    ));
    assert_eq!(shape, vec![2, 4], "div/mod reshape-arith grad shape");
    let want: Vec<f64> = RESHAPE2_BASE.iter().map(|x| 2.0 * x).collect();
    assert_close("div/mod reshape-arith grad", &grad, &want, 1e-3);
    let h = 1e-2;
    let mut fd = Vec::new();
    for i in 0..RESHAPE2_BASE.len() {
        let mut xp = RESHAPE2_BASE.to_vec();
        let mut xm = RESHAPE2_BASE.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let lp = eval_scalar(&reshape_arith2_source(
            DIV_MOD_TARGET,
            &matrix_literal(&xp, 4),
            false,
        ));
        let lm = eval_scalar(&reshape_arith2_source(
            DIV_MOD_TARGET,
            &matrix_literal(&xm, 4),
            false,
        ));
        fd.push((lp - lm) / (2.0 * h));
    }
    assert_close("div/mod reshape-arith grad vs FD", &grad, &fd, 5e-2);
}

// ---------------------------------------------------------------------------
// eval-vs-C-backend agreement for all three enabled sub-slices.
// ---------------------------------------------------------------------------

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

/// Shared 2D driver: feed a rows x cols f32 matrix, print each gradient
/// element on its own line.
fn matrix_driver(rows: usize, cols: usize, values: &[f64]) -> String {
    let n = rows * cols;
    let init = values
        .iter()
        .map(|v| format!("{v:?}f"))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {{
    int64_t shape[2] = {{{rows}, {cols}}};
    chelis_tensor* x = chelis_alloc(2, shape, CHELIS_F32);
    float xd[{n}] = {{{init}}};
    memcpy(x->data, xd, sizeof(xd));
    chelis_tensor* g = out(x);
    if (g->size != {n}) {{ printf("FAIL_SIZE %lld\n", (long long)g->size); return 1; }}
    for (int i = 0; i < {n}; i++) printf("%.6f\n", g->data[i]);
    return 0;
}}
"#
    )
}

fn parse_lines(stdout: &str) -> Vec<f64> {
    stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect()
}

/// gcc-compile a MAIN-CARRYING emitted program (the `out = grad(f)(input)`
/// applied form) directly against the emitted runtime and run it, returning
/// stdout. The bare-export `out = grad(f)` form is rejected ("`grad` is not
/// supported by IR evaluation yet") whenever the verb body contains no
/// `shape()` read, at ANY rank (chelis#613, a pre-existing bare-grad-export
/// lane limitation unrelated to this issue; the reshape-arith oracle below
/// bare-exports fine because its body reads `shape()`). The stride and
/// prod_reduce verbs here have shape()-free bodies, so their C oracles use
/// the applied form, whose emitted C still reads `batch` from the runtime
/// input tensor (`inputs[0]->shape[0]`, and the build log reports
/// "Symbolic dims: batch") and so exercises the symbolic-dim lowering end
/// to end.
fn compile_and_run_main(build_dir: &Path, stem: &str) -> String {
    let kernel = build_dir.join(format!("{stem}.c"));
    let runtime = build_dir.join("libchelis_runtime.a");
    let bin = build_dir.join("self_bin");
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
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        run.status.success(),
        "emitted program exited non-zero: stdout={} stderr={}",
        String::from_utf8_lossy(&run.stdout),
        String::from_utf8_lossy(&run.stderr)
    );
    String::from_utf8_lossy(&run.stdout).into_owned()
}

/// Parse the `tensor(shape=[...], data=[...])` line an emitted main prints.
fn parse_printed_tensor(stdout: &str) -> Vec<f64> {
    let line = stdout
        .lines()
        .find(|l| l.contains("data=["))
        .unwrap_or_else(|| panic!("no tensor in emitted-program output: {stdout}"));
    line.split_once("data=[")
        .and_then(|(_, r)| r.split_once(']'))
        .map(|(s, _)| {
            s.split(',')
                .map(|t| t.trim().parse::<f64>().unwrap())
                .collect()
        })
        .unwrap()
}

/// eval-vs-C agreement for the symbolic-batch stride grad. The emitted C
/// reads `batch` from the runtime input (the DAG stays symbolic; the build
/// log reports "Symbolic dims: batch"); pre-fix this build failed on the
/// same stride-adjoint dim_size panic as the eval lane.
#[test]
fn issue_513_stride_symbolic_batch_grad_c_backend_agrees() {
    let source = grad_source(
        STRIDE_SIG,
        STRIDE_NONLINEAR_BODY,
        &matrix_literal(&STRIDE_BASE, 4),
    );
    let (_dir, build_dir) = build_c(&source, "stride513");
    let stdout = compile_and_run_main(&build_dir, "stride513");
    let c_grad = parse_printed_tensor(&stdout);
    let (_, eval_g) = eval_grad(&source);
    assert_close("stride grad C vs eval", &c_grad, &eval_g, 1e-3);
    let want = [2.0, 0.0, 6.0, 0.0, 10.0, 0.0, 14.0, 0.0];
    assert_close("stride grad C vs analytic", &c_grad, &want, 1e-3);
}

/// eval-vs-C agreement for the symbolic-batch prod_reduce grad.
#[test]
fn issue_513_prod_reduce_symbolic_batch_grad_c_backend_agrees() {
    // Integer-valued base keeps the f32 C lane exact.
    let base = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let source = grad_source(PROD_SIG, PROD_BODY, &matrix_literal(&base, 3));
    let (_dir, build_dir) = build_c(&source, "prod513");
    let stdout = compile_and_run_main(&build_dir, "prod513");
    let c_grad = parse_printed_tensor(&stdout);
    let (_, eval_g) = eval_grad(&source);
    assert_close("prod grad C vs eval", &c_grad, &eval_g, 1e-3);
    let want = [6.0, 3.0, 2.0, 30.0, 24.0, 20.0];
    assert_close("prod grad C vs analytic", &c_grad, &want, 1e-3);
}

/// eval-vs-C agreement for the gated-domain division/mod reshape-target
/// grad (the floor_div / mod fold arms).
#[test]
fn issue_513_reshape_div_mod_target_grad_c_backend_agrees() {
    let source = reshape_arith2_source(DIV_MOD_TARGET, "", false)
        .replace("out = f(to_tensor([]))", "out = grad(f)");
    let (_dir, build_dir) = build_c(&source, "divmod513");
    let stdout = compile_and_run(
        &build_dir,
        "divmod513",
        &matrix_driver(2, 4, &RESHAPE2_BASE),
    );
    let c_grad = parse_lines(&stdout);
    let (_, eval_g) = eval_grad(&reshape_arith2_source(
        DIV_MOD_TARGET,
        &matrix_literal(&RESHAPE2_BASE, 4),
        true,
    ));
    assert_close(
        "div/mod reshape-arith grad C vs eval",
        &c_grad,
        &eval_g,
        1e-3,
    );
    let want: Vec<f64> = RESHAPE2_BASE.iter().map(|x| 2.0 * x).collect();
    assert_close(
        "div/mod reshape-arith grad C vs analytic",
        &c_grad,
        &want,
        1e-3,
    );
}

/// eval-vs-C agreement for the arithmetic reshape-target grad.
#[test]
fn issue_513_reshape_arith_target_grad_c_backend_agrees() {
    let source =
        reshape_arith_source(true, "", false).replace("out = f(to_tensor([]))", "out = grad(f)");
    let (_dir, build_dir) = build_c(&source, "rsharith513");
    let stdout = compile_and_run(
        &build_dir,
        "rsharith513",
        &matrix_driver(2, 2, &RESHAPE_BASE),
    );
    let c_grad = parse_lines(&stdout);
    let (_, eval_g) = eval_grad(&reshape_arith_source(
        true,
        &matrix_literal(&RESHAPE_BASE, 2),
        true,
    ));
    assert_close("reshape-arith grad C vs eval", &c_grad, &eval_g, 1e-3);
    assert_close(
        "reshape-arith grad C vs analytic",
        &c_grad,
        &[2.0, 4.0, 6.0, 8.0],
        1e-3,
    );
}

// ---------------------------------------------------------------------------
// NEGATIVE PARITY: paths that genuinely need a runtime shape() VALUE stay
// fail-closed with a loud, specific diagnostic. Never silently wrong.
// ---------------------------------------------------------------------------

fn expect_grad_failure(source: &str, stem: &str, needle: &str, context: &str) {
    let output = run_eval(source, stem);
    assert!(
        !output.status.success(),
        "{context}: expected fail-closed grad, but eval SUCCEEDED; if the \
         symbolic-dim machinery landed, promote this pin to an FD oracle. \
         stdout={}",
        String::from_utf8_lossy(&output.stdout),
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains(needle),
        "{context}: failure must stay LOUD and specific (needle `{needle}`); \
         stderr={stderr}",
    );
}

/// chelis#616: a stride along the SYMBOLIC axis itself now builds the
/// runtime adjoint cascade (Shape-read trim + runtime merge extent). For
/// `f(x) = sum(stride(x, 2i64))` over `[1, 2, 3, 4]`, the loss reads elements
/// 0 and 2, so the gradient is the upsample mask `[1, 0, 1, 0]`.
#[test]
fn issue_513_stride_on_symbolic_axis_grad_is_upsample_mask() {
    let source = "module Repro.StrideSymAxis\n\
sig f: tensor[n, f32] -> f32\n\
def f(x) = {\n\
  s = stride(&x, cast(2, int64))\n\
  sum(s, cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)]))\n";
    let (shape, grad) = eval_grad(source);
    assert_eq!(shape, vec![4]);
    assert_close(
        "symbolic strided-axis grad",
        &grad,
        &[1.0, 0.0, 1.0, 0.0],
        1e-3,
    );
}

/// prod_reduce over the SYMBOLIC axis needs one slice per element of the
/// runtime axis; the construction is inherently size-dependent and stays
/// fail-closed with the existing loud message.
#[test]
fn issue_513_prod_reduce_on_symbolic_axis_stays_fail_closed() {
    let source = "module Repro.ProdSymAxis\n\
sig f: tensor[n, 3, f32] -> f32\n\
def f(x) = {\n\
  p = prod_reduce(&x, cast(0, int32))\n\
  sum(p, cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)], [cast(4.0, f32), cast(5.0, f32), cast(6.0, f32)]]))\n";
    expect_grad_failure(
        source,
        "prodsymaxis",
        "prod_reduce adjoint requires a concrete axis size",
        "prod_reduce over symbolic reduced axis",
    );
}

/// chelis#616: a shrink with CONCRETE sub-range bounds on a symbolic axis
/// now builds the runtime Pad adjoint (`after = shape(x, axis) - end`). For
/// the `[[0,1], [1,3]]` sub-range over a `[2, 4]` input, the loss reads
/// `x[0][1..3]`, so the gradient is 1 exactly there.
#[test]
fn issue_513_shrink_concrete_bounds_on_symbolic_axis_grad_is_window_mask() {
    let source = "module Repro.ShrinkSymAxis\n\
sig f: tensor[batch, 4, f32] -> f32\n\
def f(x) = {\n\
  s = shrink(&x, [[cast(0, int64), cast(1, int64)], [cast(1, int64), cast(3, int64)]])\n\
  sum(sum(s, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)], [cast(5.0, f32), cast(6.0, f32), cast(7.0, f32), cast(8.0, f32)]]))\n";
    let (shape, grad) = eval_grad(source);
    assert_eq!(shape, vec![2, 4]);
    assert_close(
        "shrink sub-range on symbolic axis grad",
        &grad,
        &[0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        1e-3,
    );
}

/// The SYMBOLIC-sig arithmetic reshape target (the true school im2col form,
/// `tensor[a, b, f32]` with `reshape(p, [mul(b_d, a_d), 1i64])`) now LOWERS
/// (chelis#616): the runtime product becomes a rank-0 scalar node that the
/// reshape references as a node-valued target extent, the numel invariant is
/// enforced at run time, and the backward pass resolves the runtime extent
/// mid-evaluation. The loss is a plain double-sum, so the gradient is ones
/// everywhere; the eval and C lanes must agree exactly. (This replaces the
/// pre-#616 fail-closed pin on the lowering-time exactness-gate refusal.)
#[test]
fn issue_513_symbolic_sig_reshape_arith_target_grad_is_ones() {
    let source = "module Repro.ReshapeArithSym\n\
sig f: tensor[a, b, f32] -> f32\n\
def f(x) = {\n\
  a_d = cast(shape(x, cast(0, int32)), int64)\n\
  b_d = cast(shape(x, cast(1, int32)), int64)\n\
  p = permute(&x, cast(1, int32), cast(0, int32))\n\
  r = reshape(p, [mul(b_d, a_d), cast(1, int64)])\n\
  sum(sum(r, cast(0, int32)), cast(0, int32)) |> tensor_to_scalar\n\
}\n\
out = grad(f)(to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]]))\n";
    let (shape, eval_g) = eval_grad(source);
    assert_eq!(shape, vec![2, 2], "gradient keeps the input shape");
    assert_close("symbolic-sig reshape-arith grad", &eval_g, &[1.0; 4], 1e-3);

    // eval-vs-C agreement: the emitted main computes the same gradient by
    // reading the runtime extents and declaring the reshape target dim from
    // its scalar.
    let (_dir, build_dir) = build_c(source, "rshasym");
    let stdout = compile_and_run_main(&build_dir, "rshasym");
    let c_grad = parse_printed_tensor(&stdout);
    assert_close(
        "symbolic-sig reshape-arith grad C vs eval",
        &c_grad,
        &eval_g,
        1e-3,
    );
}

/// A FORMERLY gate-refused arithmetic target over a CONCRETE sig: the fold
/// refuses the negative intermediate (`sub(a_d, 10)` = -8 feeding
/// `floor_div`, the domain where floor, trunc, and euclidean division
/// disagree), so chelis#616 lowers the expression to a runtime scalar
/// instead of guessing. Its true runtime value makes the written target
/// [4, 4] — 16 elements over an 8-element input — so BOTH lanes must reject
/// it at run time with the numel invariant: the eval lane with a clean
/// error, the C lane with the emitted numel abort. (Pre-hardening this was
/// the silent-acceptance hole: a wildcard anon dim bound to a coincidental
/// input extent and `grad` returned plausible numbers for an ill-formed
/// program.)
#[test]
fn issue_513_reshape_arith_gate_refused_grad_numel_mismatch_errs_in_both_lanes() {
    let target = "neg(floor_div(sub(a_d, cast(10, int64)), cast(2, int64))), cast(4, int64)";
    let source = reshape_arith2_source(
        target,
        "[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)], \
         [cast(5.0, f32), cast(6.0, f32), cast(7.0, f32), cast(8.0, f32)]",
        true,
    );
    expect_grad_failure(
        &source,
        "gaterefused",
        "elements but the input has",
        "runtime numel mismatch under grad",
    );

    // C-lane error parity: the build succeeds (the mismatch is a runtime
    // property), and the emitted binary aborts at the reshape numel guard
    // instead of allocating a mis-sized view.
    let (_dir, build_dir) = build_c(&source, "gaterefused");
    let kernel = build_dir.join("gaterefused.c");
    let runtime = build_dir.join("libchelis_runtime.a");
    let bin = build_dir.join("self_bin");
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
    let run = StdCommand::new(&bin).output().expect("run emitted program");
    assert!(
        !run.status.success(),
        "C binary must abort on the runtime numel mismatch; stdout={}",
        String::from_utf8_lossy(&run.stdout)
    );
    let stderr = String::from_utf8_lossy(&run.stderr);
    assert!(
        stderr.contains("reshape numel mismatch"),
        "C abort must name the reshape numel guard; stderr={stderr}"
    );
}

/// A target the fold PROVES negative (`sub(a_d, 10)` = -8 directly as the
/// extent) is a proven-invalid program and must fail loud with the
/// negative-extent diagnostic, never fall back to wildcard dims.
#[test]
fn issue_513_reshape_arith_negative_fold_fails_loud() {
    let target = "sub(a_d, cast(10, int64)), cast(4, int64)";
    let source = reshape_arith2_source(
        target,
        "[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32), cast(4.0, f32)], \
         [cast(5.0, f32), cast(6.0, f32), cast(7.0, f32), cast(8.0, f32)]",
        true,
    );
    expect_grad_failure(
        &source,
        "negfold",
        "folds to the negative extent",
        "negative-extent arithmetic reshape target under grad",
    );
}

/// HOST-LANE PARITY GUARD for the loud refusal: a FORWARD (non-grad) use of
/// a gate-refused-but-VALID arithmetic target must keep evaluating through
/// the host lane, which computes the written expression with true runtime
/// semantics (`neg(floor_div(sub(2, 5), 2))` = 2 under floor division, so
/// the reshape is [2, 4] over the 8-element input and the loss is
/// sum(x^2) = 204). The lowering refusal must never leak into forward
/// evaluation of forms the host lane handles honestly.
#[test]
fn issue_513_reshape_arith_gate_refused_forward_still_evals_via_host() {
    let target = "neg(floor_div(sub(a_d, cast(5, int64)), cast(2, int64))), cast(4, int64)";
    let source = reshape_arith2_source(target, &matrix_literal(&RESHAPE2_BASE, 4), false);
    let loss = eval_scalar(&source);
    let want: f64 = RESHAPE2_BASE.iter().map(|x| x * x).sum();
    assert!(
        (loss - want).abs() < 1e-3,
        "forward host eval of the gate-refused-but-valid form must produce \
         sum(x^2) = {want}; got {loss}"
    );
}
