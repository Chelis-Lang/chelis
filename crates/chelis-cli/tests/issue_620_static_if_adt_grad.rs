//! Issue #620: `grad` through static control flow over ADT/list values,
//! bounded recursion unrolling, and linearity copies over params ADTs.
//!
//! The chelis#520 D1/D2 slices left three gaps that blocked every real
//! model loss downstream (the School shell's re-probe at 0.16.0):
//!
//!   1. an `if` branch yielding an ADT or list value died in lowering
//!      ("if then branch expected a single tensor value, got an ADT
//!      value"), even when the condition was compile-time-resolvable --
//!      the shape of every eps guard (`if eps <= 0 then fail(...) else
//!      body`) and of every recursive window/patch collector's base case;
//!   2. a recursive builder could not unroll at all: the Inlining-F1
//!      guard refused re-entry and fell through to a silently wrong
//!      fallback;
//!   3. a compiler-inserted or explicit `copy` over a params ADT died in
//!      "copy input expected a single tensor value" (the issue's Blocker
//!      2, hit by the curried closure `grad(fn (p) -> loss(p, x, y))`).
//!
//! The fix: `lower_if` prunes to the taken branch when the condition
//! const-folds at lowering time (the `if` analogue of D1's static arm
//! selection; exact gradient, since a static condition cannot vary under
//! input perturbation); recursion unrolls under per-name/total depth caps
//! with a loud diagnostic past the cap; `copy`/`drop` recurse through
//! tuple/ADT structure; and `concat` handles list values that are only
//! statically known post-unroll.
//!
//! Negative parity (each pinned with its diagnostic):
//!   - runtime-condition `if` with ADT branches stays rejected, citing
//!     the select/blend successor (chelis#618)
//!   - unbounded recursion errors loudly at the unroll cap
//!   - runtime-condition non-float `if` stays unrepresentable

use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;

use assert_cmd::Command;
use tempfile::{TempDir, tempdir};

/// Run `chelis eval` on a full program source; return (stdout, stderr, ok).
fn eval_program(source: &str) -> (String, String, bool) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("prog.ch");
    fs::write(&path, source).expect("write");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("run");
    (
        String::from_utf8_lossy(&output.stdout).into_owned(),
        String::from_utf8_lossy(&output.stderr).into_owned(),
        output.status.success(),
    )
}

/// Parse the first `data=[...]` tensor payload out of eval stdout.
fn parse_tensor_data(stdout: &str) -> Vec<f64> {
    let line = stdout
        .lines()
        .find(|l| l.contains("data=["))
        .unwrap_or_else(|| panic!("no tensor in eval output: {stdout}"));
    line.split_once("data=[")
        .and_then(|(_, r)| r.split_once(']'))
        .map(|(s, _)| {
            s.split(',')
                .map(|t| t.trim().parse::<f64>().unwrap())
                .collect()
        })
        .unwrap()
}

/// Parse the last printed scalar from eval stdout (forward-loss runs).
fn parse_scalar(stdout: &str) -> f64 {
    stdout
        .trim()
        .lines()
        .last()
        .and_then(|l| l.trim().parse::<f64>().ok())
        .unwrap_or_else(|| panic!("no scalar in eval output: {stdout}"))
}

fn fmt_f32_list(xs: &[f64]) -> String {
    xs.iter()
        .map(|x| format!("cast({x:?}, f32)"))
        .collect::<Vec<_>>()
        .join(", ")
}

// --- Static-condition if: ADT branches prune ---------------------------------

/// The inverse of `issue_520_d1_runtime_ctor_through_if_still_rejected`:
/// the SAME `pick` function, but called with a literal argument, so the
/// condition const-folds at lowering time, the constructor branch prunes
/// statically, and the downstream match differentiates the taken arm.
#[test]
fn issue_620_static_cond_if_adt_branch_prunes_under_grad() {
    let source = format!(
        "module Repro.If620\n\n\
         type Mode =\n\
         \x20 | ModeA\n\
         \x20 | ModeB\n\n\
         def pick(c: f32) -> Mode = if c > 0.0 then ModeA else ModeB\n\n\
         def fwd_staticpick(x: tensor[2, f32]) -> f32 = {{\n\
         \x20 match pick(cast(1.0, f32)) with {{\n\
         \x20   | ModeA => sum(x, cast(0, int32)) |> tensor_to_scalar\n\
         \x20   | ModeB => cast(0.0, f32)\n\
         \x20 }}\n\
         }}\n\n\
         out = grad(fwd_staticpick)(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "static-cond ADT-branch if under grad failed: {stderr}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad, vec![1.0, 1.0], "grad of the taken sum arm: {stdout}");
}

/// The else side of the same shape: a literal condition folding FALSE
/// selects the ModeB arm, whose gradient is the shaped zero.
#[test]
fn issue_620_static_cond_if_false_selects_else_ctor() {
    let source = format!(
        "module Repro.If620Else\n\n\
         type Mode =\n\
         \x20 | ModeA\n\
         \x20 | ModeB\n\n\
         def pick(c: f32) -> Mode = if c > 0.0 then ModeA else ModeB\n\n\
         def fwd_staticpick(x: tensor[2, f32]) -> f32 = {{\n\
         \x20 match pick(cast(-1.0, f32)) with {{\n\
         \x20   | ModeA => cast(0.0, f32)\n\
         \x20   | ModeB => sum(mul(&x, &x), cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         }}\n\n\
         out = grad(fwd_staticpick)(to_tensor([{}]))\n",
        fmt_f32_list(&[1.5, -0.5]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "false-cond ADT-branch if under grad failed: {stderr}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(
        grad,
        vec![3.0, -1.0],
        "grad of 2x for the else arm: {stdout}"
    );
}

// --- The eps fail-guard idiom -------------------------------------------------

/// `if eps <= 0 then fail(...) else { tensor body }` with a literal eps:
/// the guard prunes, the fail branch is never lowered, and the body
/// differentiates. This is the School layernorm/pool guard shape verbatim.
#[test]
fn issue_620_eps_fail_guard_body_differentiates() {
    let x = [1.5, -0.5];
    let source = format!(
        "module Repro.Guard620\n\n\
         def loss_guard(x: tensor[2, f32], eps: f32) -> f32 = {{\n\
         \x20 if lte(eps, cast(0.0, f32)) then fail(\"eps must be positive\") else {{\n\
         \x20   sum(mul(&x, &x), cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         }}\n\n\
         out = grad(loss_guard, wrt=(x))(to_tensor([{}]), cast(0.001, f32))\n",
        fmt_f32_list(&x),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "eps fail-guard grad failed: {stderr}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad.len(), 2, "grad shape: {stdout}");
    for (i, g) in grad.iter().enumerate() {
        let want = 2.0 * x[i];
        assert!(
            (g - want).abs() < 1e-5,
            "grad elem {i}: got {g}, want {want} (2x): {stdout}"
        );
    }
}

/// Forward parity for the guard: the pruned forward evaluates to the same
/// loss the host evaluator computes.
#[test]
fn issue_620_eps_fail_guard_forward_parity() {
    let x = [1.5, -0.5];
    let source = format!(
        "module Repro.Guard620F\n\n\
         def loss_guard(x: tensor[2, f32], eps: f32) -> f32 = {{\n\
         \x20 if lte(eps, cast(0.0, f32)) then fail(\"eps must be positive\") else {{\n\
         \x20   sum(mul(&x, &x), cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         }}\n\n\
         out = loss_guard(to_tensor([{}]), cast(0.001, f32))\n",
        fmt_f32_list(&x),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "guard forward failed: {stderr}");
    let loss = parse_scalar(&stdout);
    let want = x.iter().map(|v| v * v).sum::<f64>();
    assert!(
        (loss - want).abs() < 1e-5,
        "forward loss: got {loss}, want {want}"
    );
}

// --- Recursive list builder (the im2col/pool collector shape) ------------------

fn collector_program(x: &[f64], applied: &str) -> String {
    format!(
        "module Repro.Collect620\n\n\
         def collect(x: &tensor[2, f32], k: int64, n: int64) -> List[tensor[2, f32]] = {{\n\
         \x20 if gte(k, n) then [] else {{\n\
         \x20   rest = collect(x, add(k, cast(1, int64)), n)\n\
         \x20   concat([mul(x, x)], rest)\n\
         \x20 }}\n\
         }}\n\n\
         def loss_windows(x: tensor[2, f32]) -> f32 = {{\n\
         \x20 stacked = concat(collect(&x, cast(0, int64), cast(3, int64)), cast(0, int32))\n\
         \x20 sum(stacked, cast(0, int32)) |> tensor_to_scalar\n\
         }}\n\n\
         out = {applied}(to_tensor([{}]))\n",
        fmt_f32_list(x),
        applied = applied,
    )
}

/// A recursive `if k >= n then [] else concat([row], recurse)` builder
/// with literal-rooted bounds unrolls statically under grad, the appended
/// list value concats into a tensor, and the gradient matches both the
/// analytic value (3 windows of x*x sum to grad 6x) and a
/// central-difference oracle on the forward loss.
#[test]
fn issue_620_recursive_list_builder_grad_matches_fd() {
    let base = [1.5, -0.5];
    let (stdout, stderr, ok) = eval_program(&collector_program(&base, "grad(loss_windows)"));
    assert!(ok, "recursive builder grad failed: {stderr}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad.len(), 2, "grad shape: {stdout}");
    for (i, g) in grad.iter().enumerate() {
        let want = 6.0 * base[i];
        assert!(
            (g - want).abs() < 1e-4,
            "grad elem {i}: got {g}, want {want} (6x): {stdout}"
        );
    }
    // Central-difference oracle on the forward loss.
    let h = 1e-3;
    for i in 0..base.len() {
        let mut plus = base;
        plus[i] += h;
        let mut minus = base;
        minus[i] -= h;
        let (s_plus, e_plus, ok_plus) = eval_program(&collector_program(&plus, "loss_windows"));
        assert!(ok_plus, "forward(+h) failed: {e_plus}");
        let (s_minus, e_minus, ok_minus) = eval_program(&collector_program(&minus, "loss_windows"));
        assert!(ok_minus, "forward(-h) failed: {e_minus}");
        let fd = (parse_scalar(&s_plus) - parse_scalar(&s_minus)) / (2.0 * h);
        assert!(
            (grad[i] - fd).abs() < 1e-2,
            "elem {i}: grad {} != central difference {fd}",
            grad[i]
        );
    }
}

// --- Multi-argument params loss (the issue's headline ask) ---------------------

/// `grad(loss, wrt=(p))(params, x, y, eps)`: a real loss shape -- params
/// ADT destructured by match, an eps fail-guard, data tensors alongside --
/// returns the Params-shaped cotangent. d/dw sum((x*w - y)^2) = 2(xw-y)x.
#[test]
fn issue_620_multiarg_params_loss_grad() {
    let x = [2.0, 3.0];
    let w = [0.5, -1.0];
    let y = [1.0, 1.0];
    let source = format!(
        "module Repro.Loss620\n\n\
         type Params =\n\
         \x20 | Params {{ w: tensor[2, f32] }}\n\n\
         def loss(p: Params, x: tensor[2, f32], y: tensor[2, f32], eps: f32) -> f32 = {{\n\
         \x20 match p with {{\n\
         \x20   | Params {{ w: w }} => if lte(eps, cast(0.0, f32)) then fail(\"eps\") else {{\n\
         \x20     d = sub(mul(&x, &w), y)\n\
         \x20     sum(mul(&d, &d), cast(0, int32)) |> tensor_to_scalar\n\
         \x20   }}\n\
         \x20 }}\n\
         }}\n\n\
         out = grad(loss, wrt=(p))(Params {{ w: to_tensor([{w}]) }}, to_tensor([{x}]), to_tensor([{y}]), cast(0.00001, f32))\n",
        w = fmt_f32_list(&w),
        x = fmt_f32_list(&x),
        y = fmt_f32_list(&y),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "multi-arg params loss grad failed: {stderr}");
    assert!(
        stdout.contains("Params("),
        "gradient must be Params-shaped: {stdout}"
    );
    let grad = parse_tensor_data(&stdout);
    for i in 0..2 {
        let want = 2.0 * (x[i] * w[i] - y[i]) * x[i];
        assert!(
            (grad[i] - want).abs() < 1e-4,
            "grad w elem {i}: got {}, want {want}: {stdout}",
            grad[i]
        );
    }
}

/// Blocker 2 verbatim: the curried single-argument closure over a params
/// ADT (`grad(fn (p) -> loss(p, x, y))(params)`) lowers and returns the
/// Params-shaped cotangent (previously: "copy input expected a single
/// tensor value, got an ADT value").
#[test]
fn issue_620_curried_closure_over_params_adt() {
    let x = [2.0, 3.0];
    let w = [0.5, -1.0];
    let y = [1.0, 1.0];
    let source = format!(
        "module Repro.Curry620\n\n\
         type Params =\n\
         \x20 | Params {{ w: tensor[2, f32] }}\n\n\
         def loss(p: Params, x: tensor[2, f32], y: tensor[2, f32]) -> f32 = {{\n\
         \x20 match p with {{\n\
         \x20   | Params {{ w: w }} => {{\n\
         \x20     d = sub(mul(&x, &w), y)\n\
         \x20     sum(mul(&d, &d), cast(0, int32)) |> tensor_to_scalar\n\
         \x20   }}\n\
         \x20 }}\n\
         }}\n\n\
         x = to_tensor([{x}])\n\
         y = to_tensor([{y}])\n\
         out = grad(fn (p: Params) -> loss(p, x, y))(Params {{ w: to_tensor([{w}]) }})\n",
        w = fmt_f32_list(&w),
        x = fmt_f32_list(&x),
        y = fmt_f32_list(&y),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "curried closure over params ADT failed: {stderr}");
    // The top-level `x`/`y` bindings display first; parse the gradient
    // from the Params-shaped `out` line specifically.
    let out_line = stdout
        .lines()
        .find(|l| l.contains("Params("))
        .unwrap_or_else(|| panic!("no Params-shaped gradient line: {stdout}"));
    let grad = parse_tensor_data(out_line);
    for i in 0..2 {
        let want = 2.0 * (x[i] * w[i] - y[i]) * x[i];
        assert!(
            (grad[i] - want).abs() < 1e-4,
            "grad w elem {i}: got {}, want {want}: {stdout}",
            grad[i]
        );
    }
}

/// Source-surface boundary pin: the checked surface cannot express a
/// double-read of an owned ADT today. A second `match p` is a linearity
/// error (match consumes the scrutinee), source-level `copy(p)` is a
/// checker type error ("copy requires tensor input"), and `match &p` is
/// outside the borrow surface ("borrow is only valid as a direct call
/// argument"). Compiler-INSERTED Copy/Drop over ADT values (the paths
/// chelis#620 made lowerable) are therefore pinned at the unit level
/// (`copy_adt_lowers_field_wise` / `drop_adt_closes_each_leaf` in
/// chelis-ir's lower.rs) and end-to-end by the curried-closure test
/// above; this pin documents why no checked-source e2e exists.
#[test]
fn issue_620_owned_adt_double_read_stays_a_linearity_error() {
    let w = [1.5, -0.5];
    let source = format!(
        "module Repro.Copy620\n\n\
         type Params =\n\
         \x20 | Params {{ w: tensor[2, f32] }}\n\n\
         def fwd_two_reads(p: Params) -> f32 = {{\n\
         \x20 a = match p with {{\n\
         \x20   | Params {{ w: w }} => sum(w, cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         \x20 b = match p with {{\n\
         \x20   | Params {{ w: w }} => sum(mul(&w, &w), cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         \x20 add(a, b)\n\
         }}\n\n\
         out = grad(fwd_two_reads)(Params {{ w: to_tensor([{}]) }})\n",
        fmt_f32_list(&w),
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(!ok, "owned-ADT double read must stay a linearity error");
    assert!(
        stderr.contains("already consumed"),
        "diagnostic must name the consumed scrutinee: {stderr}"
    );
}

/// A grad over a struct argument named `params` -- the conventional
/// pytree name, which collides with the reserved Deep `params` tag and
/// therefore desugars through chelis-surf's MetaExpr param wrapper --
/// binds and differentiates like any other name. Regression pin: the
/// wrapper form was silently dropped from the lowering's param-name walk,
/// so `params` never bound, its body references lowered to bogus Loads,
/// and the match failed as a "runtime scrutinee" (reproduced on 0.16.0).
/// Same guard applies to any reserved-tag name (`match`, `if`, ...).
#[test]
fn issue_620_param_named_params_binds_and_differentiates() {
    let g = [1.0, 1.0];
    let b = [0.0, 0.1];
    let x = [1.0, 2.0];
    let source = format!(
        "module Repro.Named620\n\n\
         type P2 =\n\
         \x20 | P2 {{ g: tensor[2, f32], b: tensor[2, f32] }}\n\n\
         def loss2f(params: P2, x: tensor[2, f32], eps: f32) -> f32 = {{\n\
         \x20 match params with {{\n\
         \x20   | P2 {{ g: g, b: b }} => if lte(eps, cast(0.0, f32)) then fail(\"eps\") else {{\n\
         \x20     scaled = add(mul(&x, &g), b)\n\
         \x20     sum(mul(&scaled, &scaled), cast(0, int32)) |> tensor_to_scalar\n\
         \x20   }}\n\
         \x20 }}\n\
         }}\n\n\
         out = grad(loss2f, wrt=(params))(P2 {{ g: to_tensor([{g}]), b: to_tensor([{b}]) }}, to_tensor([{x}]), cast(0.00001, f32))\n",
        g = fmt_f32_list(&g),
        b = fmt_f32_list(&b),
        x = fmt_f32_list(&x),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "grad over a param named `params` failed: {stderr}");
    assert!(
        stdout.contains("P2("),
        "gradient must be P2-shaped: {stdout}"
    );
    let grad = parse_tensor_data(&stdout);
    for i in 0..2 {
        let want = 2.0 * (x[i] * g[i] + b[i]) * x[i];
        assert!(
            (grad[i] - want).abs() < 1e-5,
            "dgamma elem {i}: got {}, want {want}: {stdout}",
            grad[i]
        );
    }
}

// --- C-backend agreement --------------------------------------------------------

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

fn c_agree_body() -> String {
    "type Mode =\n\
     \x20 | ModeA\n\
     \x20 | ModeB\n\n\
     def pick(c: f32) -> Mode = if c > 0.0 then ModeA else ModeB\n\n\
     def fwd_cguard(x: tensor[2, f32]) -> f32 = {\n\
     \x20 match pick(cast(1.0, f32)) with {\n\
     \x20   | ModeA => if lte(cast(0.001, f32), cast(0.0, f32)) then fail(\"eps\") else {\n\
     \x20     sum(mul(&x, &x), cast(0, int32)) |> tensor_to_scalar\n\
     \x20   }\n\
     \x20   | ModeB => cast(0.0, f32)\n\
     \x20 }\n\
     }\n"
    .to_string()
}

/// Compiled lane: `chelis build --target c` of a grad whose body prunes a
/// static-condition constructor pick AND an eps fail-guard must compile,
/// run, and agree with the eval lane and the analytic gradient 2x.
#[test]
fn issue_620_static_if_grad_matches_c_backend() {
    let source = format!(
        "module Repro.C620\n\n{}\nout = grad(fwd_cguard)\n",
        c_agree_body()
    );
    let (_dir, build_dir) = build_c(&source, "if620");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* out(chelis_tensor* arg0);
int main(void) {
    int shape[1] = {2};
    chelis_tensor* x = chelis_alloc(1, shape, CHELIS_F32);
    float xd[2] = {1.0f, 2.0f};
    memcpy(x->data, xd, sizeof(xd));
    chelis_tensor* g = out(x);
    if (g->size != 2) { printf("FAIL_SIZE %d\n", g->size); return 1; }
    for (int i = 0; i < 2; i++) printf("%.6f\n", g->data[i]);
    return 0;
}
"#;
    let stdout = compile_and_run(&build_dir, "if620", driver);
    let c_grad: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect();
    assert_eq!(c_grad.len(), 2, "C grad length: {stdout}");
    let eval_source = format!(
        "module Repro.C620E\n\n{}\nout = grad(fwd_cguard)(to_tensor([{}]))\n",
        c_agree_body(),
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (eval_stdout, eval_stderr, ok) = eval_program(&eval_source);
    assert!(ok, "eval lane failed: {eval_stderr}");
    let eval_grad = parse_tensor_data(&eval_stdout);
    for (i, (c, e)) in c_grad.iter().zip(eval_grad.iter()).enumerate() {
        assert!((c - e).abs() < 1e-5, "elem {i}: C backend {c} != eval {e}");
        let want = 2.0 * (i as f64 + 1.0);
        assert!(
            (c - want).abs() < 1e-5,
            "elem {i}: C backend {c} != analytic {want}"
        );
    }
}

// --- Negative parity ------------------------------------------------------------

/// A RUNTIME condition with ADT-valued branches stays rejected; the
/// diagnostic keeps the pinned prefix and cites the select/blend successor
/// (chelis#618). Twin of the #520 pin, asserting the new citation.
#[test]
fn issue_620_runtime_cond_adt_branch_rejected_cites_618() {
    let source = format!(
        "module Repro.Neg620A\n\n\
         type Mode =\n\
         \x20 | ModeA\n\
         \x20 | ModeB\n\n\
         def pick(c: f32) -> Mode = if c > 0.0 then ModeA else ModeB\n\n\
         def fwd_dynpick(x: tensor[2, f32]) -> f32 = {{\n\
         \x20 match pick(tensor_to_scalar(sum(&x, cast(0, int32)))) with {{\n\
         \x20   | ModeA => sum(x, cast(0, int32)) |> tensor_to_scalar\n\
         \x20   | ModeB => cast(0.0, f32)\n\
         \x20 }}\n\
         }}\n\n\
         out = grad(fwd_dynpick)(to_tensor([{}]))\n",
        fmt_f32_list(&[-1.0, -2.0]),
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(!ok, "runtime-cond ADT branch must stay rejected");
    assert!(
        stderr.contains("expected a single tensor value, got an ADT value"),
        "pinned prefix must survive: {stderr}"
    );
    assert!(
        stderr.contains("chelis#618"),
        "diagnostic must cite the select/blend successor: {stderr}"
    );
}

/// Unbounded recursion (a base case the static fold can never satisfy)
/// errors loudly at the unroll cap, naming the callee and the limit --
/// never a hang, never the pre-#620 silent argument collapse.
#[test]
fn issue_620_unbounded_recursion_errors_at_unroll_cap() {
    let source = format!(
        "module Repro.Neg620B\n\n\
         def spin(x: &tensor[2, f32], k: int64) -> f32 = {{\n\
         \x20 if gte(k, cast(0, int64)) then spin(x, add(k, cast(1, int64))) else {{\n\
         \x20   sum(x, cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         }}\n\n\
         def loss_spin(x: tensor[2, f32]) -> f32 = spin(&x, cast(0, int64))\n\n\
         out = grad(loss_spin)(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(!ok, "unbounded recursion must be rejected, not hang");
    assert!(
        stderr.contains("static unroll limit"),
        "diagnostic must name the unroll limit: {stderr}"
    );
    assert!(
        stderr.contains("spin"),
        "diagnostic must name the recursive callee: {stderr}"
    );
}

/// A runtime-condition `if` whose value is non-float stays on the
/// unrepresentable path (only static conditions changed lanes).
#[test]
fn issue_620_runtime_cond_nonfloat_if_still_unrepresentable() {
    let source = format!(
        "module Repro.Neg620C\n\n\
         def pick_i(c: f32) -> int32 = if c > 0.0 then cast(1, int32) else cast(2, int32)\n\n\
         def fwd(x: tensor[2, f32]) -> f32 = {{\n\
         \x20 scale = cast(pick_i(tensor_to_scalar(sum(&x, cast(0, int32)))), f32)\n\
         \x20 mul(sum(x, cast(0, int32)) |> tensor_to_scalar, scale)\n\
         }}\n\n\
         out = grad(fwd)(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(!ok, "runtime-cond non-float if must stay rejected");
    assert!(
        stderr.contains("not supported by IR evaluation"),
        "diagnostic must stay on the unrepresentable path: {stderr}"
    );
}
