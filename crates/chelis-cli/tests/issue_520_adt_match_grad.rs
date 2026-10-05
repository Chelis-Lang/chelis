//! Issue #520 (D1/D2 slices): `grad` through a `match` body and `grad` over
//! an ADT/record argument.
//!
//! D1 slice (static arm selection): a `match` whose scrutinee is a
//! compile-time-known constructor value resolves to the taken arm at
//! lowering time, so `grad` differentiates the selected arm's body. The
//! scrutinee tag is discrete: perturbing tensor inputs cannot change the
//! taken arm, so differentiating only that arm is the exact gradient.
//! Covered in BOTH the host-eval lane (`chelis eval`) and the compiled lane
//! (`chelis build --target c`).
//!
//! D2 slice (field-wise ADT gradient, eval lane): `grad(f)(Ctor { .. })`
//! over an ADT argument whose fields are all float tensors returns a
//! structurally matching gradient value (`Ctor(grad_t, ...)`), the pytree
//! contract of spec/design/differentiable_language.md Decision 6 / Phase 2.
//! The ADT argument may appear ALONGSIDE plain tensor arguments (the
//! chelis#520 closing bar `grad(model_forward, wrt=params)(x, params)`):
//! the result is a tuple whose ADT slot is the field-wise gradient struct
//! and whose tensor slots are bare tensor gradients, or the bare struct
//! when `wrt` narrows to the ADT alone. The compiled lane keeps rejecting
//! an ADT-param grad export (the C ABI has no ADT value representation);
//! that rejection is pinned below.
//!
//! Also covers chelis#614: a multi-target grad binding `out = grad(f)(a,
//! b)` (tuple payload) displays every slot in the eval lane instead of
//! silently dropping them behind a "defs-only" breadcrumb. An unused
//! (zero-adjoint) argument in a multi-target result is displayed as its
//! shaped zero, keeping every slot in its correct `out.0..out.N` position
//! rather than dropping it -- a dropped slot would shift and mislabel every
//! later gradient (pure-tensor and ADT-mixed cases pinned below).
//!
//! Negative parity (each pinned with its diagnostic):
//!   - runtime (non-constructor) scrutinee in a differentiated `match`
//!   - runtime-dependent constructor (an `if` between constructors) feeding
//!     a differentiated `match` scrutinee
//!   - guarded arm reached during static arm selection
//!   - explicitly targeting a pure enum (no float leaf in any variant)
//!   - compiled-lane `out = grad(f)` export over an ADT-typed param
//!
//! Recursive cotangent parity also pins mixed tensor/non-tensor ADT fields
//! and mixed sibling variants as legal: differentiable fields receive their
//! cotangents while discrete fields remain present as unit.
//!
//! Zero-fill parity (unused multi-target argument -> shaped zero slot, the
//! negative-of-the-bug pinned by output assertions rather than a
//! diagnostic, each red on the pre-zero-fill behavior):
//!   - ADT arg alongside an unused tensor arg (zero tensor slot)
//!   - pure-tensor grad with an unused MIDDLE arg (zero slot, no mislabel)
//!   - pure-tensor grad with an unused LEADING arg (zero slot, no vanish)

mod common;

use common::authored_c_symbol;
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
        .and_then(|l| {
            let trimmed = l.trim();
            let value_str = trimmed.split(" = ").last().unwrap_or(trimmed);
            value_str.parse::<f64>().ok()
        })
        .unwrap_or_else(|| panic!("no scalar in eval output: {stdout}"))
}

fn fmt_f32_list(xs: &[f64]) -> String {
    xs.iter()
        .map(|x| format!("cast({x:?}, f32)"))
        .collect::<Vec<_>>()
        .join(", ")
}

// --- D1: grad through a static-scrutinee match --------------------------------

const D1_HEADER: &str = "module Repro.D1\n\n\
type Mode =\n\
  | ModeA\n\
  | ModeB\n\n";

/// The issue's D1 reproducer body: match on a statically-known nullary
/// constructor; the taken arm is linear in `x`, the dead arm is constant.
fn d1_match_fn(body_a: &str, body_b: &str) -> String {
    format!(
        "def fwd_match(x: tensor[2, f32]) -> f32 = match ModeA with {{\n\
         \x20   | ModeA => {body_a}\n\
         \x20   | ModeB => {body_b}\n\
         \x20 }}\n\
"
    )
}

/// D1 acceptance oracle (issue #520 facet 2): grad through the match body
/// equals the analytic gradient of the TAKEN arm, [1, 1].
#[test]
fn issue_520_d1_grad_through_static_match_eval() {
    let source = format!(
        "{D1_HEADER}{}\nout = grad(fwd_match)(to_tensor([{}]))\n",
        d1_match_fn("sum(x, cast(0, i32)) |> tensor_to_scalar", "cast(0.0, f32)"),
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "grad eval failed: {stderr}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad.len(), 2, "grad shape: {stdout}");
    for (i, g) in grad.iter().enumerate() {
        assert!((g - 1.0).abs() < 1e-6, "grad elem {i}: got {g}, want 1");
    }
}

/// Control: the identical body WITHOUT the match already grads to [1, 1];
/// the D1 result above must match it exactly.
#[test]
fn issue_520_d1_control_without_match_eval() {
    let source = format!(
        "module Repro.Ctrl\n\n\
         def fwd_plain(x: tensor[2, f32]) -> f32 = sum(x, cast(0, i32)) |> tensor_to_scalar\n\n\
         out = grad(fwd_plain)(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "control grad eval failed: {stderr}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad, vec![1.0, 1.0], "control grad: {stdout}");
}

/// Nonlinear taken arm, finite-difference oracle: grad of
/// sum(mul(x, x)) through the match is 2x, FD-validated from the forward
/// loss (the forward `match` already evaluates in the host runtime).
#[test]
fn issue_520_d1_static_match_nonlinear_matches_fd() {
    let base = [1.0, 2.0];
    let nonlinear = d1_match_fn(
        "sum(mul(&x, &x), cast(0, i32)) |> tensor_to_scalar",
        "cast(0.0, f32)",
    );
    let source = format!(
        "{D1_HEADER}{nonlinear}\nout = grad(fwd_match)(to_tensor([{}]))\n",
        fmt_f32_list(&base),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "nonlinear grad eval failed: {stderr}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad.len(), 2);
    for (i, g) in grad.iter().enumerate() {
        assert!(
            (g - 2.0 * base[i]).abs() < 1e-4,
            "analytic grad elem {i}: got {g}, want {}",
            2.0 * base[i]
        );
    }
    // Central-difference oracle from the forward loss.
    let h = 1e-2;
    for i in 0..base.len() {
        let mut xp = base.to_vec();
        let mut xm = base.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let fwd = |xs: &[f64]| {
            let src = format!(
                "{D1_HEADER}{nonlinear}\nout = fwd_match(to_tensor([{}]))\n",
                fmt_f32_list(xs),
            );
            let (stdout, stderr, ok) = eval_program(&src);
            assert!(ok, "forward eval failed: {stderr}");
            parse_scalar(&stdout)
        };
        let fd = (fwd(&xp) - fwd(&xm)) / (2.0 * h);
        assert!(
            (grad[i] - fd).abs() < 5e-2,
            "elem {i}: analytic {} vs finite-difference {fd}",
            grad[i]
        );
    }
}

/// Nested static matches: both scrutinees are compile-time constructors,
/// so both resolve statically and the innermost taken arm's gradient is
/// exact (2x for the sum-of-squares arm).
#[test]
fn issue_520_d1_nested_static_match_grads_taken_arm() {
    let source = format!(
        "module Repro.D1Nested\n\n\
         type Mode =\n\
         \x20 | ModeA\n\
         \x20 | ModeB\n\n\
         type Kind =\n\
         \x20 | KindX\n\
         \x20 | KindY\n\n\
         def fwd_nested(x: tensor[2, f32]) -> f32 = match ModeA with {{\n\
         \x20   | ModeA => match KindY with {{\n\
         \x20     | KindX => cast(0.0, f32)\n\
         \x20     | KindY => sum(mul(&x, &x), cast(0, i32)) |> tensor_to_scalar\n\
         \x20   }}\n\
         \x20   | ModeB => cast(0.0, f32)\n\
         \x20 }}\n\
\n\
         out = grad(fwd_nested)(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "nested static match grad failed: {stderr}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad, vec![2.0, 4.0], "grad of inner taken arm: {stdout}");
}

/// A constructor application whose FIELD value is runtime-dependent while
/// the tag is static: the bound field must wire into the DAG so gradients
/// flow through it. grad of sum(t) where t = x*x is 2x; FD-validated.
#[test]
fn issue_520_d1_static_tag_runtime_field_gradient_flows() {
    let base = [1.0, 2.0];
    let body = "module Repro.D1Field\n\n\
                type Box =\n\
                \x20 | Box { t: tensor[2, f32] }\n\n\
                def fwd_field(x: tensor[2, f32]) -> f32 = match Box { t: mul(&x, &x) } with {\n\
                \x20   | Box { t } => sum(t, cast(0, i32)) |> tensor_to_scalar\n\
                }\n\n";
    let source = format!(
        "{body}out = grad(fwd_field)(to_tensor([{}]))\n",
        fmt_f32_list(&base),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "static-tag runtime-field grad failed: {stderr}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad.len(), 2);
    for (i, g) in grad.iter().enumerate() {
        assert!(
            (g - 2.0 * base[i]).abs() < 1e-4,
            "analytic grad elem {i}: got {g}, want {}",
            2.0 * base[i]
        );
    }
    // Central-difference oracle from the forward loss.
    let h = 1e-2;
    for i in 0..base.len() {
        let mut xp = base.to_vec();
        let mut xm = base.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let fwd = |xs: &[f64]| {
            let src = format!("{body}out = fwd_field(to_tensor([{}]))\n", fmt_f32_list(xs),);
            let (stdout, stderr, ok) = eval_program(&src);
            assert!(ok, "forward eval failed: {stderr}");
            parse_scalar(&stdout)
        };
        let fd = (fwd(&xp) - fwd(&xm)) / (2.0 * h);
        assert!(
            (grad[i] - fd).abs() < 5e-2,
            "elem {i}: analytic {} vs finite-difference {fd}",
            grad[i]
        );
    }
}

/// Static selection of a constant taken arm must behave exactly like the
/// no-match constant-function control: both produce the exact shaped zero
/// cotangent. The match must introduce no divergence from the control.
#[test]
fn issue_520_d1_constant_taken_arm_matches_constant_fn_control() {
    let match_source = format!(
        "{D1_HEADER}{}\nout = grad(fwd_match)(to_tensor([{}]))\n",
        d1_match_fn("sum(x, cast(0, i32)) |> tensor_to_scalar", "cast(0.0, f32)")
            .replace("match ModeA", "match ModeB"),
        fmt_f32_list(&[1.0, 2.0]),
    );
    let control_source = format!(
        "module Repro.CtrlConst\n\n\
         def fwd_const(x: tensor[2, f32]) -> f32 = cast(0.0, f32)\n\n\
         out = grad(fwd_const)(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (match_stdout, match_stderr, match_ok) = eval_program(&match_source);
    let (control_stdout, control_stderr, control_ok) = eval_program(&control_source);
    assert!(match_ok, "constant-arm grad failed: {match_stderr}");
    assert!(
        control_ok,
        "constant-fn grad control failed: {control_stderr}"
    );
    assert_eq!(parse_tensor_data(&match_stdout), vec![0.0, 0.0]);
    assert_eq!(parse_tensor_data(&control_stdout), vec![0.0, 0.0]);
}

// --- D1 compiled lane ----------------------------------------------------------

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
            "--emit-c",
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

/// D1 compiled lane: `chelis build --target c` of `out = grad(fwd_match)`
/// must compile, run, and agree with the eval lane and the analytic
/// gradient of the taken arm (2x for the sum-of-squares arm).
#[test]
fn issue_520_d1_grad_through_static_match_c_backend_agrees() {
    let source = format!(
        "{D1_HEADER}{}\nout = grad(fwd_match)\n",
        d1_match_fn(
            "sum(mul(&x, &x), cast(0, i32)) |> tensor_to_scalar",
            "cast(0.0, f32)"
        ),
    );
    let (_dir, build_dir) = build_c(&source, "match520");
    let driver = r#"
#include <stdio.h>
#include <string.h>
#include "chelis_runtime.h"
extern chelis_tensor* CHELIS_TEST_OUT(chelis_tensor* arg0);
static chelis_tensor* out(chelis_tensor* arg0) { chelis_tensor_retain(arg0); return arg0; }
int main(void) {
    int64_t shape[1] = {2};
    chelis_tensor* x = chelis_alloc(1, shape, CHELIS_DTYPE_F32);
    float xd[2] = {1.0f, 2.0f};
    chelis_tensor_write* x_guard = chelis_tensor_begin_write(x);
    chelis_write_view x_view = chelis_tensor_write_view(x_guard);
    memcpy(x_view.data, xd, sizeof(xd));
    chelis_tensor_end_write(x_guard);
    chelis_tensor* g = CHELIS_TEST_OUT(x);
    chelis_read_view g_view = chelis_tensor_read_view(g);
    if (g_view.count != 2) { printf("FAIL_SIZE %lld\n", (long long)g_view.count); return 1; }
    for (int i = 0; i < 2; i++) printf("%.6f\n", ((const float *)g_view.data)[i]);
    chelis_tensor_release(g);
    chelis_tensor_release(x);
    return 0;
}
"#
    .replace("CHELIS_TEST_OUT", &authored_c_symbol("out"));
    let stdout = compile_and_run(&build_dir, "match520", &driver);
    let c_grad: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect();
    assert_eq!(c_grad.len(), 2, "C grad length: {stdout}");
    // Eval lane on the same body.
    let eval_source = format!(
        "{D1_HEADER}{}\nout = grad(fwd_match)(to_tensor([{}]))\n",
        d1_match_fn(
            "sum(mul(&x, &x), cast(0, i32)) |> tensor_to_scalar",
            "cast(0.0, f32)"
        ),
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

// --- D2: grad over an ADT/record argument (eval lane) --------------------------

fn d2_source(arm_body: &str, input: &[f64]) -> String {
    format!(
        "module Repro.D2\n\n\
         type Box =\n\
         \x20 | Box {{ t: tensor[2, f32] }}\n\n\
         def fwd_box(p: Box) -> f32 = match p with {{\n\
         \x20   | Box {{ t }} => {arm_body}\n\
         \x20 }}\n\
\n\
         out = grad(fwd_box)(Box {{ t: to_tensor([{}]) }})\n",
        fmt_f32_list(input),
    )
}

/// D2 acceptance oracle (issue #520 facet 1): grad of a Box-taking function
/// returns a Box-shaped gradient whose `t` field is [1, 1].
#[test]
fn issue_520_d2_grad_adt_arg_eval_returns_box_gradient() {
    let source = d2_source("sum(t, cast(0, i32)) |> tensor_to_scalar", &[1.0, 2.0]);
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "D2 grad eval failed: {stderr}");
    // Structural check: the gradient prints as a Box-constructed value,
    // not a bare tensor or tuple.
    assert!(
        stdout.contains("Box("),
        "gradient must be Box-shaped (pytree contract): {stdout}"
    );
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad.len(), 2, "grad field shape: {stdout}");
    for (i, g) in grad.iter().enumerate() {
        assert!((g - 1.0).abs() < 1e-6, "grad elem {i}: got {g}, want 1");
    }
}

/// D2 nonlinear + FD oracle: grad of sum(mul(t, t)) w.r.t. Box { t } is
/// Box(2t), validated against a central-difference gradient of the forward.
#[test]
fn issue_520_d2_grad_adt_arg_nonlinear_matches_fd() {
    let base = [1.5, -0.5];
    let arm = "sum(mul(&t, &t), cast(0, i32)) |> tensor_to_scalar";
    let (stdout, stderr, ok) = eval_program(&d2_source(arm, &base));
    assert!(ok, "D2 nonlinear grad eval failed: {stderr}");
    assert!(stdout.contains("Box("), "Box-shaped gradient: {stdout}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad.len(), 2);
    for (i, g) in grad.iter().enumerate() {
        assert!(
            (g - 2.0 * base[i]).abs() < 1e-4,
            "analytic grad elem {i}: got {g}, want {}",
            2.0 * base[i]
        );
    }
    let h = 1e-2;
    for i in 0..base.len() {
        let mut xp = base.to_vec();
        let mut xm = base.to_vec();
        xp[i] += h;
        xm[i] -= h;
        let fwd = |xs: &[f64]| {
            let src = format!(
                "module Repro.D2\n\n\
                 type Box =\n\
                 \x20 | Box {{ t: tensor[2, f32] }}\n\n\
                 def fwd_box(p: Box) -> f32 = match p with {{\n\
                 \x20   | Box {{ t }} => {arm}\n\
                 \x20 }}\n\
\n\
                 out = fwd_box(Box {{ t: to_tensor([{}]) }})\n",
                fmt_f32_list(xs),
            );
            let (stdout, stderr, ok) = eval_program(&src);
            assert!(ok, "forward eval failed: {stderr}");
            parse_scalar(&stdout)
        };
        let fd = (fwd(&xp) - fwd(&xm)) / (2.0 * h);
        assert!(
            (grad[i] - fd).abs() < 5e-2,
            "elem {i}: analytic {} vs finite-difference {fd}",
            grad[i]
        );
    }
}

/// D2 over a multi-constructor sum type: the gradient corresponds to the
/// constructed variant (design doc Phase 2: "the gradient corresponds to
/// whichever variant was constructed"). The value's tag statically selects
/// the arm, so grad(f)(B { u }) differentiates the B arm only.
#[test]
fn issue_520_d2_multi_ctor_value_gradient_follows_taken_variant() {
    let source = format!(
        "module Repro.D2Sum\n\n\
         type Pick =\n\
         \x20 | A {{ s: tensor[2, f32] }}\n\
         \x20 | B {{ u: tensor[2, f32] }}\n\n\
         def fwd_pick(p: Pick) -> f32 = match p with {{\n\
         \x20   | A {{ s }} => sum(s, cast(0, i32)) |> tensor_to_scalar\n\
         \x20   | B {{ u }} => sum(mul(&u, &u), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd_pick)(B {{ u: to_tensor([{}]) }})\n",
        fmt_f32_list(&[3.0, 4.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "multi-ctor grad eval failed: {stderr}");
    assert!(stdout.contains("B("), "B-shaped gradient: {stdout}");
    let grad = parse_tensor_data(&stdout);
    // B arm is sum of squares: grad = 2u = [6, 8]; the A arm (grad would
    // be [1, 1]) must NOT have been differentiated.
    assert_eq!(grad.len(), 2);
    assert!((grad[0] - 6.0).abs() < 1e-4, "grad[0]: {stdout}");
    assert!((grad[1] - 8.0).abs() < 1e-4, "grad[1]: {stdout}");
}

/// Composed usage: the checker now types `grad(f : Box -> f32)` as
/// `Box -> Box` (the pytree gradient type), so a def can declare and
/// return the gradient struct.
#[test]
fn issue_520_d2_composed_def_returns_typed_box_gradient() {
    let source = format!(
        "module Repro.D2Composed\n\n\
         type Box =\n\
         \x20 | Box {{ t: tensor[2, f32] }}\n\n\
         def fwd_box(p: Box) -> f32 = match p with {{\n\
         \x20   | Box {{ t }} => sum(mul(&t, &t), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         def jac(x: tensor[2, f32]) -> Box = grad(fwd_box)(Box {{ t: x }})\n\n\
         out = jac(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "composed D2 grad eval failed: {stderr}");
    assert!(stdout.contains("Box("), "Box-typed gradient: {stdout}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad, vec![2.0, 4.0], "grad of sum(t*t) is 2t: {stdout}");
}

/// D2 zero-fill: a two-tensor-field struct where only one field reaches
/// the output. The unused field's gradient must be an explicit zero
/// tensor OF ITS OWN SHAPE (here [3], distinct from the used field's
/// [2]), so the gradient struct mirrors the argument structure exactly.
#[test]
fn issue_520_d2_unused_field_gets_shaped_zero_gradient() {
    let source = format!(
        "module Repro.D2Zero\n\n\
         type Pair =\n\
         \x20 | Pair {{ a: tensor[2, f32], b: tensor[3, f32] }}\n\n\
         def fwd_pair(p: Pair) -> f32 = match p with {{\n\
         \x20   | Pair {{ a, b: _ }} => sum(mul(&a, &a), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd_pair)(Pair {{ a: to_tensor([{}]), b: to_tensor([{}]) }})\n",
        fmt_f32_list(&[1.0, 2.0]),
        fmt_f32_list(&[5.0, 6.0, 7.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "two-field D2 grad eval failed: {stderr}");
    assert!(stdout.contains("Pair("), "Pair-shaped gradient: {stdout}");
    assert!(
        stdout.contains("data=[2.0, 4.0]"),
        "used field grad 2a: {stdout}"
    );
    assert!(
        stdout.contains("shape=[3], data=[0.0, 0.0, 0.0]"),
        "unused field must get a zero tensor of its own [3] shape: {stdout}"
    );
}

/// A field declared through a zero-parameter `typealias` to a float
/// tensor is inside the D2 slice: the runtime's type-level gate must
/// resolve the alias the way the checker does, not reject the field as
/// an opaque nested type.
#[test]
fn issue_520_d2_alias_typed_float_field_accepted() {
    let source = format!(
        "module Repro.D2Alias\n\n\
         type V2 = tensor[2, f32]\n\n\
         type Box2 =\n\
         \x20 | Box2 {{ t: V2 }}\n\n\
         def fwd_alias(p: Box2) -> f32 = match p with {{\n\
         \x20   | Box2 {{ t }} => sum(mul(&t, &t), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd_alias)(Box2 {{ t: to_tensor([{}]) }})\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "alias-typed float field grad failed: {stderr}");
    assert!(stdout.contains("Box2("), "Box2-shaped gradient: {stdout}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad, vec![2.0, 4.0], "grad of sum(t*t) is 2t: {stdout}");
}

// --- Negative parity ------------------------------------------------------------

/// A scrutinee whose constructor is runtime-dependent (an `if` choosing
/// between constructors inside a called function) must be rejected
/// loudly, never silently resolved to one arm. The `if` lowering cannot
/// represent an ADT-valued branch, which is exactly what keeps a
/// data-dependent constructor out of static arm selection.
#[test]
fn issue_520_d1_runtime_ctor_through_if_still_rejected() {
    let source = format!(
        "module Repro.Neg0\n\n\
         type Mode =\n\
         \x20 | ModeA\n\
         \x20 | ModeB\n\n\
         def pick(c: f32) -> Mode = if c > 0.0 then ModeA else ModeB\n\n\
         def fwd_dynpick(x: tensor[2, f32]) -> f32 = match pick(tensor_to_scalar(sum(&x, cast(0, i32)))) with {{\n\
         \x20   | ModeA => sum(x, cast(0, i32)) |> tensor_to_scalar\n\
         \x20   | ModeB => cast(0.0, f32)\n\
         \x20 }}\n\
\n\
         out = grad(fwd_dynpick)(to_tensor([{}]))\n",
        fmt_f32_list(&[-1.0, -2.0]),
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(
        !ok,
        "a data-dependent constructor must never be statically selected"
    );
    assert!(
        stderr.contains("expected a single tensor value, got an ADT value"),
        "diagnostic must name the ADT-in-branch shape: {stderr}"
    );
}

/// A `match` whose scrutinee is a runtime value (not a compile-time-known
/// constructor) stays rejected, with a diagnostic naming the construct.
#[test]
fn issue_520_d1_runtime_scrutinee_match_still_rejected() {
    let source = format!(
        "module Repro.Neg1\n\n\
         def fwd_dyn(x: tensor[2, f32]) -> f32 = match tensor_to_scalar(sum(&x, cast(0, i32))) with {{\n\
         \x20   | 0.0 => cast(0.0, f32)\n\
         \x20   | _ => sum(x, cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd_dyn)(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(!ok, "runtime-scrutinee match grad must stay rejected");
    assert!(
        stderr.contains("match") && stderr.contains("runtime scrutinee"),
        "diagnostic must name the runtime-scrutinee match: {stderr}"
    );
}

/// A guarded arm encountered during static arm selection stays rejected
/// (guards need runtime evaluation), with a named diagnostic.
#[test]
fn issue_520_d1_guarded_arm_still_rejected() {
    let source = format!(
        "module Repro.Neg2\n\n\
         type Mode =\n\
         \x20 | ModeA\n\
         \x20 | ModeB\n\n\
         def fwd_guard(x: tensor[2, f32]) -> f32 = match ModeA with {{\n\
         \x20   | ModeA if true => sum(x, cast(0, i32)) |> tensor_to_scalar\n\
         \x20   | _ => cast(0.0, f32)\n\
         \x20 }}\n\
\n\
         out = grad(fwd_guard)(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(!ok, "guarded static match grad must stay rejected");
    assert!(
        stderr.contains("guard"),
        "diagnostic must name the arm guard: {stderr}"
    );
}

/// Mixed tensor/non-tensor fields are legal recursive cotangents: the float
/// field receives its gradient and the discrete field remains present as unit.
#[test]
fn issue_520_d2_mixed_field_struct_replaces_discrete_field_with_unit() {
    let source = format!(
        "module Repro.Neg3\n\n\
         type Mixed =\n\
         \x20 | Mixed {{ t: tensor[2, f32], n: i32 }}\n\n\
         def fwd_mixed(p: Mixed) -> f32 = match p with {{\n\
         \x20   | Mixed {{ t, n: _ }} => sum(t, cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd_mixed)(Mixed {{ t: to_tensor([{}]), n: cast(3, i32) }})\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(
        ok,
        "mixed-field ADT grad must preserve the discrete field as unit: {stderr}"
    );
    assert!(
        stdout.contains("Mixed(tensor(shape=[2], data=[1.0, 1.0]), ())"),
        "mixed-field cotangent must be Mixed([1,1], ()): {stdout}"
    );
}

/// A discrete field in a sibling variant does not poison a float-clean
/// executed variant. The cotangent preserves the executed constructor.
#[test]
fn issue_520_d2_mixed_sibling_variant_preserves_executed_constructor() {
    let source = format!(
        "module Repro.Neg6\n\n\
         type Pick =\n\
         \x20 | A {{ s: tensor[2, f32] }}\n\
         \x20 | B {{ u: tensor[2, f32], n: i32 }}\n\n\
         def fwd_pick(p: Pick) -> f32 = match p with {{\n\
         \x20   | A {{ s }} => sum(mul(&s, &s), cast(0, i32)) |> tensor_to_scalar\n\
         \x20   | B {{ u, n: _ }} => sum(u, cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd_pick)(A {{ s: to_tensor([{}]) }})\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(
        ok,
        "mixed sibling variant must not poison the executed constructor: {stderr}"
    );
    assert!(
        stdout.contains("A(tensor(shape=[2], data=[2.0, 4.0]))"),
        "executed A cotangent must preserve A and equal 2s: {stdout}"
    );
}

/// A pure enum (no fields in any variant) has no continuous payload:
/// explicitly selecting it as `wrt` is a checker error because the target
/// contains no differentiable float leaf.
#[test]
fn issue_520_d2_pure_enum_grad_rejected() {
    let source = "module Repro.Neg7\n\n\
type Mode =\n\
  | ModeA\n\
  | ModeB\n\n\
def fwd_mode(m: Mode) -> f32 = match m with {\n\
    | ModeA => cast(1.0, f32)\n\
    | ModeB => cast(2.0, f32)\n\
}\n\n\
out = grad(fwd_mode, wrt=m)(ModeA)\n";
    let (_stdout, stderr, ok) = eval_program(source);
    assert!(!ok, "grad over a pure enum must stay rejected");
    assert!(
        stderr.contains("grad `wrt` index 0 is not differentiable"),
        "diagnostic must name the all-unit target: {stderr}"
    );
}

// --- D2 multi-argument: grad over an ADT param alongside plain args ------------
//
// The chelis#520 closing bar: `grad(model_forward, wrt=params)(x, params)`
// works for a forward taking an ADT params argument ALONGSIDE plain tensor
// args. The result is a tuple whose ADT slot is the field-wise gradient
// struct and whose tensor slot is the bare tensor gradient (the pytree
// contract extended across arguments). Flipped from the pre-close negatives
// #20/#21.

/// The exact issue "together" reproducer: a forward with ADT params and a
/// plain tensor arg, differentiated wrt only the params struct. The result
/// is the field-wise gradient struct, packed in the constructor shape.
/// `sum(mul(x, w))` has analytic gradient wrt `w` equal to `x`.
#[test]
fn issue_520_together_multi_arg_adt_grad_wrt_params() {
    let x = [2.0, 3.0];
    let w = [5.0, 7.0];
    let source = format!(
        "module Repro.Together\n\n\
         type Params =\n\
         \x20 | Params {{ w: tensor[2, f32] }}\n\n\
         def model_forward(x: tensor[2, f32], p: Params) -> f32 = match p with {{\n\
         \x20   | Params {{ w }} => sum(mul(&x, &w), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(model_forward, wrt=p)(to_tensor([{x}]), Params {{ w: to_tensor([{w}]) }})\n",
        x = fmt_f32_list(&x),
        w = fmt_f32_list(&w),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "issue-shaped multi-arg ADT grad failed: {stderr}");
    // Single differentiated target (the ADT): the result is the bare
    // Params-shaped gradient struct, not a tuple.
    assert!(
        stdout.contains("Params("),
        "gradient must be a Params-shaped struct: {stdout}"
    );
    let grad = parse_tensor_data(&stdout);
    // d/dw sum(x * w) = x.
    assert_eq!(grad.len(), 2, "grad field shape: {stdout}");
    for (i, g) in grad.iter().enumerate() {
        assert!(
            (g - x[i]).abs() < 1e-6,
            "grad w elem {i}: got {g}, want {}",
            x[i]
        );
    }
}

/// Default `wrt` over the same shape differentiates BOTH arguments: the
/// result is a tuple whose slot 0 is the tensor gradient (wrt x) and whose
/// slot 1 is the Params-shaped gradient (wrt the struct). This is the
/// full multi-target pytree; slot 0 is `d/dx sum(x*w) = w` and slot 1 is
/// `d/dw sum(x*w) = x` packed as Params.
#[test]
fn issue_520_multi_arg_adt_grad_default_wrt_returns_tuple() {
    let x = [2.0, 3.0];
    let w = [5.0, 7.0];
    let source = format!(
        "module Repro.MultiDefault\n\n\
         type Params =\n\
         \x20 | Params {{ w: tensor[2, f32] }}\n\n\
         def model_forward(x: tensor[2, f32], p: Params) -> f32 = match p with {{\n\
         \x20   | Params {{ w }} => sum(mul(&x, &w), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(model_forward)(to_tensor([{x}]), Params {{ w: to_tensor([{w}]) }})\n",
        x = fmt_f32_list(&x),
        w = fmt_f32_list(&w),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "default-wrt multi-arg ADT grad failed: {stderr}");
    // Slot 0: bare tensor gradient wrt x = w = [5, 7].
    let tensor_line = stdout
        .lines()
        .find(|l| l.contains("out.0"))
        .unwrap_or_else(|| panic!("no out.0 slot: {stdout}"));
    assert!(
        !tensor_line.contains("Params(") && tensor_line.contains("data=[5.0, 7.0]"),
        "slot 0 must be the bare tensor gradient w = [5,7]: {stdout}"
    );
    // Slot 1: Params-shaped gradient wrt w = x = [2, 3].
    let adt_line = stdout
        .lines()
        .find(|l| l.contains("out.1"))
        .unwrap_or_else(|| panic!("no out.1 slot: {stdout}"));
    assert!(
        adt_line.contains("Params(") && adt_line.contains("data=[2.0, 3.0]"),
        "slot 1 must be a Params-shaped gradient x = [2,3]: {stdout}"
    );
}

/// Shared-adjoint distinct-root path: `sum(add(t, y))` has adjoint 1 for
/// BOTH the ADT field and the tensor arg, so the two gradient roots are the
/// same DAG node. Each pytree leaf must still appear as its own root (the
/// `add_root` id-dedup would otherwise collapse the tuple to one slot).
#[test]
fn issue_520_multi_arg_adt_grad_shared_adjoint_keeps_both_slots() {
    let source = format!(
        "module Repro.Shared\n\n\
         type Box =\n\
         \x20 | Box {{ t: tensor[2, f32] }}\n\n\
         def fwd_two(p: Box, y: tensor[2, f32]) -> f32 = match p with {{\n\
         \x20   | Box {{ t }} => sum(add(t, y), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd_two)(Box {{ t: to_tensor([{a}]) }}, to_tensor([{a}]))\n",
        a = fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "shared-adjoint multi-arg grad failed: {stderr}");
    assert!(
        stdout.contains("Box(tensor(shape=[2], data=[1.0, 1.0]))"),
        "ADT slot must survive as its own [1,1] gradient: {stdout}"
    );
    let tensor_line = stdout
        .lines()
        .find(|l| l.contains("out.1"))
        .unwrap_or_else(|| panic!("no out.1 slot: {stdout}"));
    assert!(
        tensor_line.contains("data=[1.0, 1.0]"),
        "tensor slot must survive as its own [1,1] gradient: {stdout}"
    );
}

/// Multi-arg nonlinear + finite-difference oracle: `sum(add(mul(t,t),
/// mul(y,y)))` grads to 2t (Box slot) and 2y (tensor slot), each validated
/// against a central-difference gradient of the forward.
#[test]
fn issue_520_multi_arg_adt_grad_nonlinear_matches_fd() {
    let base_t = [1.5, -0.5];
    let base_y = [0.75, 2.0];
    let program = |t: &[f64], y: &[f64], applied: &str| {
        format!(
            "module Repro.MultiFD\n\n\
             type Box =\n\
             \x20 | Box {{ t: tensor[2, f32] }}\n\n\
             def fwd_two(p: Box, y: tensor[2, f32]) -> f32 = match p with {{\n\
             \x20   | Box {{ t }} => sum(add(mul(&t, &t), mul(&y, &y)), cast(0, i32)) |> tensor_to_scalar\n\
             \x20 }}\n\
\n\
             out = {applied}(Box {{ t: to_tensor([{tl}]) }}, to_tensor([{yl}]))\n",
            tl = fmt_f32_list(t),
            yl = fmt_f32_list(y),
        )
    };
    let (stdout, stderr, ok) = eval_program(&program(&base_t, &base_y, "grad(fwd_two)"));
    assert!(ok, "multi-arg nonlinear grad failed: {stderr}");
    // Box slot (out.0): 2t; tensor slot (out.1): 2y.
    let box_line = stdout.lines().find(|l| l.contains("out.0")).unwrap();
    let want_t: Vec<f64> = base_t.iter().map(|v| 2.0 * v).collect();
    for (i, w) in want_t.iter().enumerate() {
        assert!(
            box_line.contains(&format!("{w:?}")),
            "box grad elem {i} want {w}: {stdout}"
        );
    }
    // Finite-difference each input against the forward loss.
    let fwd = |t: &[f64], y: &[f64]| {
        let (s, e, ok) = eval_program(&program(t, y, "fwd_two"));
        assert!(ok, "forward failed: {e}");
        parse_scalar(&s)
    };
    let h = 1e-2;
    for i in 0..base_t.len() {
        let (mut tp, mut tm) = (base_t.to_vec(), base_t.to_vec());
        tp[i] += h;
        tm[i] -= h;
        let fd = (fwd(&tp, &base_y) - fwd(&tm, &base_y)) / (2.0 * h);
        assert!(
            (2.0 * base_t[i] - fd).abs() < 5e-2,
            "t elem {i}: analytic {} vs fd {fd}",
            2.0 * base_t[i]
        );
    }
    for i in 0..base_y.len() {
        let (mut yp, mut ym) = (base_y.to_vec(), base_y.to_vec());
        yp[i] += h;
        ym[i] -= h;
        let fd = (fwd(&base_t, &yp) - fwd(&base_t, &ym)) / (2.0 * h);
        assert!(
            (2.0 * base_y[i] - fd).abs() < 5e-2,
            "y elem {i}: analytic {} vs fd {fd}",
            2.0 * base_y[i]
        );
    }
}

/// `wrt`-narrowing the OTHER way: differentiate only the plain tensor arg
/// among the several. No ADT slot is produced; the result is the bare
/// tensor gradient. `d/dy sum(t + y) = [1, 1]`.
#[test]
fn issue_520_multi_arg_grad_wrt_tensor_only_returns_bare() {
    let source = format!(
        "module Repro.WrtTensor\n\n\
         type Box =\n\
         \x20 | Box {{ t: tensor[2, f32] }}\n\n\
         def fwd_two(p: Box, y: tensor[2, f32]) -> f32 = match p with {{\n\
         \x20   | Box {{ t }} => sum(add(t, y), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd_two, wrt=y)(Box {{ t: to_tensor([{a}]) }}, to_tensor([{a}]))\n",
        a = fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "wrt-tensor-only multi-arg grad failed: {stderr}");
    assert!(
        !stdout.contains("Box("),
        "wrt=y must not pack an ADT slot: {stdout}"
    );
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad, vec![1.0, 1.0], "d/dy sum(t + y) = [1,1]: {stdout}");
}

/// Multi-arg zero-fill: an unused ADT field in a multi-argument grad still
/// gets an explicit zero tensor OF ITS OWN SHAPE, and the tensor slot is
/// preserved. Guards the per-argument boundary alignment of the repack.
#[test]
fn issue_520_multi_arg_adt_grad_unused_field_shaped_zero() {
    let source = format!(
        "module Repro.MultiZero\n\n\
         type Pair =\n\
         \x20 | Pair {{ a: tensor[2, f32], b: tensor[3, f32] }}\n\n\
         def fwd(p: Pair, y: tensor[2, f32]) -> f32 = match p with {{\n\
         \x20   | Pair {{ a, b: _ }} => sum(add(mul(&a, &a), y), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd)(Pair {{ a: to_tensor([{a}]), b: to_tensor([{b}]) }}, to_tensor([{a}]))\n",
        a = fmt_f32_list(&[1.0, 2.0]),
        b = fmt_f32_list(&[5.0, 6.0, 7.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "multi-arg zero-fill grad failed: {stderr}");
    // Pair slot: a grad = 2a = [2,4]; b unused -> zeros of its own [3] shape.
    assert!(
        stdout.contains("Pair(") && stdout.contains("data=[2.0, 4.0]"),
        "used field grad 2a: {stdout}"
    );
    assert!(
        stdout.contains("shape=[3], data=[0.0, 0.0, 0.0]"),
        "unused field must get a [3]-shaped zero: {stdout}"
    );
    // Tensor slot: d/dy = [1,1].
    let tensor_line = stdout
        .lines()
        .find(|l| l.contains("out.1"))
        .unwrap_or_else(|| panic!("no out.1 slot: {stdout}"));
    assert!(
        tensor_line.contains("data=[1.0, 1.0]"),
        "tensor slot d/dy = [1,1]: {stdout}"
    );
}

/// Compiled lane: `out = grad(fwd_box)` over an ADT-typed param stays
/// rejected (no ADT values in the C ABI), with the existing loud
/// unresolved-grad build error.
#[test]
fn issue_520_d2_build_lane_adt_param_grad_export_still_rejected() {
    let source = "module Repro.Neg5\n\n\
type Box =\n\
  | Box { t: tensor[2, f32] }\n\n\
def fwd_box(p: Box) -> f32 = match p with {\n\
    | Box { t } => sum(t, cast(0, i32)) |> tensor_to_scalar\n\
}\n\n\
out = grad(fwd_box)\n";
    let dir = tempdir().expect("tempdir");
    let src_path = dir.path().join("neg5.ch");
    fs::write(&src_path, source).expect("write");
    let build_dir = dir.path().join("build");
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(dir.path())
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            src_path.to_str().unwrap(),
            "--target",
            "c",
            "-o",
            build_dir.to_str().unwrap(),
        ])
        .output()
        .expect("run");
    assert!(
        !output.status.success(),
        "ADT-param grad export must stay rejected in the compiled lane"
    );
    let stderr = String::from_utf8_lossy(&output.stderr);
    assert!(
        stderr.contains("grad"),
        "diagnostic must mention grad: {stderr}"
    );
}

// --- chelis#614: multi-target grad binding display ----------------------------

/// chelis#614 regression: a multi-target gradient binding
/// `out = grad(f)(a, b)` (tuple-valued grad payload) must PRINT its values
/// in the eval lane, not silently drop them behind the misleading
/// "input contains only def declarations" breadcrumb. The tuple-valued
/// binding owns flattened root names (`out.0`, `out.1`), and the host
/// runtime must eager-evaluate the `out` def and reconstruct each slot.
#[test]
fn issue_614_multi_target_grad_binding_displays_values() {
    let source = format!(
        "module Repro.GradTuple614\n\n\
         def fwd(x: tensor[2, f32], y: tensor[2, f32]) -> f32 = sum(add(mul(&x, &x), mul(&y, &y)), cast(0, i32)) |> tensor_to_scalar\n\n\
         out = grad(fwd)(to_tensor([{x}]), to_tensor([{y}]))\n",
        x = fmt_f32_list(&[1.0, 2.0]),
        y = fmt_f32_list(&[5.0, 6.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "multi-target grad eval failed: {stderr}");
    assert!(
        !stderr.contains("only def declarations"),
        "must not emit the defs-only breadcrumb for a tuple-valued binding: {stderr}"
    );
    // d/dx sum(x^2 + y^2) = 2x = [2,4]; d/dy = 2y = [10,12].
    assert!(
        stdout.contains("data=[2.0, 4.0]"),
        "slot 0 must display 2x = [2,4]: {stdout}"
    );
    assert!(
        stdout.contains("data=[10.0, 12.0]"),
        "slot 1 must display 2y = [10,12]: {stdout}"
    );
}

// --- Adversarial multi-argument alignment pins (chelis#520 D2) -----------------

/// TWO ADT arguments: each gradient slot must map to its OWN constructor.
/// `sum(u*u + v*v)` grads to `2u` for the first ADT (`A`) and `2v` for the
/// second (`B`); a per-argument repack that transposed the slots or shared
/// a boundary would surface here as a mislabeled constructor or value.
#[test]
fn issue_520_multi_arg_two_adt_args_align_per_constructor() {
    let source = format!(
        "module Repro.TwoAdt\n\n\
         type A =\n\
         \x20 | A {{ u: tensor[2, f32] }}\n\n\
         type B =\n\
         \x20 | B {{ v: tensor[2, f32] }}\n\n\
         def fwd(p: A, q: B) -> f32 = match p with {{\n\
         \x20   | A {{ u }} =>\n\
         \x20     match q with {{\n\
         \x20       | B {{ v }} => sum(add(mul(&u, &u), mul(&v, &v)), cast(0, i32)) |> tensor_to_scalar\n\
         \x20     }}\n\
         \x20 }}\n\
\n\
         out = grad(fwd)(A {{ u: to_tensor([{u}]) }}, B {{ v: to_tensor([{v}]) }})\n",
        u = fmt_f32_list(&[1.0, 2.0]),
        v = fmt_f32_list(&[3.0, 4.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "two-ADT-arg grad failed: {stderr}");
    let a_line = stdout
        .lines()
        .find(|l| l.contains("out.0"))
        .unwrap_or_else(|| panic!("no out.0 slot: {stdout}"));
    assert!(
        a_line.contains("A(") && a_line.contains("data=[2.0, 4.0]"),
        "slot 0 must be A-shaped 2u = [2,4]: {stdout}"
    );
    let b_line = stdout
        .lines()
        .find(|l| l.contains("out.1"))
        .unwrap_or_else(|| panic!("no out.1 slot: {stdout}"));
    assert!(
        b_line.contains("B(") && b_line.contains("data=[6.0, 8.0]"),
        "slot 1 must be B-shaped 2v = [6,8]: {stdout}"
    );
}

/// Three-way shared-adjoint distinct-root path: `sum(add(add(t, y), z))`
/// has adjoint `1` for the ADT field `t`, the tensor `y`, and the tensor
/// `z` -- all three the SAME CSE-collapsed DAG node. Each pytree leaf must
/// still keep its own root (the identity-`Copy` materialization in
/// `lower_subexpr_program_inner`), so the tuple carries three distinct
/// [1,1] slots instead of collapsing onto the first. Stresses the Copy fix
/// harder than the two-way `shared_adjoint` pin.
#[test]
fn issue_520_multi_arg_three_way_shared_adjoint_keeps_all_slots() {
    let source = format!(
        "module Repro.ThreeShare\n\n\
         type Box =\n\
         \x20 | Box {{ t: tensor[2, f32] }}\n\n\
         def fwd(p: Box, y: tensor[2, f32], z: tensor[2, f32]) -> f32 = match p with {{\n\
         \x20   | Box {{ t }} => sum(add(add(t, y), z), cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd)(Box {{ t: to_tensor([{a}]) }}, to_tensor([{a}]), to_tensor([{a}]))\n",
        a = fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "three-way shared-adjoint grad failed: {stderr}");
    assert!(
        stdout.contains("Box(tensor(shape=[2], data=[1.0, 1.0]))"),
        "ADT slot must survive as its own [1,1] gradient: {stdout}"
    );
    for slot in ["out.1", "out.2"] {
        let line = stdout
            .lines()
            .find(|l| l.contains(slot))
            .unwrap_or_else(|| panic!("no {slot} slot: {stdout}"));
        assert!(
            line.contains("data=[1.0, 1.0]"),
            "{slot} must survive as its own [1,1] gradient: {stdout}"
        );
    }
}

/// Soundness pin: a multi-argument grad whose plain tensor argument is
/// UNUSED (no adjoint) yields a shaped ZERO for that slot, not a dropped
/// root. The ADT slot keeps its field roots and the unused tensor slot is
/// filled with its own [0,0] gradient, so the per-slot boundaries stay
/// aligned (the pytree contract: the gradient of an unused input is zero).
/// The IR lowering zero-fills the adjoint-free tensor slot the same way it
/// already zero-fills adjoint-free ADT fields (chelis#520 D2 / chelis#614).
#[test]
fn issue_520_multi_arg_adt_unused_tensor_zero_slot() {
    let source = format!(
        "module Repro.DropTensor\n\n\
         type Box =\n\
         \x20 | Box {{ t: tensor[2, f32] }}\n\n\
         def fwd(p: Box, y: tensor[2, f32]) -> f32 = match p with {{\n\
         \x20   | Box {{ t }} => sum(t, cast(0, i32)) |> tensor_to_scalar\n\
         \x20 }}\n\
\n\
         out = grad(fwd)(Box {{ t: to_tensor([{a}]) }}, to_tensor([{a}]))\n",
        a = fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(
        ok,
        "unused-tensor-adjoint multi-arg grad must succeed with a zero slot: {stderr}"
    );
    // ADT slot (out.0): d/d t of sum(t) = [1,1], packed in the Box shape.
    assert!(
        stdout.contains("Box(tensor(shape=[2], data=[1.0, 1.0]))"),
        "ADT slot must be Box([1,1]): {stdout}"
    );
    // Tensor slot (out.1): y is unused, so its gradient is a shaped zero, not
    // a dropped/mislabeled slot.
    let tensor_line = stdout
        .lines()
        .find(|l| l.contains("out.1"))
        .unwrap_or_else(|| panic!("no out.1 slot: {stdout}"));
    assert!(
        tensor_line.contains("data=[0.0, 0.0]"),
        "unused tensor slot must be a [0,0] zero gradient: {stdout}"
    );
}

/// Pure-tensor multi-argument grad with an UNUSED MIDDLE argument. The
/// analytic Jacobian is `d/dx = 2x`, `d/dy = 0` (y is not read), `d/dz = 2z`.
/// Before the tensor-lane zero-fill, the adjoint-free `y` slot was dropped
/// from the result tuple, so the eval-root display (chelis#614, fixed
/// `out.0..out.N` names) shifted z's gradient into the `out.1` (y) slot and
/// dropped `out.2` entirely -- a silently wrong Jacobian. This pins that the
/// unused slot is exactly zero and does NOT carry the neighbor's gradient.
#[test]
fn issue_520_multi_arg_pure_tensor_unused_middle_zero_slot() {
    let source = format!(
        "module Repro.PureMid\n\n\
         def fwd(x: tensor[2, f32], y: tensor[2, f32], z: tensor[2, f32]) -> f32 = sum(add(mul(&x, &x), mul(&z, &z)), cast(0, i32)) |> tensor_to_scalar\n\n\
         out = grad(fwd)(to_tensor([{x}]), to_tensor([{y}]), to_tensor([{z}]))\n",
        x = fmt_f32_list(&[1.0, 2.0]),
        y = fmt_f32_list(&[3.0, 4.0]),
        z = fmt_f32_list(&[5.0, 6.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(
        ok,
        "pure-tensor multi-arg grad with an unused middle arg failed: {stderr}"
    );
    // out.0 = d/dx = 2x = [2,4].
    let l0 = stdout
        .lines()
        .find(|l| l.contains("out.0"))
        .unwrap_or_else(|| panic!("no out.0 slot: {stdout}"));
    assert!(
        l0.contains("data=[2.0, 4.0]"),
        "out.0 must be 2x = [2,4]: {stdout}"
    );
    // out.1 = d/dy: y is unused, so this slot must be a shaped zero. The bug
    // this pins would instead shift z's gradient ([10,12]) into the y slot.
    let l1 = stdout
        .lines()
        .find(|l| l.contains("out.1"))
        .unwrap_or_else(|| panic!("no out.1 slot: {stdout}"));
    assert!(
        l1.contains("data=[0.0, 0.0]"),
        "out.1 (unused y) must be a [0,0] zero gradient: {stdout}"
    );
    assert!(
        !l1.contains("data=[10.0, 12.0]"),
        "out.1 must NOT carry z's gradient (the mislabel this fix removes): {stdout}"
    );
    // out.2 = d/dz = 2z = [10,12], in its own slot (not dropped).
    let l2 = stdout
        .lines()
        .find(|l| l.contains("out.2"))
        .unwrap_or_else(|| panic!("no out.2 slot (z gradient was dropped): {stdout}"));
    assert!(
        l2.contains("data=[10.0, 12.0]"),
        "out.2 must be 2z = [10,12]: {stdout}"
    );
}

/// Pure-tensor multi-argument grad with an UNUSED LEADING argument. The
/// analytic Jacobian is `d/dx = 0` (x is not read), `d/dy = 2y`. Before the
/// zero-fill, dropping the leading `x` root left only one surviving root,
/// which `pack_dag_roots` collapses to a bare tensor; every `out.N` lookup
/// then missed and the whole gradient vanished behind the misleading
/// "input contains only def declarations; nothing to evaluate" breadcrumb.
/// This pins that the gradient survives with `out.0` a shaped zero.
#[test]
fn issue_520_multi_arg_pure_tensor_unused_leading_no_vanish() {
    let source = format!(
        "module Repro.PureLead\n\n\
         def fwd(x: tensor[2, f32], y: tensor[2, f32]) -> f32 = sum(mul(&y, &y), cast(0, i32)) |> tensor_to_scalar\n\n\
         out = grad(fwd)(to_tensor([{x}]), to_tensor([{y}]))\n",
        x = fmt_f32_list(&[1.0, 2.0]),
        y = fmt_f32_list(&[3.0, 4.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(
        ok,
        "pure-tensor multi-arg grad with an unused leading arg failed: {stderr}"
    );
    assert!(
        !stderr.contains("nothing to evaluate"),
        "gradient must not vanish behind the defs-only breadcrumb: {stderr}"
    );
    // out.0 = d/dx: x is unused -> [0,0] (not dropped, which previously
    // collapsed the whole tuple to a single bare tensor).
    let l0 = stdout
        .lines()
        .find(|l| l.contains("out.0"))
        .unwrap_or_else(|| panic!("no out.0 slot: {stdout}"));
    assert!(
        l0.contains("data=[0.0, 0.0]"),
        "out.0 (unused x) must be a [0,0] zero gradient: {stdout}"
    );
    // out.1 = d/dy = 2y = [6,8].
    let l1 = stdout
        .lines()
        .find(|l| l.contains("out.1"))
        .unwrap_or_else(|| panic!("no out.1 slot: {stdout}"));
    assert!(
        l1.contains("data=[6.0, 8.0]"),
        "out.1 must be 2y = [6,8]: {stdout}"
    );
}
