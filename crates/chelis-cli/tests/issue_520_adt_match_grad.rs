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
//! over a single ADT argument whose fields are all float tensors returns a
//! structurally matching gradient value (`Ctor(grad_t, ...)`), the pytree
//! contract of spec/design/differentiable_language.md Decision 6 / Phase 2.
//! The compiled lane keeps rejecting an ADT-param grad export (the C ABI
//! has no ADT value representation); that rejection is pinned below.
//!
//! Negative parity (each pinned with its diagnostic):
//!   - runtime (non-constructor) scrutinee in a differentiated `match`
//!   - guarded arm reached during static arm selection
//!   - mixed tensor/non-tensor ADT fields in a grad argument
//!   - ADT grad argument in a multi-argument call
//!   - compiled-lane `out = grad(f)` export over an ADT-typed param

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

// --- D1: grad through a static-scrutinee match --------------------------------

const D1_HEADER: &str = "module Repro.D1\n\n\
type Mode =\n\
  | ModeA\n\
  | ModeB\n\n";

/// The issue's D1 reproducer body: match on a statically-known nullary
/// constructor; the taken arm is linear in `x`, the dead arm is constant.
fn d1_match_fn(body_a: &str, body_b: &str) -> String {
    format!(
        "def fwd_match(x: tensor[2, f32]) -> f32 = {{\n\
         \x20 match ModeA with {{\n\
         \x20   | ModeA => {body_a}\n\
         \x20   | ModeB => {body_b}\n\
         \x20 }}\n\
         }}\n"
    )
}

/// D1 acceptance oracle (issue #520 facet 2): grad through the match body
/// equals the analytic gradient of the TAKEN arm, [1, 1].
#[test]
fn issue_520_d1_grad_through_static_match_eval() {
    let source = format!(
        "{D1_HEADER}{}\nout = grad(fwd_match)(to_tensor([{}]))\n",
        d1_match_fn(
            "sum(x, cast(0, int32)) |> tensor_to_scalar",
            "cast(0.0, f32)"
        ),
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
         def fwd_plain(x: tensor[2, f32]) -> f32 = {{\n\
         \x20 sum(x, cast(0, int32)) |> tensor_to_scalar\n\
         }}\n\n\
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
        "sum(mul(&x, &x), cast(0, int32)) |> tensor_to_scalar",
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
            "sum(mul(&x, &x), cast(0, int32)) |> tensor_to_scalar",
            "cast(0.0, f32)"
        ),
    );
    let (_dir, build_dir) = build_c(&source, "match520");
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
    let stdout = compile_and_run(&build_dir, "match520", driver);
    let c_grad: Vec<f64> = stdout
        .lines()
        .map(|l| l.trim().parse::<f64>().expect("element"))
        .collect();
    assert_eq!(c_grad.len(), 2, "C grad length: {stdout}");
    // Eval lane on the same body.
    let eval_source = format!(
        "{D1_HEADER}{}\nout = grad(fwd_match)(to_tensor([{}]))\n",
        d1_match_fn(
            "sum(mul(&x, &x), cast(0, int32)) |> tensor_to_scalar",
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
         def fwd_box(p: Box) -> f32 = {{\n\
         \x20 match p with {{\n\
         \x20   | Box {{ t: t }} => {arm_body}\n\
         \x20 }}\n\
         }}\n\n\
         out = grad(fwd_box)(Box {{ t: to_tensor([{}]) }})\n",
        fmt_f32_list(input),
    )
}

/// D2 acceptance oracle (issue #520 facet 1): grad of a Box-taking function
/// returns a Box-shaped gradient whose `t` field is [1, 1].
#[test]
fn issue_520_d2_grad_adt_arg_eval_returns_box_gradient() {
    let source = d2_source("sum(t, cast(0, int32)) |> tensor_to_scalar", &[1.0, 2.0]);
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
    let arm = "sum(mul(&t, &t), cast(0, int32)) |> tensor_to_scalar";
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
                 def fwd_box(p: Box) -> f32 = {{\n\
                 \x20 match p with {{\n\
                 \x20   | Box {{ t: t }} => {arm}\n\
                 \x20 }}\n\
                 }}\n\n\
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
         def fwd_pick(p: Pick) -> f32 = {{\n\
         \x20 match p with {{\n\
         \x20   | A {{ s: s }} => sum(s, cast(0, int32)) |> tensor_to_scalar\n\
         \x20   | B {{ u: u }} => sum(mul(&u, &u), cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         }}\n\n\
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
         def fwd_box(p: Box) -> f32 = {{\n\
         \x20 match p with {{\n\
         \x20   | Box {{ t: t }} => sum(mul(&t, &t), cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         }}\n\n\
         def jac(x: tensor[2, f32]) -> Box = {{\n\
         \x20 grad(fwd_box)(Box {{ t: x }})\n\
         }}\n\n\
         out = jac(to_tensor([{}]))\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (stdout, stderr, ok) = eval_program(&source);
    assert!(ok, "composed D2 grad eval failed: {stderr}");
    assert!(stdout.contains("Box("), "Box-typed gradient: {stdout}");
    let grad = parse_tensor_data(&stdout);
    assert_eq!(grad, vec![2.0, 4.0], "grad of sum(t*t) is 2t: {stdout}");
}

// --- Negative parity ------------------------------------------------------------

/// A `match` whose scrutinee is a runtime value (not a compile-time-known
/// constructor) stays rejected, with a diagnostic naming the construct.
#[test]
fn issue_520_d1_runtime_scrutinee_match_still_rejected() {
    let source = format!(
        "module Repro.Neg1\n\n\
         def fwd_dyn(x: tensor[2, f32]) -> f32 = {{\n\
         \x20 match tensor_to_scalar(sum(&x, cast(0, int32))) with {{\n\
         \x20   | 0.0 => cast(0.0, f32)\n\
         \x20   | _ => sum(x, cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         }}\n\n\
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
         def fwd_guard(x: tensor[2, f32]) -> f32 = {{\n\
         \x20 match ModeA with {{\n\
         \x20   | ModeA if true => sum(x, cast(0, int32)) |> tensor_to_scalar\n\
         \x20   | _ => cast(0.0, f32)\n\
         \x20 }}\n\
         }}\n\n\
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

/// Mixed tensor/non-tensor fields in an ADT grad argument stay rejected
/// with a diagnostic naming the non-differentiable field.
#[test]
fn issue_520_d2_mixed_field_struct_rejected() {
    let source = format!(
        "module Repro.Neg3\n\n\
         type Mixed =\n\
         \x20 | Mixed {{ t: tensor[2, f32], n: int32 }}\n\n\
         def fwd_mixed(p: Mixed) -> f32 = {{\n\
         \x20 match p with {{\n\
         \x20   | Mixed {{ t: t, n: _ }} => sum(t, cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         }}\n\n\
         out = grad(fwd_mixed)(Mixed {{ t: to_tensor([{}]), n: cast(3, int32) }})\n",
        fmt_f32_list(&[1.0, 2.0]),
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(!ok, "mixed-field ADT grad arg must stay rejected");
    assert!(
        stderr.contains("`n`") && stderr.contains("float tensor"),
        "diagnostic must name the non-tensor field: {stderr}"
    );
}

/// An ADT grad argument in a multi-argument call stays rejected (the D2
/// slice packs gradients for single-argument calls only).
#[test]
fn issue_520_d2_multi_arg_adt_grad_rejected() {
    let source = format!(
        "module Repro.Neg4\n\n\
         type Box =\n\
         \x20 | Box {{ t: tensor[2, f32] }}\n\n\
         def fwd_two(p: Box, y: tensor[2, f32]) -> f32 = {{\n\
         \x20 match p with {{\n\
         \x20   | Box {{ t: t }} => sum(add(t, y), cast(0, int32)) |> tensor_to_scalar\n\
         \x20 }}\n\
         }}\n\n\
         out = grad(fwd_two)(Box {{ t: to_tensor([{a}]) }}, to_tensor([{a}]))\n",
        a = fmt_f32_list(&[1.0, 2.0]),
    );
    let (_stdout, stderr, ok) = eval_program(&source);
    assert!(!ok, "multi-arg ADT grad must stay rejected");
    assert!(
        stderr.contains("single-argument"),
        "diagnostic must name the single-argument restriction: {stderr}"
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
def fwd_box(p: Box) -> f32 = {\n\
  match p with {\n\
    | Box { t: t } => sum(t, cast(0, int32)) |> tensor_to_scalar\n\
  }\n\
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
