//! Issue Chelis-Lang/chelis#289 — `grad` over a function whose callee
//! carries a precision **variable** `p` must monomorphize `p` to the
//! concrete call-site precision, exactly as an ordinary (non-grad) call
//! through the same polymorphic callee already does.
//!
//! The reproducer differentiates `loss`, a fully-concrete `f32` entry
//! point, whose body routes through `lin_p`, a precision-polymorphic
//! callee written in the canonical def-level explicit-quantifier form
//! `def lin_p[p: Numeric](x: tensor[2, p], ...)` (spec §P4b). Before the fix the
//! grad transform lowered `loss`'s body in a fresh sub-context whose
//! precision-substitution map was empty, so inlining `lin_p` tripped the
//! §5.8.1 "monomorphization missed precision var `p`" tripwire — even
//! though the only call site supplies a concrete `f32`.
//!
//! Spec authority: spec/04-type-system.md §5.7 (grad), §5.8 / §5.8.1
//! (precision monomorphization). The fix seeds the grad sub-context's
//! precision substitutions from the call site so the polymorphic callee
//! monomorphizes to the concrete precision.
//!
//! Per the project backend-numerics discipline these tests do not stop
//! at "it builds": each one compiles the emitted C, runs the binary, and
//! checks the printed gradient against the analytic value. For
//! `sum(lin_p(x, w))` = `sum(x * w)` the analytic gradient w.r.t. `x` is
//! `w`, and w.r.t. `w` is `x`.

use assert_cmd::Command;
use serde_json::Value;
use std::path::Path;
use std::process::Command as StdCommand;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;
use common::{build_and_run, parse_tensor_data, write_file};

fn assert_grad_matches(actual: &[f64], expected: &[f64], tol: f64, msg: &str) {
    assert_eq!(
        actual.len(),
        expected.len(),
        "{msg}: expected length {} got {}; actual={actual:?}",
        expected.len(),
        actual.len(),
    );
    for (i, (a, e)) in actual.iter().zip(expected.iter()).enumerate() {
        let diff = (a - e).abs();
        assert!(
            diff <= tol,
            "{msg}: element {i} mismatch: actual={a} expected={e} diff={diff} tol={tol}",
        );
    }
}

fn run_build(path: &Path) -> std::process::Output {
    let out_dir = path.parent().expect("source path has parent");
    let bin = assert_cmd::cargo::cargo_bin("chelis");
    StdCommand::new(bin)
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            out_dir.to_str().unwrap(),
        ])
        .output()
        .expect("spawn chelis")
}

fn run_check(path: &Path) -> (Value, std::process::Output) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    let json = serde_json::from_slice(&output.stdout).expect("check output should be json");
    (json, output)
}

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

// The headline reproducer text, parameterized over the differentiated
// argument we project out of the grad tuple. `loss(x, w) = sum(x * w)`,
// so `d/dx = w` and `d/dw = x`.
//
// `lin_p` is precision-polymorphic via the canonical def-level explicit
// quantifier `[p]` (spec/02-surf-syntax.md §P4b; matching the stdlib
// `def assert_close_tensor[n, p](...)` shape). The concrete dimension `2`
// stays concrete; only the precision slot is the type variable `p`. The
// differentiated entry point `loss` is fully concrete `f32`.
//
// The compiled-and-run numeric cases use single-`wrt` grad
// (`grad(loss, wrt=target)(x, w)`), which lowers to a single gradient
// tensor — NOT a tuple. The build/check cases below exercise the full
// two-output `grad(loss)(x, w)` form. This split keeps the numeric
// assertions off the unrelated `chelis_tuple_get` C-codegen path for
// projecting a grad tuple, which is its own (non-#289) backend issue;
// the precision-monomorphization behavior under test here is identical
// for the single-output and tuple forms.
fn reproducer_source_wrt(wrt_target: &str) -> String {
    format!(
        "def lin_p[p: Numeric](x: tensor[2, p], w: tensor[2, p]) -> tensor[2, p] = mul(x, w)\n\
         def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 =\n\
           tensor_to_scalar(sum(lin_p(x, w), cast(0, i32)))\n\
         def dloss(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] =\n\
           grad(loss, wrt={wrt_target})(x, w)\n\
         out = dloss(to_tensor([3.0, 4.0]), to_tensor([5.0, 6.0]))\n",
    )
}

fn reproducer_source(proj: &str) -> String {
    format!(
        "def lin_p[p: Numeric](x: tensor[2, p], w: tensor[2, p]) -> tensor[2, p] = mul(x, w)\n\
         def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 =\n\
           tensor_to_scalar(sum(lin_p(x, w), cast(0, i32)))\n\
         def dloss(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] =\n\
           (grad(loss)(x, w)).{proj}\n\
         out = dloss(to_tensor([3.0, 4.0]), to_tensor([5.0, 6.0]))\n",
    )
}

// =====================================================================
// Positive: the reproducer builds AND differentiates correctly through
// the precision-polymorphic callee. d/dx sum(x*w) = w, d/dw = x.
// =====================================================================

#[test]
fn issue_289_grad_through_precision_var_callee_dx_equals_w() {
    // out = d/dx with x=[3,4], w=[5,6]; analytic grad = w = [5, 6].
    let stdout = build_and_run(&reproducer_source_wrt("x"), "grad_pvar_dx");
    let actual = parse_tensor_data(&stdout, "out");
    assert_grad_matches(&actual, &[5.0, 6.0], 1e-5, "issue #289 d/dx = w");
}

#[test]
fn issue_289_grad_through_precision_var_callee_dw_equals_x() {
    // out = d/dw with x=[3,4], w=[5,6]; analytic grad = x = [3, 4].
    let stdout = build_and_run(&reproducer_source_wrt("w"), "grad_pvar_dw");
    let actual = parse_tensor_data(&stdout, "out");
    assert_grad_matches(&actual, &[3.0, 4.0], 1e-5, "issue #289 d/dw = x");
}

#[test]
fn issue_289_reproducer_check_is_clean() {
    // The differentiated entry point `loss` is fully concrete `f32`, so
    // `grad(loss)` has a concrete scalar-float output and the whole
    // program must type-check cleanly. (The issue's "grad requires a
    // scalar floating output, got <error>" check error was a downstream
    // symptom of the precision var not concretizing.) Exercises the full
    // two-output `grad(loss)(x, w).0` tuple form.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("repro_check.ch");
    write_file(&path, &reproducer_source("0"));
    let (json, _) = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "issue #289: precision-polymorphic grad callee must check cleanly; got {errs:?}",
    );
}

#[test]
fn issue_289_reproducer_build_does_not_surface_monomorphization_tripwire() {
    // Exercises the full two-output `grad(loss)(x, w).0` tuple form: the
    // build must succeed and must NOT trip the §5.8.1 tripwire. (This is
    // a build-only assertion; it does not run the produced binary, so the
    // unrelated grad-tuple `chelis_tuple_get` runtime issue does not
    // affect it.)
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("repro_build.ch");
    write_file(&path, &reproducer_source("0"));
    let output = run_build(&path);
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        output.status.success(),
        "issue #289: reproducer build must succeed; output={combined}",
    );
    assert!(
        !combined.contains("monomorphization missed precision var"),
        "issue #289: build must not trip the §5.8.1 monomorphization \
         tripwire for a concrete-f32 grad call site; output={combined}",
    );
}

// =====================================================================
// Control: the inline-f32 reimplementation (no precision-var callee)
// already differentiates; it must keep producing the same gradient. If
// this regresses, the fix broke the ordinary grad path rather than the
// polymorphic one.
// =====================================================================

#[test]
fn issue_289_control_inline_f32_callee_dx_equals_w() {
    let source = "def lin_f32(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] = mul(x, w)\n\
         def loss(x: tensor[2, f32], w: tensor[2, f32]) -> f32 =\n\
           tensor_to_scalar(sum(lin_f32(x, w), cast(0, i32)))\n\
         def dloss(x: tensor[2, f32], w: tensor[2, f32]) -> tensor[2, f32] =\n\
           grad(loss, wrt=x)(x, w)\n\
         out = dloss(to_tensor([3.0, 4.0]), to_tensor([5.0, 6.0]))\n";
    let stdout = build_and_run(source, "grad_inline_f32_control");
    let actual = parse_tensor_data(&stdout, "out");
    assert_grad_matches(&actual, &[5.0, 6.0], 1e-5, "issue #289 control d/dx = w");
}

// =====================================================================
// Negative parity: a genuinely under-determined precision — a
// polymorphic callee with NO concrete call site supplying its
// precision — must still surface a CLEAR diagnostic, not silently
// succeed and not panic. Here `loss` itself is precision-polymorphic
// (`tensor[2, p]`), so differentiating it has no concrete precision to
// monomorphize against. The fix concretizes only when a concrete
// call-site precision exists, so this must still be rejected.
// =====================================================================

#[test]
fn issue_289_negative_grad_over_polymorphic_loss_with_no_concrete_site_is_rejected() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("grad_poly_no_site.ch");
    // `loss` has a precision-variable scalar output and `dloss` is
    // declared polymorphic too, with no concrete `out =` call site
    // pinning `p`. There is no concrete precision to monomorphize the
    // grad transform against, so this must be rejected with a clear
    // diagnostic rather than panicking or emitting silently-wrong code.
    write_file(
        &path,
        "def lin_p[p: Numeric](x: tensor[2, p], w: tensor[2, p]) -> tensor[2, p] = mul(x, w)\n\
         def loss[p: Float](x: tensor[2, p], w: tensor[2, p]) -> tensor[p] =\n\
           sum(lin_p(x, w), cast(0, i32))\n\
         def dloss[p: Float](x: tensor[2, p], w: tensor[2, p]) -> tensor[2, p] =\n\
           (grad(loss)(x, w)).0\n",
    );
    let output = run_build(&path);
    let combined = format!(
        "{}\n{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr),
    );
    assert!(
        !output.status.success(),
        "issue #289 negative parity: grad over a polymorphic loss with no \
         concrete call site must be rejected, not silently built; output={combined}",
    );
    // The rejection must be a clean diagnostic, never the raw internal
    // panic string. (`build` catches lowering panics and reports them,
    // but the negative case here should be caught earlier / cleanly.)
    assert!(
        !combined.contains("BUG: monomorphization missed precision var"),
        "issue #289 negative parity: under-determined precision must yield a \
         clear diagnostic, not the internal monomorphization-BUG panic; \
         output={combined}",
    );
}
