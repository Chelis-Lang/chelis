//! chelis#2178: `grad` over a CHECKED float-to-integer / float-to-bool
//! `cast` must reject structurally, never contribute a silent zero.
//!
//! [04-NUM-14] (`spec/04-type-system.md`) states the rule without
//! qualification:
//!
//! > A float source cast to an integer or bool target is piecewise
//! > constant and structurally rejects `grad` with
//! > `AdRejectionReason::PiecewiseConstant`; it never contributes a
//! > silent zero. A bool or integer source is a discrete forward-only
//! > value and carries no cotangent, irrespective of target.
//!
//! Before the fix, `cast(cast(u, i64), f32)` under `grad` evaluated to
//! `out = 0.0` on `main` and on the 0.18.10 release, while the sibling
//! `cast_trunc` in the same position rejected cleanly -- the failure
//! channel existed and was wired; the checked form simply never reached
//! it.
//!
//! This file pins BOTH directions of the atom:
//!
//! * the rejection fires for every float -> integer width and for
//!   float -> bool, in the surface the issue measured (`chelis eval`);
//! * it does NOT fire for a float-to-float cast (which keeps
//!   [04-NUM-14]'s exact backward cast), for a discrete SOURCE cast
//!   (which carries no cotangent to suppress), or for a discrete cast
//!   that is off the gradient path entirely.
//!
//! Lane scope: the eval lane only, which is the lane the issue measured.
//! `chelis build --target c` declines this program shape for an
//! unrelated reason (`body applies/binds grad in a position the host
//! lane can't resolve`), so there is no compiled-lane row to compare
//! against. The rejection itself is lane-independent: it is raised by
//! `chelis_ir::grad::grad_dag_checked`, which every `grad` entry point
//! in the workspace routes through, and `chelis-ir`'s own unit tests
//! (`grad_cast_float_to_integer_rejects` and siblings) pin it at that
//! layer.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// Evaluate a whole program; first stdout line on success, stderr on
/// failure.
fn eval_program(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    if !out.status.success() {
        return Err(String::from_utf8_lossy(&out.stderr).into_owned());
    }
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

/// `grad(f)(seed)` where `f` round-trips its argument through `target`.
fn grad_through_cast_program(target: &str, seed: &str) -> String {
    format!(
        "module M.Main\n\
         def f(x: f32) -> f32 = cast(cast(x, {target}), f32)\n\
         out = print(grad(f)({seed}))\n"
    )
}

fn assert_piecewise_constant_rejection(stderr: &str, target: &str) {
    assert!(
        stderr.contains("cast"),
        "the rejection must name the offending op for {target}: {stderr}"
    );
    assert!(
        stderr.contains("non-differentiable") || stderr.contains("piecewise constant"),
        "the rejection must state WHY (piecewise constant) for {target}: {stderr}"
    );
}

// ===========================================================================
// The rejection fires (the defect chelis#2178 reported)
// ===========================================================================

/// The issue's reproducer, widened to every integer target width.
/// `2.0` is in-domain for the checked cast, so the forward direction
/// does not trap: the ONLY thing under test is the adjoint.
#[test]
fn grad_over_a_checked_float_to_integer_cast_rejects() {
    for target in ["i8", "i16", "i32", "i64"] {
        let stderr = eval_program(&grad_through_cast_program(target, "2.0"))
            .unwrap_or_else(|e| e)
            .to_string();
        assert!(
            !stderr.is_empty(),
            "[04-NUM-14]: f32 -> {target} under grad must not produce a value at all"
        );
        assert_piecewise_constant_rejection(&stderr, target);
    }
}

/// The bool target is measured separately: `Prim::Bool` is not an
/// integer dtype, so a rejection keyed only on `is_integer()` would miss
/// it. The seed is `1.0`, which [04-NUM-4] admits, so the forward cast
/// is in-domain and the adjoint is again the only thing under test.
#[test]
fn grad_over_a_checked_float_to_bool_cast_rejects() {
    let stderr = eval_program(&grad_through_cast_program("bool", "1.0"))
        .unwrap_or_else(|e| e)
        .to_string();
    assert!(
        !stderr.is_empty(),
        "[04-NUM-14]: f32 -> bool under grad must not produce a value at all"
    );
    assert_piecewise_constant_rejection(&stderr, "bool");
}

/// The exact silent value the issue measured. Kept as its own assertion
/// because "the program errored" and "the program did not print 0.0" are
/// different claims, and only the second one falsifies the defect.
#[test]
fn grad_over_a_checked_float_to_integer_cast_never_prints_a_zero() {
    let out = eval_program(&grad_through_cast_program("i64", "2.0"));
    assert!(
        out.is_err(),
        "[04-NUM-14] forbids the silent zero; got out = {:?}",
        out.unwrap()
    );
}

/// The sibling operation, unchanged: `cast_trunc` already rejected
/// through the same channel ([05-OP-6]). This is the control that shows
/// the two rungs now agree on the gradient disposition.
#[test]
fn cast_trunc_control_still_rejects_through_the_same_channel() {
    let program = "module M.Main\n\
                   def f(x: f32) -> f32 = cast(cast_trunc(x, i64), f32)\n\
                   out = print(grad(f)(2.0))\n";
    let stderr = eval_program(program).expect_err("[05-OP-6] carries the no_grad rule");
    assert_piecewise_constant_rejection(&stderr, "cast_trunc");
}

// ===========================================================================
// NEGATIVE PARITY: the rejection must not over-fire
// ===========================================================================

/// [04-NUM-14]: "a float-to-float cast has adjoint `cast(g,
/// source_dtype)`". `f(x) = cast(cast(x, f64), f32) ** 2` has derivative
/// `2x`, so `x = 3.0` must still yield exactly `6.0` -- not a rejection,
/// and not a zero.
#[test]
fn grad_over_a_float_to_float_cast_still_computes_the_backward_cast() {
    let program = "module M.Main\n\
                   def f(x: f32) -> f32 = mul(cast(cast(x, f64), f32), cast(cast(x, f64), f32))\n\
                   out = print(grad(f)(3.0))\n";
    let line = eval_program(program)
        .expect("[04-NUM-14]: a float-to-float cast keeps its exact backward cast");
    assert_eq!(
        line, "6.0",
        "d(x*x)/dx at x=3 must be 6.0 through a float-to-float cast round trip; got {line}"
    );
}

/// [04-NUM-14]: "A bool or integer source is a discrete forward-only
/// value and carries no cotangent, irrespective of target." That is the
/// ABSENCE of a gradient variable, not a suppressed one, so an
/// integer-to-float cast on the gradient path must still differentiate.
/// `grad_zero_placeholder_matrix::grad_without_abs_is_correct_in_both_lanes`
/// depends on this row staying green.
#[test]
fn grad_over_an_integer_source_cast_is_accepted() {
    let program = "def g(x: tensor[4, f32]) -> tensor[f32] = {\n\
           w = cast(to_tensor([cast(-100, i64), cast(200, i64), \
         cast(-300, i64), cast(400, i64)]), f32)\n\
           sum(mul(copy(x), w), 0)\n\
         }\n\
         def compute_grad(x: tensor[4, f32]) -> tensor[4, f32] = grad(g, wrt=x)(x)\n\
         out = print(compute_grad(to_tensor([0.1, 0.2, 0.3, 0.4])))\n";
    let line = eval_program(program)
        .expect("an integer-source cast under grad must not reject ([04-NUM-14])");
    assert!(
        line.contains("data=[-100.0, 200.0, -300.0, 400.0]"),
        "grad of sum(x*w) wrt x = w; got: {line}"
    );
}

/// The rejection is scoped to the GRADIENT PATH by `grad_dag_checked`'s
/// live-node scan. A float-to-integer cast that the differentiated
/// output does not reach is not on that path and must not reject --
/// otherwise the fix would turn unrelated discrete arithmetic in the
/// same function body into a grad failure.
#[test]
fn a_float_to_integer_cast_off_the_gradient_path_does_not_reject() {
    let program = "module M.Main\n\
                   def f(x: f32) -> f32 = {\n\
                     unused = cast(4.0, i64)\n\
                     mul(x, x)\n\
                   }\n\
                   out = print(grad(f)(3.0))\n";
    let line =
        eval_program(program).expect("a discrete cast off the gradient path must not reject grad");
    assert_eq!(
        line, "6.0",
        "d(x*x)/dx at x=3 must be 6.0 with a dead discrete cast present; got {line}"
    );
}

/// The FORWARD direction is untouched: a checked float-to-integer cast
/// outside any `grad` still evaluates under [04-NUM-14]'s exact
/// finalization rule. This separates "the adjoint now rejects" from "the
/// operation now fails", which would be a much larger change.
#[test]
fn the_forward_checked_cast_is_unaffected() {
    let program = "module M.Main\n\
                   out = print(cast(cast(2.0, i64), f32))\n";
    let line = eval_program(program).expect("the forward checked cast must still evaluate");
    assert_eq!(
        line, "2.0",
        "[04-NUM-14]'s forward cast is unchanged by the AD rejection; got {line}"
    );
}
