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
//! Lane scope: BOTH lanes. The issue recorded that
//! `chelis build --target c` declined this program shape for an
//! unrelated reason (`body applies/binds grad in a position the host
//! lane can't resolve`), and an earlier revision of this file repeated
//! that as "there is no compiled-lane row to compare against". That was
//! true of the base and is no longer true here: the rejection is raised
//! in `grad_dag_checked`, which runs BEFORE the host-lane resolution
//! check, so the C lane now reports the same structural rejection. It is
//! pinned below by `the_compiled_c_lane_reports_the_same_rejection`.

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
    grad_through_cast_program_from("f32", target, seed)
}

/// The same, with an explicit float SOURCE dtype. [04-NUM-14] scopes the
/// rejection by the source being a float, not by it being `f32`, and
/// `is_piecewise_constant_cast` keys on `Prim::is_float()` -- so the
/// narrow dtypes belong in the matrix too.
fn grad_through_cast_program_from(source: &str, target: &str, seed: &str) -> String {
    format!(
        "module M.Main\n\
         def f(x: {source}) -> {source} = cast(cast(x, {target}), {source})\n\
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
/// live-node scan, which treats an indexed operation's INDEX input as a
/// stop-gradient boundary: index math is not data and carries no
/// cotangent. A float-to-integer cast feeding `gather`'s indices is
/// therefore not on the gradient path and must not reject -- otherwise
/// this fix would break every embedding-style lookup whose indices are
/// computed from float data.
///
/// This replaces an earlier dead-binding version of this test, which was
/// vacuous: an unused `let` never reaches the lowered DAG at all, so it
/// could not distinguish liveness from anything else -- a dead
/// `cast_trunc`, which always rejects when live, did not reject there
/// either. The cast below IS in the lowered program.
#[test]
fn a_float_to_integer_cast_on_an_index_edge_does_not_reject() {
    let program = "module M.Main\n\
                   def f(table: tensor[4, f32], raw: tensor[2, f32]) -> f32 = \
                   tensor_to_scalar(sum(gather(table, cast(raw, i64), 0i32), 0i32))\n\
                   out = print(grad(f, wrt=table)(to_tensor([1.0f32, 2.0f32, 3.0f32, 4.0f32]), \
                   to_tensor([1.0f32, 3.0f32])))\n";
    let line =
        eval_program(program).expect("a discrete cast on an index edge must not reject grad");
    assert_eq!(
        line, "tensor(shape=[4], data=[0.0, 1.0, 0.0, 1.0])",
        "gather's adjoint must still scatter the cotangent to the gathered rows; got {line}"
    );
}

/// [04-NUM-14] scopes the rejection by the source being a FLOAT, not by
/// it being `f32`. Every active float dtype is therefore a source, and
/// the narrow ones are where a silent zero would do the most damage.
/// `is_piecewise_constant_cast` keys on `Prim::is_float()`, whose
/// membership is itself pinned to §1.1's active set by `chelis-types`,
/// so this loop and that predicate close the dimension together.
#[test]
fn every_active_float_source_rejects_not_just_f32() {
    for source in ["f16", "bf16", "f32", "f64"] {
        let program = grad_through_cast_program_from(source, "i32", &format!("2.0{source}"));
        let stderr = eval_program(&program).unwrap_or_else(|e| e).to_string();
        assert!(
            !stderr.is_empty(),
            "[04-NUM-14]: {source} -> i32 under grad must not produce a value at all"
        );
        assert_piecewise_constant_rejection(&stderr, source);
    }
}

/// The compiled lane reports the SAME structural rejection.
///
/// The issue recorded that `chelis build --target c` declined this shape
/// for an unrelated reason, and an earlier revision of this file claimed
/// there was therefore no compiled-lane row to be had. That was true of
/// the base: the host lane's "can't resolve `grad` in this position"
/// check fired first. It is not true here, because `grad_dag_checked`
/// runs BEFORE that check, so the rejection now reaches the C lane. The
/// build must fail for the RIGHT reason, not the incidental one.
#[test]
fn the_compiled_c_lane_reports_the_same_rejection() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    write_file(
        &path,
        "module M.Main\n\
         def f(x: f32) -> f32 = cast(cast(x, i64), f32)\n\
         def d(y: f32) -> f32 = grad(f)(y)\n\
         out = d(2.0f32)\n",
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("out").to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    assert!(!out.status.success(), "the C lane must reject this program");
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    assert_piecewise_constant_rejection(&stderr, "c-lane");
    assert!(
        !stderr.contains("can't lower these defs"),
        "the C lane must fail on the AD rejection, not the incidental host-lane \
         resolution error that masked it on the base: {stderr}"
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
