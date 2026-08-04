//! chelis#860 - the post-desugar operand-dtype chokepoint, plus the
//! decided chelis#724 (integer `mean`) and chelis#726 (bool arithmetic)
//! checker rejections it enforces.
//!
//! The class under test: the float-only / capability acceptance checks
//! used to run only inside `infer_app`'s post-unify block, so any
//! application form that never builds an `app` node bypassed them -
//! concretely the bare-callee pipe stage (`cast(2, int32) |> recip`),
//! which reached the runtime while the direct spelling was rejected.
//! The fix routes every application surface (direct application, pipe
//! stages, reduction first-argument checks, and the polymorphic
//! cross-row pass) through one policy function,
//! `chelis_types::infer::operand_dtype_rejection`.
//!
//! Every negative row asserts a CHECK-TIME rejection: stderr carries
//! the capability citation and never the runtime `numeric trap` brand
//! (the runtime traps stay as defense in depth behind the checker).

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// Eval a whole program; first stdout line on success, stderr on failure.
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

/// The program must be rejected at CHECK time with a diagnostic carrying
/// `citation`; a runtime trap does not satisfy this (that would mean the
/// checker hole is still open and only defense-in-depth fired).
fn rejection_stderr(program: &str, label: &str) -> String {
    match eval_program(program) {
        Ok(v) => panic!(
            "{label}: program must be rejected by the checker, but it \
             evaluated and returned {v}. Program:\n{program}"
        ),
        Err(stderr) => stderr,
    }
}

fn assert_check_rejects(program: &str, citation: &str, label: &str) {
    let stderr = rejection_stderr(program, label);
    assert!(
        stderr.contains(citation),
        "{label}: rejection must cite {citation}; got: {stderr}"
    );
    assert!(
        !stderr.contains("numeric trap"),
        "{label}: the rejection must be CHECK-TIME, not a runtime \
         trap reached through a checker hole (chelis#860); got: {stderr}"
    );
}

const INT64_TENSOR: &str =
    "to_tensor([cast(100, int64), cast(400, int64), cast(200, int64), cast(50, int64)])";

// ===========================================================================
// chelis#724 - integer mean is a check-time rejection in every form.
// ===========================================================================

#[test]
fn integer_mean_direct_form_rejected() {
    assert_check_rejects(
        &format!("module M.Main\nout = print(mean({INT64_TENSOR}, 0))\n"),
        "chelis#724",
        "mean_direct",
    );
}

#[test]
fn integer_mean_def_body_form_rejected() {
    assert_check_rejects(
        &format!(
            "module M.Main\n\
             def f(x: tensor[4, int64]) -> tensor[int64] = mean(x, 0)\n\
             out = print(f({INT64_TENSOR}))\n"
        ),
        "chelis#724",
        "mean_def_body",
    );
}

#[test]
fn integer_mean_polymorphic_wrapper_rejected_at_instantiation() {
    assert_check_rejects(
        &format!(
            "module M.Main\n\
             sig my_mean: tensor[4, p] -> tensor[p]\n\
             def my_mean(x) = mean(x, 0)\n\
             def call(y: tensor[4, int64]) -> tensor[int64] = my_mean(y)\n\
             out = print(call({INT64_TENSOR}))\n"
        ),
        "chelis#724",
        "mean_poly_wrapper",
    );
}

#[test]
fn integer_mean_direct_and_polymorphic_paths_share_the_canonical_diagnostic() {
    const BASE: &str = "mean on operand precision `int64` is not admitted per the chelis#724 \
capability decision: mean is float-only (f32, f64, bf16, f16). An integer mean has no \
authored rounding, and a fractional result inside an integer tensor violates \
spec/04-type-system.md section 9 [04-NUM-1]";
    let direct = rejection_stderr(
        &format!("module M.Main\nout = print(mean({INT64_TENSOR}, 0))\n"),
        "mean_direct_canonical",
    );
    let polymorphic = rejection_stderr(
        &format!(
            "module M.Main\n\
             sig my_mean: tensor[4, p] -> tensor[p]\n\
             def my_mean(x) = mean(x, 0)\n\
             def call(y: tensor[4, int64]) -> tensor[int64] = my_mean(y)\n\
             out = print(call({INT64_TENSOR}))\n"
        ),
        "mean_poly_canonical",
    );
    assert!(direct.contains(BASE), "direct diagnostic drifted: {direct}");
    assert!(
        polymorphic.contains(BASE),
        "polymorphic diagnostic must reuse the direct base: {polymorphic}"
    );
    assert!(
        polymorphic.contains("Reached through the polymorphic sig for `my_mean`"),
        "polymorphic context must remain available as a hint: {polymorphic}"
    );
}

#[test]
fn bool_mean_rejected_too() {
    // mean is float-only: bool is rejected alongside the integers.
    assert_check_rejects(
        "module M.Main\nout = print(mean(to_tensor([true, false]), 0))\n",
        "chelis#724",
        "mean_bool",
    );
}

/// Positive parity: float mean still checks and computes.
#[test]
fn float_mean_still_computes() {
    let got = eval_program(
        "module M.Main\nout = print(mean(cast(to_tensor([1.0, 2.0, 3.0, 4.0]), f32), 0))\n",
    )
    .expect("float mean");
    assert_eq!(got, "2.5");
}

/// Positive parity: the rejection's own hint (cast first) works.
#[test]
fn cast_then_mean_idiom_computes() {
    let got = eval_program(&format!(
        "module M.Main\nout = print(mean(cast({INT64_TENSOR}, f32), 0))\n"
    ))
    .expect("cast-then-mean");
    assert_eq!(got, "187.5");
}

// ===========================================================================
// chelis#726 - bool arithmetic is a check-time rejection in every form.
// ===========================================================================

#[test]
fn bool_add_direct_form_rejected() {
    assert_check_rejects(
        "module M.Main\n\
         out = print(add(to_tensor([true, false]), to_tensor([true, true])))\n",
        "chelis#726",
        "bool_add_direct",
    );
}

#[test]
fn bool_sub_mul_neg_and_floor_div_rejected() {
    for op in ["sub", "mul"] {
        assert_check_rejects(
            &format!(
                "module M.Main\n\
                 out = print({op}(to_tensor([true, false]), to_tensor([true, true])))\n"
            ),
            "chelis#726",
            &format!("bool_{op}_direct"),
        );
    }
    assert_check_rejects(
        "module M.Main\nout = print(neg(to_tensor([true, false])))\n",
        "chelis#726",
        "bool_neg_direct",
    );
    assert_check_rejects(
        "module M.Main\n\
         out = print(floor_div(to_tensor([true, false]), to_tensor([true, true])))\n",
        "chelis#726",
        "bool_floor_div_direct",
    );
}

#[test]
fn bool_sum_is_not_reclassified_as_a_726_arithmetic_cell() {
    let stderr = rejection_stderr(
        "module M.Main\nout = print(sum(to_tensor([true, false, true]), 0))\n",
        "bool_sum_existing_rejection",
    );
    assert!(
        stderr.contains("reduce_sum is not defined on bool tensors"),
        "the pre-existing reduction disposition should remain explicit: {stderr}"
    );
    assert!(
        !stderr.contains("chelis#726"),
        "#726 did not authorize widening its arithmetic roster to bool sum: {stderr}"
    );
}

#[test]
fn bool_add_def_body_form_rejected() {
    assert_check_rejects(
        "module M.Main\n\
         def f(x: tensor[2, bool], y: tensor[2, bool]) -> tensor[2, bool] = add(x, y)\n\
         out = print(f(to_tensor([true, false]), to_tensor([true, true])))\n",
        "chelis#726",
        "bool_add_def_body",
    );
}

#[test]
fn bool_add_polymorphic_wrapper_rejected_at_instantiation() {
    assert_check_rejects(
        "module M.Main\n\
         sig both: tensor[2, p] -> tensor[2, p] -> tensor[2, p]\n\
         def both(x, y) = add(x, y)\n\
         def call(a: tensor[2, bool], b: tensor[2, bool]) -> tensor[2, bool] = both(a, b)\n\
         out = print(call(to_tensor([true, false]), to_tensor([true, true])))\n",
        "chelis#726",
        "bool_add_poly_wrapper",
    );
}

#[test]
fn bool_add_direct_and_polymorphic_paths_share_the_canonical_diagnostic() {
    const BASE: &str = "add on bool operands is not admitted per the chelis#726 capability \
decision and spec/04-type-system.md section 9 [04-NUM-4]: bool is exactly {0, 1} and not \
a numeric dtype, so arithmetic on it has no authored meaning";
    let direct = rejection_stderr(
        "module M.Main\n\
         out = print(add(to_tensor([true, false]), to_tensor([true, true])))\n",
        "bool_add_direct_canonical",
    );
    let polymorphic = rejection_stderr(
        "module M.Main\n\
         sig both: tensor[2, p] -> tensor[2, p] -> tensor[2, p]\n\
         def both(x, y) = add(x, y)\n\
         def call(a: tensor[2, bool], b: tensor[2, bool]) -> tensor[2, bool] = both(a, b)\n\
         out = print(call(to_tensor([true, false]), to_tensor([true, true])))\n",
        "bool_add_poly_canonical",
    );
    assert!(direct.contains(BASE), "direct diagnostic drifted: {direct}");
    assert!(
        polymorphic.contains(BASE),
        "polymorphic diagnostic must reuse the direct base: {polymorphic}"
    );
    assert!(
        polymorphic.contains("Reached through the polymorphic sig for `both`"),
        "polymorphic context must remain available as a hint: {polymorphic}"
    );
}

#[test]
fn bool_neg_through_bare_pipe_stage_rejected() {
    // The chelis#860 mechanism applied to the chelis#726 cell: a bare
    // pipe stage must take the same rejection as the direct form.
    assert_check_rejects(
        "module M.Main\nout = print(to_tensor([true, false]) |> neg)\n",
        "chelis#726",
        "bool_neg_pipe",
    );
}

#[test]
fn bool_add_through_canonical_pipe_call_stage_rejected() {
    assert_check_rejects(
        "module M.Main\n\
         out = print(to_tensor([true, false]) |> add(to_tensor([true, false])))\n",
        "chelis#726",
        "bool_add_pipe_call_stage",
    );
}

/// Positive parity: bool logic and the counting idiom survive the
/// rejection (the exact surfaces the diagnostics point at).
#[test]
fn bool_logic_and_counting_idiom_still_compute() {
    let got = eval_program(
        "module M.Main\n\
         out = print(and(to_tensor([true, false]), to_tensor([true, true])))\n",
    )
    .expect("and on bool");
    assert_eq!(got, "tensor(shape=[2], data=[true, false])");

    let got = eval_program(
        "module M.Main\n\
         out = print(sum(cast(to_tensor([true, false, true]), int64), 0))\n",
    )
    .expect("counting idiom");
    assert_eq!(got, "2");
}

/// Positive parity: integer and float arithmetic are untouched.
#[test]
fn numeric_arithmetic_still_computes() {
    assert_eq!(
        eval_program("module M.Main\nout = print(add(cast(2, int32), cast(3, int32)))\n")
            .expect("int add"),
        "5"
    );
    assert_eq!(
        eval_program(
            "module M.Main\n\
             out = print(sum(to_tensor([cast(1, int64), cast(2, int64)]), 0))\n"
        )
        .expect("int sum"),
        "3"
    );
}

// ===========================================================================
// chelis#860 - the pipe bypass itself: float-only ops on integers reject
// at CHECK time through the bare-stage form, matching the direct form.
// ===========================================================================

#[test]
fn int_recip_through_bare_pipe_stage_rejected_at_check_time() {
    // Scalar transcendental rejections use the generic acceptance
    // message (the same one the direct form `recip(cast(2, int8))`
    // produces); the section 5.4 citation is the concrete-tensor form's.
    for (width, label) in [("int8", "i8"), ("int32", "i32"), ("int64", "i64")] {
        assert_check_rejects(
            &format!("module M.Main\nout = print(cast(2, {width}) |> recip)\n"),
            "does not accept argument type",
            &format!("pipe_recip_{label}"),
        );
    }
}

#[test]
fn int_exp_through_bare_pipe_stage_rejected_at_check_time() {
    assert_check_rejects(
        "module M.Main\nout = print(to_tensor([cast(2, int32)]) |> exp)\n",
        "5.4",
        "pipe_exp_int_tensor",
    );
}

#[test]
fn multi_stage_pipe_rejects_at_the_offending_stage() {
    // First stage is fine (float recip); second stage pipes the float
    // through a cast to int32 via a lambda, then the THIRD bare stage
    // must reject.
    assert_check_rejects(
        "module M.Main\n\
         out = print(cast(2.0, f32) |> recip |> fn (v) -> cast(floor(v), int32) |> recip)\n",
        "does not accept argument type",
        "pipe_multi_stage",
    );
}

/// Positive parity: pipes over admissible dtypes are untouched.
#[test]
fn float_pipe_stages_still_compute() {
    assert_eq!(
        eval_program("module M.Main\nout = print(cast(2.0, f32) |> recip)\n")
            .expect("float pipe recip"),
        "0.5"
    );
    assert_eq!(
        eval_program("module M.Main\nout = print(to_tensor([4.0]) |> sqrt)\n")
            .expect("float pipe sqrt"),
        "tensor(shape=[1], data=[2.0])"
    );
}
