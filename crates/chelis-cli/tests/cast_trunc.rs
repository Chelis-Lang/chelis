//! [05-OP-6] `cast_trunc` acceptance matrix
//! (`spec/05-risc-primitives.md` §3.8; `spec/design/archive/named_truncating_cast.md`
//! §8; the chelis#759 ladder's float-to-integer rung, which unblocks the
//! chelis#1091 ecosystem break).
//!
//! The contract under test:
//!
//! * a FINITE source truncates toward zero, finalized at the target width;
//! * a truncated integer outside the target range TRAPS `overflow`
//!   (never wraps, never saturates);
//! * `NaN` / `+-inf` TRAP `domain`;
//! * scalar and tensor surfaces agree, and the eval and compiled C lanes
//!   produce byte-identical output over {f32,f64} -> {i32,i64};
//! * a non-float source, a non-integer target, and a gradient through it
//!   are each a clean typed error.
//!
//! The checked default (`cast`, [04-NUM-14]) is unchanged and is pinned
//! by `issue_759_checked_cast_default.rs`.

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

/// Evaluate a printed expression.
fn eval_expr(expr: &str) -> Result<String, String> {
    eval_program(&format!("module M.Main\nout = print({expr})\n"))
}

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build, link, and run a program through the C lane; `(stdout, stderr, ok)`.
fn c_lane_run(program: &str, name: &str) -> Result<(String, String, bool), String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
    let built = Command::cargo_bin("chelis")
        .expect("binary")
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
        .expect("chelis build should run");
    if !built.status.success() {
        return Err(String::from_utf8_lossy(&built.stderr).into_owned());
    }
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    Ok((
        String::from_utf8_lossy(&run.stdout).into_owned(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
        run.status.success(),
    ))
}

// ===========================================================================
// Truncation toward zero: scalar and tensor
// ===========================================================================

#[test]
fn scalar_fractional_values_truncate_toward_zero() {
    for (expr, expected) in [
        ("cast_trunc(1.9, i32)", "1"),
        ("cast_trunc(-1.9, i32)", "-1"),
        ("cast_trunc(0.9, i32)", "0"),
        ("cast_trunc(-0.9, i32)", "0"),
        ("cast_trunc(3.0, i32)", "3"),
        ("cast_trunc(-3.0, i32)", "-3"),
    ] {
        assert_eq!(
            eval_expr(expr).unwrap_or_else(|e| panic!("`{expr}` must evaluate: {e}")),
            expected,
            "`{expr}` truncates toward zero, not toward -inf"
        );
    }
}

#[test]
fn tensor_surface_truncates_elementwise_toward_zero() {
    assert_eq!(
        eval_expr("cast_trunc(to_tensor([1.9, -1.9, 0.5, -0.5, 3.0]), i32)")
            .expect("tensor cast_trunc must evaluate"),
        "tensor(shape=[5], data=[1, -1, 0, 0, 3])"
    );
}

/// The two rungs agree exactly where the checked default is defined; they
/// differ ONLY on the fractional case.
#[test]
fn agrees_with_the_checked_default_on_integral_values() {
    for expr in ["3.0", "-3.0", "0.0"] {
        assert_eq!(
            eval_expr(&format!("cast({expr}, i64)")).expect("checked default"),
            eval_expr(&format!("cast_trunc({expr}, i64)")).expect("truncating rung"),
            "the rungs agree on an already-integral {expr}"
        );
    }
    assert_eq!(
        eval_expr("cast_trunc(1.9, i64)").expect("truncating rung defines the fractional case"),
        "1"
    );
    assert!(
        eval_expr("cast(1.9, i64)").is_err(),
        "the checked default still traps on the fractional case ([04-NUM-14])"
    );
}

/// The checked default's Domain diagnostic must teach the shipped
/// migration target, not call it future work.
#[test]
fn the_checked_default_teaches_cast_trunc_as_the_named_form() {
    let stderr = eval_expr("cast(1.9, i32)").expect_err("the checked default traps");
    assert!(
        stderr.contains("cast_trunc"),
        "the fractional-cast hint must name `cast_trunc` now that it ships: {stderr}"
    );
}

// ===========================================================================
// Traps
// ===========================================================================

#[test]
fn out_of_range_truncated_values_trap_overflow_never_saturate() {
    for (expr, prim) in [
        ("cast_trunc(1e30, i32)", "i32"),
        ("cast_trunc(-1e30, i32)", "i32"),
        ("cast_trunc(300.9, i8)", "i8"),
        ("cast_trunc(to_tensor([300.9]), i8)", "i8"),
    ] {
        let stderr = eval_expr(expr).expect_err("an out-of-range value must trap");
        assert!(
            stderr.contains(&format!("numeric trap: overflow in cast_trunc at {prim}")),
            "`{expr}` must trap with the branded overflow diagnostic, \
             never saturate to the width maximum: {stderr}"
        );
    }
}

#[test]
fn non_finite_sources_trap_domain() {
    for expr in [
        "cast_trunc(sqrt(-1.0), i32)",
        "cast_trunc(div(1.0, 0.0), i32)",
        "cast_trunc(sqrt(to_tensor([-1.0])), i32)",
    ] {
        let stderr = eval_expr(expr).expect_err("a non-finite source must trap");
        assert!(
            stderr.contains("numeric trap: domain in cast_trunc at i32"),
            "`{expr}`: truncation of a non-finite value has no integer meaning \
             and must trap Domain, not Overflow: {stderr}"
        );
    }
}

// ===========================================================================
// Negative parity: the check-time source/target contract
// ===========================================================================

#[test]
fn a_non_float_source_is_a_check_time_type_error() {
    for expr in ["cast_trunc(3, i64)", "cast_trunc(to_tensor([3]), i64)"] {
        let stderr = eval_expr(expr).expect_err("an integer source is not a truncating cast");
        assert!(
            stderr.contains("is not a float dtype") && stderr.contains("cast_trunc"),
            "`{expr}` must be rejected at CHECK time naming the source dtype: {stderr}"
        );
    }
}

#[test]
fn a_non_integer_target_is_a_check_time_type_error() {
    for (expr, target) in [
        ("cast_trunc(1.9, f32)", "f32"),
        ("cast_trunc(1.9, f64)", "f64"),
        ("cast_trunc(1.9, bool)", "bool"),
    ] {
        let stderr = eval_expr(expr).expect_err("only an integer target truncates");
        assert!(
            stderr.contains(&format!(
                "`cast_trunc` target `{target}` is not an integer dtype"
            )),
            "`{expr}` must be rejected at CHECK time naming the target dtype: {stderr}"
        );
    }
}

#[test]
fn a_gradient_through_cast_trunc_is_a_clean_error_not_a_silent_zero() {
    let program = "module M.Main\n\
                   def f(x: f32) -> f32 = cast(cast_trunc(x, i32), f32)\n\
                   out = print(grad(f)(2.5))\n";
    let stderr = eval_program(program)
        .expect_err("[05-OP-6] carries the no_grad rule; a gradient goal must fail");
    assert!(
        stderr.contains("cast_trunc"),
        "the rejection must name the offending op: {stderr}"
    );
    assert!(
        stderr.contains("non-differentiable") || stderr.contains("piecewise constant"),
        "the rejection must state WHY (piecewise constant), not just fail: {stderr}"
    );
    assert!(
        !stderr.is_empty(),
        "a silent zero gradient would mask a modeling bug ([05-OP-6])"
    );
}

// ===========================================================================
// Cross-lane agreement: eval vs compiled C, over {f32,f64} x {i32,i64}
// ===========================================================================

/// The §8 acceptance matrix. Each row is evaluated on BOTH lanes and the
/// stdout compared byte for byte.
const AGREEMENT_MATRIX: &[(&str, &str, &str)] = &[
    // (label, source expression at the stated dtype, target dtype)
    ("f32_pos_frac", "1.9f32", "i32"),
    ("f32_neg_frac", "-1.9f32", "i32"),
    ("f32_pos_small", "0.9f32", "i32"),
    ("f32_neg_small", "-0.9f32", "i32"),
    ("f32_neg_zero", "-0.0f32", "i32"),
    ("f32_integral", "3.0f32", "i32"),
    ("f32_to_i64", "1.9f32", "i64"),
    ("f64_pos_frac", "1.9f64", "i32"),
    ("f64_neg_frac", "-1.9f64", "i32"),
    ("f64_integral", "-3.0f64", "i32"),
    ("f64_to_i64", "1.9f64", "i64"),
    ("f64_neg_to_i64", "-1.9f64", "i64"),
    // Boundary: the largest i32 and its fractional neighbour, which
    // truncates back INTO range rather than overflowing.
    ("f64_i32_max", "2147483647.0f64", "i32"),
    ("f64_i32_max_frac", "2147483647.9f64", "i32"),
    ("f64_i32_min", "-2147483648.0f64", "i32"),
    ("f64_i32_min_frac", "-2147483647.9f64", "i32"),
];

#[test]
fn eval_and_compiled_c_lanes_agree_byte_for_byte() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    for (label, source, target) in AGREEMENT_MATRIX {
        let program = format!(
            "module M.Main\n\
             def f() -> {target} = cast_trunc({source}, {target})\n\
             out = print(f())\n"
        );
        let evaluated = eval_program(&program)
            .unwrap_or_else(|e| panic!("{label}: eval lane must succeed: {e}"));
        let (stdout, stderr, ok) = c_lane_run(&program, &format!("ct_{label}"))
            .unwrap_or_else(|e| panic!("{label}: C lane must build: {e}"));
        assert!(ok, "{label}: compiled binary must succeed; stderr={stderr}");
        assert_eq!(
            stdout.lines().next().unwrap_or("").trim(),
            evaluated,
            "{label}: `cast_trunc({source}, {target})` must be identical on \
             both lanes ([05-OP-6] declares identical eval-vs-compiled \
             behavior at the [04-NUM-8] widths)"
        );
    }
}

#[test]
fn compiled_c_lane_traps_overflow_with_the_same_brand() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
                   def f() -> tensor[1, i8] = cast_trunc(to_tensor([300.9]), i8)\n\
                   out = print(f())\n";
    let (stdout, stderr, ok) = c_lane_run(program, "ct_overflow").expect("C lane");
    assert!(
        !ok && stderr.contains("numeric trap: overflow in cast_trunc at i8"),
        "the compiled lane must raise the SAME branded overflow diagnostic as \
         eval; got ok={ok} stdout={stdout} stderr={stderr}"
    );
    let eval_err = eval_expr("cast_trunc(to_tensor([300.9]), i8)").expect_err("eval traps too");
    assert!(
        eval_err.contains("numeric trap: overflow in cast_trunc at i8"),
        "and eval's brand must match: {eval_err}"
    );
}

#[test]
fn compiled_c_lane_traps_domain_on_non_finite_with_the_same_brand() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "module M.Main\n\
                   def f() -> tensor[1, i32] = cast_trunc(sqrt(to_tensor([-1.0])), i32)\n\
                   out = print(f())\n";
    let (stdout, stderr, ok) = c_lane_run(program, "ct_domain").expect("C lane");
    assert!(
        !ok && stderr.contains("numeric trap: domain in cast_trunc at i32"),
        "the compiled lane must Domain-trap on NaN with the same brand; \
         got ok={ok} stdout={stdout} stderr={stderr}"
    );
}

/// chelis#759 MEDIUM-1: for a tensor carrying BOTH an out-of-range and a
/// non-finite element, both lanes must report the kind belonging to
/// whichever element comes FIRST. Eval used to run a whole-buffer
/// non-finite pre-pass and answer `domain` where C answered `overflow`,
/// which contradicts [05-OP-6]'s identical-lanes clause.
///
/// The i8 rows are the discriminating ones: at i64 both offender
/// kinds are caught in the same pass, so a narrower width is required to
/// see the divergence at all.
#[test]
fn mixed_offender_tensors_agree_on_the_trap_kind_across_lanes() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    // `sqrt` over a tensor produces both offender kinds in one DAG-lane
    // op: a finite out-of-range element and a NaN. Building the buffer
    // from scalar expressions instead (`to_tensor([1e30, sqrt(-1.0)])`)
    // would route it through the HOST tensor lane, which is a different
    // code path and not what MEDIUM-1 is about.
    let cases = [
        ("mix_over_first_i32", "[1e30, -1.0]", "i32", "overflow"),
        ("mix_nan_first_i32", "[-1.0, 1e30]", "i32", "domain"),
        // The discriminating rows: 90000.0 -> sqrt -> 300.0, which fits
        // i64 but not i8. Only a narrow width exposes an out-of-order
        // width check.
        ("mix_over_first_i8", "[90000.0, -1.0]", "i8", "overflow"),
        ("mix_nan_first_i8", "[-1.0, 90000.0]", "i8", "domain"),
    ];
    for (label, elements, target, expected_kind) in cases {
        let expr = format!("cast_trunc(sqrt(to_tensor({elements})), {target})");
        let eval_err = eval_expr(&expr).expect_err("both elements offend; one must trap");
        let branded = format!("numeric trap: {expected_kind} in cast_trunc at {target}");
        assert!(
            eval_err.contains(&branded),
            "{label}: eval must report the FIRST offender's kind (`{branded}`): {eval_err}"
        );

        let program = format!(
            "module M.Main\n\
             def f() -> tensor[2, {target}] = {expr}\n\
             out = print(f())\n"
        );
        let (stdout, stderr, ok) = c_lane_run(&program, label).expect("C lane");
        assert!(
            !ok && stderr.contains(&branded),
            "{label}: the compiled lane must report the SAME kind as eval \
             (`{branded}`); got ok={ok} stdout={stdout} stderr={stderr}"
        );
    }
}

/// The HOST tensor lane (a tensor built from scalar expressions, which
/// carries `*` dims rather than lowering into the DAG) has no
/// `cast_trunc` emission yet, and it must REJECT LOUDLY rather than fall
/// through to an identity pass-through.
///
/// This is pinned deliberately, because the checked `cast`'s host arm
/// ends in `_ => arg_vars[0].0.clone()` and therefore emits NO conversion
/// at all for this shape -- it reinterprets the f32 buffer as i32. That
/// is a pre-existing silent-wrong-answer path on `cast` (chelis#729 /
/// chelis#730 territory, not fixed here). `cast_trunc` must not acquire
/// the same hole by someone "fixing" this rejection with a fallback arm.
#[test]
fn host_lane_tensor_cast_trunc_rejects_loudly_rather_than_passing_through() {
    let program = "module M.Main\n\
                   def f() -> tensor[2, i32] = \
                   cast_trunc(to_tensor([1.9, sqrt(4.0)]), i32)\n\
                   out = print(f())\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--emit-c",
            path.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            dir.path().join("out").to_str().unwrap(),
        ])
        .output()
        .expect("chelis build should run");
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !out.status.success(),
        "the host tensor lane has no cast_trunc emission; it must not build. \
         If this starts passing, verify the emitted C actually CONVERTS rather \
         than reinterpreting the buffer: stderr={stderr}"
    );
    assert!(
        stderr.contains("cast_trunc") && stderr.contains("float-to-integer only"),
        "the rejection must be the typed [05-OP-6] one naming the op, not a \
         generic failure: {stderr}"
    );
}

// ===========================================================================
// Surface round-trips
// ===========================================================================

#[test]
fn canonical_formatting_preserves_the_rung() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    write_file(
        &path,
        "module M.Main\n\ndef f(x: f32) -> i32 = cast_trunc(x, i32)\n",
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", path.to_str().unwrap()])
        .output()
        .expect("chelis fmt should run");
    assert!(out.status.success(), "fmt must succeed");
    let formatted = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        formatted.contains("cast_trunc(x, i32)"),
        "the formatter must not rewrite `cast_trunc` into the checked `cast`: {formatted}"
    );
}
