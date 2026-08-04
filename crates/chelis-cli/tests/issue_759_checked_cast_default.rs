//! chelis#759 / chelis#729 rework - the CHECKED cast is the default on
//! every eval surface (spec/04-type-system.md section 5.2, authored at
//! the PR #857 rework; `spec/design/dtype_semantics.md` section C3's
//! cast ladder).
//!
//! One authored rule per direction, identical on the scalar and tensor
//! surfaces:
//!
//! * float -> int: require an integral finite value, then the width check. Out of
//!   range TRAPS `overflow` (pre-rework the tensor surface SATURATED:
//!   `cast(300.0, int8)` was `127`).
//! * int -> narrower int: exact value or `overflow` trap (pre-rework
//!   the tensor surface WRAPPED: int32 `300 -> int8` was `44`).
//! * NaN / inf -> int: `domain` trap.
//! * anything -> bool: STRICT {0, 1} membership or `domain` trap
//!   (pre-rework the tensor surface encoded nonzero-to-1:
//!   `cast(2, bool)` was `true`). Scalar->bool now WORKS under the
//!   same rule (pre-rework it was a loud "unsupported cast" hole).
//! * any -> float: IEEE RNE finalize at the target width; overflow is
//!   the correctly signed infinity, never a trap ([04-NUM-2]).
//!
//! The named lossy/wrapping cast forms remain chelis#759's future surface.
//! The compiled C lane adopted the checked ladder in chelis#729 Phase 3;
//! the former red rows below are unconditional cross-lane regressions.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// Run a printed expression through `chelis eval --file`; return the
/// first printed line verbatim, or the stderr on failure.
fn eval_lane_str(expr: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    write_file(&path, &format!("module M.Main\nout = print({expr})\n"));
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

/// Eval a whole program (for `if`-shaped fold probes); first stdout
/// line on success, stderr on failure.
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

/// The expression must trap with the branded numeric-trap diagnostic
/// carrying `kind` and `prim` (section C2 shape: `numeric trap: <kind>
/// in <op> at <prim>`).
fn assert_cast_traps(expr: &str, kind: &str, prim: &str, label: &str) {
    match eval_lane_str(expr) {
        Ok(v) => panic!(
            "{label}: `{expr}` must trap ({kind} at {prim}) under the checked \
             cast default (spec/04 section 5.2, chelis#759), but it succeeded \
             and returned {v}"
        ),
        Err(stderr) => {
            assert!(
                stderr.contains(&format!("numeric trap: {kind}")) && stderr.contains(prim),
                "{label}: `{expr}` failed but without the branded {kind}-at-{prim} \
                 diagnostic. Got: {stderr}"
            );
        }
    }
}

// ===========================================================================
// float -> int: integral-and-in-range or trap.
// ===========================================================================

#[test]
fn tensor_fractional_float_to_int_traps_domain() {
    assert_cast_traps(
        "cast(to_tensor([3.5, -3.5]), int8)",
        "domain",
        "int8",
        "tensor_fractional_f2i",
    );
}

#[test]
fn scalar_fractional_float_to_int_traps_domain() {
    let stderr = eval_lane_str("cast(3.5, int8)").expect_err("fractional cast must trap");
    assert!(
        stderr.contains("numeric trap: domain in cast at int8"),
        "expected branded Domain trap: {stderr}"
    );
    // chelis#759's float-to-int rung SHIPPED as `cast_trunc`
    // ([05-OP-6]), so the hint now names it as the migration target
    // rather than calling it future work.
    let expected_hint_suffix = "; hint: fractional float-to-int conversion must state its \
        rounding explicitly: use `cast_trunc` to truncate toward zero ([05-OP-6]), or apply \
        `floor` or `round` before `cast`; the remaining named lossy cast forms are tracked \
        by chelis#759";
    assert!(
        stderr.contains(expected_hint_suffix),
        "fractional runtime cast must carry the complete teaching suffix: {stderr}"
    );
    for spelling in ["cast_trunc", "floor", "round", "chelis#759"] {
        assert!(
            stderr.contains(spelling),
            "fractional cast diagnostic must teach `{spelling}`: {stderr}"
        );
    }
}

#[test]
fn integral_float_to_int_casts_exactly_on_both_eval_surfaces() {
    let tensor = eval_lane_str("cast(to_tensor([3.0, -3.0]), int8)").expect("integral tensor");
    assert_eq!(tensor, "tensor(shape=[2], data=[3, -3])");
    assert_eq!(
        eval_lane_str("cast(3.0, int8)").expect("integral scalar"),
        "3"
    );
}

#[test]
fn tensor_float_to_int_out_of_range_traps_overflow_not_saturate() {
    // Pre-rework: saturated to 127 on this surface.
    assert_cast_traps(
        "cast(to_tensor([300.0]), int8)",
        "overflow",
        "int8",
        "tensor_f2i_overflow",
    );
}

#[test]
fn scalar_float_to_int_out_of_range_traps_overflow() {
    assert_cast_traps(
        "cast(300.0, int8)",
        "overflow",
        "int8",
        "scalar_f2i_overflow",
    );
}

#[test]
fn tensor_non_finite_to_int_traps_domain() {
    // sqrt(-1.0) is NaN; div(1.0, 0.0) is +inf (IEEE, [04-NUM-2]).
    assert_cast_traps(
        "cast(sqrt(to_tensor([-1.0])), int32)",
        "domain",
        "int32",
        "tensor_nan_to_int",
    );
    assert_cast_traps(
        "cast(div(to_tensor([1.0]), to_tensor([0.0])), int32)",
        "domain",
        "int32",
        "tensor_inf_to_int",
    );
}

// ===========================================================================
// int -> narrower int: exact in range; overflow trap out of range (no wrap).
// ===========================================================================

#[test]
fn tensor_int_narrowing_in_range_is_exact() {
    let got = eval_lane_str("cast(to_tensor([127, -128, 0]), int8)").expect("in-range eval");
    assert_eq!(got, "tensor(shape=[3], data=[127, -128, 0])");
}

#[test]
fn tensor_int_narrowing_out_of_range_traps_overflow_not_wrap() {
    // Pre-rework: wrapped two's-complement to 44 on this surface.
    assert_cast_traps(
        "cast(to_tensor([300]), int8)",
        "overflow",
        "int8",
        "tensor_i2i_overflow",
    );
}

#[test]
fn scalar_int_narrowing_out_of_range_traps_overflow() {
    assert_cast_traps(
        "cast(cast(300, int32), int8)",
        "overflow",
        "int8",
        "scalar_i2i_overflow",
    );
}

// ===========================================================================
// -> bool: strict {0, 1} membership on both surfaces.
// ===========================================================================

#[test]
fn tensor_cast_to_bool_accepts_exact_zero_one_only() {
    let got = eval_lane_str("cast(to_tensor([1, 0]), bool)").expect("0/1 eval");
    assert_eq!(got, "tensor(shape=[2], data=[true, false])");
}

#[test]
fn tensor_cast_to_bool_rejects_nonzero_nonone_with_domain_trap() {
    // Pre-rework: nonzero encoded true on this surface (cast(2, bool) = true).
    assert_cast_traps(
        "cast(to_tensor([2]), bool)",
        "domain",
        "bool",
        "tensor_to_bool_strict",
    );
}

#[test]
fn scalar_cast_to_bool_works_under_the_strict_rule() {
    // Pre-rework the scalar surface had NO bool arm (loud "unsupported
    // cast"); the unified ladder gives it the same strict rule.
    assert_eq!(
        eval_lane_str("cast(1, bool)").expect("cast(1, bool)"),
        "true"
    );
    assert_eq!(
        eval_lane_str("cast(0, bool)").expect("cast(0, bool)"),
        "false"
    );
    assert_cast_traps("cast(2, bool)", "domain", "bool", "scalar_to_bool_strict");
    assert_cast_traps(
        "cast(0.5, bool)",
        "domain",
        "bool",
        "scalar_fractional_to_bool",
    );
}

// ===========================================================================
// -> float: finalize, never trap (negative parity for the trap rules).
// ===========================================================================

#[test]
fn casts_to_float_targets_finalize_and_never_trap() {
    // f16 overflow goes to inf per IEEE ([04-NUM-2]), not a trap.
    assert_eq!(
        eval_lane_str("cast(to_tensor([1000000.0]), f16)").expect("f16 overflow eval"),
        "tensor(shape=[1], data=[inf])"
    );
    // int64 above 2^53 to f64 is the lossy-by-design direction ([04-NUM-6]).
    assert_eq!(
        eval_lane_str("cast(9007199254740993i64, f64)").expect("int64->f64"),
        "9007199254740992.0"
    );
    // int32 above 2^24 similarly rounds at f32 width. This is a
    // by-design lossy direction, not a checked-cast trap.
    assert_eq!(
        eval_lane_str("cast(16777217i32, f32)").expect("int32->f32"),
        "16777216.0"
    );
}

#[test]
fn int64_to_bf16_rounds_once_at_the_target_width_on_both_eval_surfaces() {
    // This exact integer is one above a bf16 midpoint but first becomes the
    // midpoint if it is rounded through f64. Direct target-width RNE must
    // therefore choose the upper bf16 value ([04-NUM-14]).
    const ABOVE_MIDPOINT: &str = "4629700416936869889i64";
    assert_eq!(
        eval_lane_str(&format!("cast({ABOVE_MIDPOINT}, bf16)")).expect("scalar int64->bf16"),
        "4.65e18"
    );
    assert_eq!(
        eval_lane_str(&format!("cast(to_tensor([{ABOVE_MIDPOINT}]), bf16)"))
            .expect("tensor int64->bf16"),
        "tensor(shape=[1], data=[4.65e18])"
    );
}

// ===========================================================================
// The fold rule: a trapping cast DECLINES TO FOLD (section C2's decline
// clause); the condition falls to runtime and traps loudly there. A fold
// must never bake the trap away into a taken-or-deleted branch.
// ===========================================================================

#[test]
fn static_if_with_trapping_cast_condition_traps_at_runtime_not_folds() {
    let program = "def pick() -> f32 = if lt(cast(300.0, int8), 10i8) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    match eval_program(program) {
        Ok(v) => panic!(
            "a compile-time out-of-range cast condition must not fold to a \
             branch; the checked ladder declines and the runtime trap fires. \
             Got a folded/evaluated answer: {v}"
        ),
        Err(stderr) => assert!(
            stderr.contains("numeric trap: overflow") && stderr.contains("int8"),
            "expected the branded overflow trap from the unfolded runtime \
             condition; got: {stderr}"
        ),
    }
}

// ===========================================================================
// Compiled C lane: chelis#729 Phase 3 checked-cast regressions. These retain
// the original assertions that were red before the compiled lane adopted the
// same checked ladder as eval (dtype_semantics.md section C3).
// ===========================================================================

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build + link + run a program through the C lane; return
/// `(stdout, stderr, ok)`.
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

#[test]
fn c_tensor_float_to_int_out_of_range_traps_overflow() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "def f() -> tensor[1, int8] = cast(to_tensor([300.0]), int8)\n\
                   out = print(f())\n";
    let (stdout, stderr, ok) = c_lane_run(program, "c_checked_cast_f2i").expect("C lane");
    assert!(
        !ok && stderr.contains("overflow"),
        "compiled cast(300.0, int8) must trap with the branded overflow \
         diagnostic (chelis#729 Phase 3); got ok={ok} stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn c_tensor_fractional_float_to_int_traps_domain() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "def f() -> tensor[1, int8] = cast(to_tensor([3.5]), int8)\n\
                   out = print(f())\n";
    let (stdout, stderr, ok) =
        c_lane_run(program, "c_checked_cast_fractional_f2i").expect("C lane");
    assert!(
        !ok && stderr.contains("domain"),
        "compiled fractional float->int cast must Domain-trap (chelis#729 Phase 3); \
         got ok={ok} stdout={stdout} stderr={stderr}"
    );
}

#[test]
fn c_tensor_int_narrowing_out_of_range_traps_overflow() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = "def f() -> tensor[1, int8] = cast(to_tensor([300]), int8)\n\
                   out = print(f())\n";
    let (stdout, stderr, ok) = c_lane_run(program, "c_checked_cast_i2i").expect("C lane");
    assert!(
        !ok && stderr.contains("overflow"),
        "compiled int32->int8 out-of-range cast must trap (chelis#729 Phase 3); \
         got ok={ok} stdout={stdout} stderr={stderr}"
    );
}
