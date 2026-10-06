//! chelis#1180: `>` must evaluate its operands in authored order.
//!
//! Surf desugared `a > b` as `cmplt(b, a)`. Deep application evaluates its
//! arguments left to right, so the right operand's effects and traps ran
//! before the left operand's, in every executable lane. The fix desugars
//! `a > b` to the existing `gt` builtin, whose value semantics are the
//! [05-OP-defined] `cmplt(b, a)` (spec/05-risc-primitives.md section 3.2)
//! while its application evaluates the authored left operand first
//! (spec/03-deep-syntax.md section 4.4, spec/02-surf-syntax.md section 2).
//!
//! Coverage:
//! - Surf-to-Deep: `chelis deep` lowers `lhs() > rhs()` to an in-order
//!   `gt` application, with the left operand printed first.
//! - eval lane: effectful operands run left to right; comparison values
//!   including NaN behavior are unchanged; when both operands trap, the
//!   left operand's trap surfaces.
//! - compiled C lane: byte-identical stdout with eval on the effectful
//!   program (transcript order plus [05-OBS-6] labeled roots), and the
//!   left trap surfaces with the same branded diagnostic class.
//! - negative parity: `<`, `<=`, `>=` keep their (already correct)
//!   left-to-right order.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Run `chelis eval --file`; `Ok(full stdout)` on success, `Err(stderr)`
/// on a non-zero exit (the trap rows assert on that stderr).
fn eval_stdout(program: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
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
    Ok(String::from_utf8_lossy(&out.stdout).trim_end().to_string())
}

/// Run `chelis deep` and return stdout.
fn deep_stdout(program: &str) -> String {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["deep", path.to_str().unwrap()])
        .output()
        .expect("chelis deep should run");
    assert!(
        out.status.success(),
        "chelis deep failed: {}",
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Build with the C backend, link, and run. `Ok((full stdout, stderr,
/// exit ok))`; a trap exits non-zero with a branded stderr, which is not
/// an `Err` here.
fn c_lane(program: &str, name: &str) -> Result<(String, String, bool), String> {
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
        String::from_utf8_lossy(&run.stdout).trim_end().to_string(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
        run.status.success(),
    ))
}

/// Two effectful operands with distinct markers around one comparison.
fn effectful_comparison_program(operator: &str) -> String {
    format!(
        "def lhs() -> f32 ! {{ IO }} = {{\n\
         \x20 _ = print(\"LHS\")\n\
         \x20 1.0\n\
         }}\n\
         \n\
         def rhs() -> f32 ! {{ IO }} = {{\n\
         \x20 _ = print(\"RHS\")\n\
         \x20 2.0\n\
         }}\n\
         \n\
         verdict = lhs() {operator} rhs()\n"
    )
}

/// Both operands trap with the same dtype but different trap kinds, so
/// whichever operand evaluates first is visible in the branded message.
const BOTH_OPERANDS_TRAP: &str = "verdict = cast_trunc(300.9, i8) > cast_trunc(sqrt(-1.0), i8)\n";

/// Value semantics around the fix: strict order, equal operands, and both
/// NaN positions. Every root is a bool, so the assertions are exact.
const GT_VALUE_TABLE: &str = "strict = 2.0 > 1.0\n\
     flipped = 1.0 > 2.0\n\
     equal_operands = 1.0 > 1.0\n\
     nan_left = div(0.0, 0.0) > 1.0\n\
     nan_right = 1.0 > div(0.0, 0.0)\n";

// ===========================================================================
// Surf-to-Deep
// ===========================================================================

#[test]
fn deep_lowers_gt_to_an_in_order_gt_application() {
    let deep = deep_stdout(&effectful_comparison_program(">"));
    // Scope the order assertions to the `verdict` binding; the printed
    // Deep carries span metadata, so match name atoms rather than the
    // metadata-free canonical spelling.
    let verdict = &deep[deep.find("verdict").expect("verdict binding present")..];
    let gt_offset = verdict
        .find("} gt)")
        .expect("`lhs() > rhs()` must lower through the gt built-in");
    let lhs_offset = verdict.find("} lhs)").expect("left operand call present");
    let rhs_offset = verdict.find("} rhs)").expect("right operand call present");
    assert!(
        gt_offset < lhs_offset && lhs_offset < rhs_offset,
        "`lhs() > rhs()` must lower to an in-order `gt` application:\n{deep}"
    );
    assert!(
        !deep.contains("cmplt"),
        "`>` must not lower through operand-swapped cmplt:\n{deep}"
    );
}

// ===========================================================================
// eval lane
// ===========================================================================

#[test]
fn eval_gt_runs_effectful_operands_left_to_right() {
    let stdout = eval_stdout(&effectful_comparison_program(">")).expect("eval succeeds");
    assert_eq!(
        stdout, "LHS\nRHS\nverdict = false",
        "the authored left operand must evaluate before the right one"
    );
}

#[test]
fn eval_gt_value_semantics_are_unchanged_including_nan() {
    let stdout = eval_stdout(GT_VALUE_TABLE).expect("eval succeeds");
    assert_eq!(
        stdout,
        "strict = true\n\
         flipped = false\n\
         equal_operands = false\n\
         nan_left = false\n\
         nan_right = false",
        "`>` must keep cmplt(b, a) value semantics, false on either NaN"
    );
}

#[test]
fn eval_gt_surfaces_the_left_operands_trap_first() {
    let stderr = eval_stdout(BOTH_OPERANDS_TRAP).expect_err("both operands trap");
    assert!(
        stderr.contains("overflow in cast_trunc at i8"),
        "the authored LEFT operand's overflow trap must surface: {stderr}"
    );
    assert!(
        !stderr.contains("domain in cast_trunc"),
        "the right operand's domain trap must not be reached first: {stderr}"
    );
}

#[test]
fn eval_other_comparisons_keep_left_to_right_order() {
    for (operator, verdict) in [("<", "true"), ("<=", "true"), (">=", "false")] {
        let stdout = eval_stdout(&effectful_comparison_program(operator)).expect("eval succeeds");
        assert_eq!(
            stdout,
            format!("LHS\nRHS\nverdict = {verdict}"),
            "`{operator}` must evaluate its authored left operand first"
        );
    }
}

// ===========================================================================
// consumer parity: the invariant grammar admits the new desugared form
// ===========================================================================

#[test]
fn opaque_invariant_written_with_gt_stays_well_formed() {
    // The D-WF invariant grammar and its boolean-shape check must admit
    // `gt`, the form `>` now desugars to; before chelis#1180 they only
    // ever saw the operand-swapped `cmplt` spelling.
    let program = "module Stats.Prob\n\
         export (probability)\n\
         @opaque\n\
         @invariant(p) p.value > 0.0\n\
         type Probability =\n\
         \x20 | Probability { value: f32 }\n\
         def probability(x: f32) -> Option[Probability] =\n\
         \x20 if x > 0.0 then Some(Probability { value: x }) else None\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("m.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let stdout = String::from_utf8_lossy(&out.stdout);
    assert!(
        out.status.success() && stdout.contains("\"score\": 1"),
        "a `>` invariant predicate must stay well-formed:\nstdout: {stdout}\nstderr: {}",
        String::from_utf8_lossy(&out.stderr)
    );
}

// ===========================================================================
// compiled C lane
// ===========================================================================

#[test]
fn c_lane_gt_effect_order_matches_eval_byte_for_byte() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let program = effectful_comparison_program(">");
    let eval = eval_stdout(&program).expect("eval succeeds");
    let (stdout, stderr, ok) = c_lane(&program, "gt_effect_order").expect("C lane");
    assert!(ok, "compiled binary must succeed; stderr: {stderr}");
    assert_eq!(
        stdout, eval,
        "compiled C stdout must match eval byte for byte ([05-OBS-6])"
    );
    assert_eq!(
        stdout, "LHS\nRHS\nverdict = false",
        "both lanes must observe the authored operand order"
    );
}

#[test]
fn c_lane_gt_value_semantics_match_eval() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let eval = eval_stdout(GT_VALUE_TABLE).expect("eval succeeds");
    let (stdout, stderr, ok) = c_lane(GT_VALUE_TABLE, "gt_value_table").expect("C lane");
    assert!(ok, "compiled binary must succeed; stderr: {stderr}");
    assert_eq!(stdout, eval, "compiled C values must match eval");
}

#[test]
fn tensor_gt_agrees_between_eval_and_c() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    // The tensor path lowers `gt` through the Tier 2 DAG rewrite to
    // cmplt(b, a) over already-evaluated operand values; both lanes must
    // produce the same elementwise verdicts.
    let program = "mask = to_tensor([1.0, 3.0], f32) > to_tensor([2.0, 2.0], f32)\n";
    let eval = eval_stdout(program).expect("eval succeeds");
    assert!(
        eval.contains("mask ="),
        "eval must produce the labeled mask root: {eval}"
    );
    let (stdout, stderr, ok) = c_lane(program, "gt_tensor").expect("C lane");
    assert!(ok, "compiled binary must succeed; stderr: {stderr}");
    assert_eq!(stdout, eval, "tensor `>` must agree across lanes");
}

#[test]
fn c_lane_gt_surfaces_the_left_operands_trap_first() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (stdout, stderr, ok) = c_lane(BOTH_OPERANDS_TRAP, "gt_trap_order").expect("C lane");
    assert!(
        !ok,
        "both operands trap, so the binary must exit non-zero; stdout: {stdout}"
    );
    assert!(
        stderr.contains("overflow in cast_trunc at i8"),
        "the authored LEFT operand's overflow trap must surface: {stderr}"
    );
    assert!(
        !stderr.contains("domain in cast_trunc"),
        "the right operand's domain trap must not be reached first: {stderr}"
    );
}
