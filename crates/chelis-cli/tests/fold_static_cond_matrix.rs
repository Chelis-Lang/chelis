//! chelis#720 regression matrix for `fold_static_cond`: cast operands must be
//! finalized as sealed f16/bf16 scalars before comparison. Folding through an
//! f32 memo deletes the branch IEEE f16/bf16 semantics require.
//!
//! Sibling of chelis#711 (the integer Const arm of the same fold, threshold
//! 2^53); this one fires at 2049 (f16) / 257 (bf16). The fixes do not
//! overlap: #711's checked-i64 folding does not touch the Cast arm.
//!
//! Bounding controls use executed effectful host-lane programs, not generated
//! C text. Their three-row truth table distinguishes exact `<` from `<=`,
//! swapped operands, f64-rounded integers, wrong condition dataflow, wrong
//! fail behavior, and loss of either branch. Exact stdout and stderr are the
//! acceptance oracle.
//!
//! The Phase 3 i8 row also proves that folding cannot hide the required
//! checked-overflow trap.

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
        .map(|output| output.status.success())
        .unwrap_or(false)
}

struct CRun {
    stdout: String,
    stderr: String,
    ok: bool,
}

fn build_and_run_c(program: &str, name: &str) -> Result<CRun, String> {
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
    Ok(CRun {
        stdout: String::from_utf8_lossy(&run.stdout).into_owned(),
        stderr: String::from_utf8_lossy(&run.stderr).into_owned(),
        ok: run.status.success(),
    })
}

fn eval_first_line(program: &str) -> Result<String, String> {
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
    Ok(String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string())
}

const I64_EXACT_LOW: u64 = 9007199254740992;
const I64_EXACT_HIGH: u64 = 9007199254740993;
const TAKEN_FAIL_STDERR: &str = "i64 invariant violated\n";
const PRINT_222_STDOUT: &str = "222.0\nout = ()\n";
const STATIC_PICK_222_STDOUT: &str = "222.0\npick = 222.0\nout = ()\n";

#[derive(Clone, Copy)]
enum ExpectedHostOutcome {
    Fail,
    Print222,
}

#[derive(Clone, Copy)]
struct HostTruthCase {
    name: &'static str,
    left: u64,
    right: u64,
    expected: ExpectedHostOutcome,
}

const HOST_LT_TRUTH_TABLE: [HostTruthCase; 3] = [
    HostTruthCase {
        name: "low_high",
        left: I64_EXACT_LOW,
        right: I64_EXACT_HIGH,
        expected: ExpectedHostOutcome::Fail,
    },
    HostTruthCase {
        name: "equal_equal",
        left: I64_EXACT_HIGH,
        right: I64_EXACT_HIGH,
        expected: ExpectedHostOutcome::Print222,
    },
    HostTruthCase {
        name: "high_low",
        left: I64_EXACT_HIGH,
        right: I64_EXACT_LOW,
        expected: ExpectedHostOutcome::Print222,
    },
];

fn effectful_host_program(
    condition: &str,
    then_expr: &str,
    else_expr: &str,
    case: HostTruthCase,
) -> String {
    format!(
        "def pick(left: i64, right: i64) -> f32 = if {condition} \
         then {then_expr} else {else_expr}\n\
         out = print(pick({}i64, {}i64))\n",
        case.left, case.right
    )
}

#[derive(Debug)]
enum HostTruthError {
    Setup(String),
    Mismatch(String),
}

fn verify_host_truth_table(
    label: &str,
    condition: &str,
    then_expr: &str,
    else_expr: &str,
) -> Result<(), HostTruthError> {
    for case in HOST_LT_TRUTH_TABLE {
        let program = effectful_host_program(condition, then_expr, else_expr, case);
        let run = build_and_run_c(&program, &format!("{label}_{}", case.name))
            .map_err(HostTruthError::Setup)?;
        match case.expected {
            ExpectedHostOutcome::Fail => {
                if run.ok || !run.stdout.is_empty() || run.stderr != TAKEN_FAIL_STDERR {
                    return Err(HostTruthError::Mismatch(format!(
                        "{}: expected exact fail, got ok={}, stdout={:?}, stderr={:?}",
                        case.name, run.ok, run.stdout, run.stderr
                    )));
                }
            }
            ExpectedHostOutcome::Print222 => {
                if !run.ok || run.stdout != PRINT_222_STDOUT || !run.stderr.is_empty() {
                    return Err(HostTruthError::Mismatch(format!(
                        "{}: expected exact 222.0 output, got ok={}, stdout={:?}, stderr={:?}",
                        case.name, run.ok, run.stdout, run.stderr
                    )));
                }
            }
        }
    }
    Ok(())
}

fn assert_host_mutation_rejected(label: &str, condition: &str, then_expr: &str, else_expr: &str) {
    match verify_host_truth_table(label, condition, then_expr, else_expr) {
        Err(HostTruthError::Mismatch(error)) => {
            assert!(!error.is_empty(), "mutation mismatch must explain itself");
        }
        Err(HostTruthError::Setup(error)) => {
            panic!("mutation `{label}` did not build and run: {error}")
        }
        Ok(()) => panic!("semantic truth table accepted mutation `{label}`"),
    }
}

// ===========================================================================
// Effectful host-lane exact-integer controls
// ===========================================================================

#[test]
fn effectful_host_lt_truth_table_is_exact() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    verify_host_truth_table(
        "fold_fail_truth",
        "lt(left, right)",
        "fail(\"i64 invariant violated\")",
        "222.0",
    )
    .unwrap_or_else(|error| panic!("exact host `<` truth table failed: {error:?}"));
}

#[test]
fn semantic_controls_reject_lte() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    assert_host_mutation_rejected(
        "fold_fail_mut_lte",
        "or(lt(left, right), eq(left, right))",
        "fail(\"i64 invariant violated\")",
        "222.0",
    );
}

#[test]
fn semantic_controls_reject_swapped_operands() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    assert_host_mutation_rejected(
        "fold_fail_mut_swapped",
        "lt(right, left)",
        "fail(\"i64 invariant violated\")",
        "222.0",
    );
}

#[test]
fn semantic_controls_reject_f64_rounded_comparison() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    assert_host_mutation_rejected(
        "fold_fail_mut_f64",
        "lt(cast(left, f64), cast(right, f64))",
        "fail(\"i64 invariant violated\")",
        "222.0",
    );
}

#[test]
fn semantic_controls_reject_wrong_condition_dataflow() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    assert_host_mutation_rejected(
        "fold_fail_mut_dataflow",
        "lt(left, left)",
        "fail(\"i64 invariant violated\")",
        "222.0",
    );
}

#[test]
fn semantic_controls_reject_wrong_fail_message() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    assert_host_mutation_rejected(
        "fold_fail_mut_message",
        "lt(left, right)",
        "fail(\"wrong invariant\")",
        "222.0",
    );
}

#[test]
fn semantic_controls_reject_wrong_fail_target() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    assert_host_mutation_rejected("fold_fail_mut_target", "lt(left, right)", "111.0", "222.0");
}

#[test]
fn semantic_controls_reject_lost_surviving_branch() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    assert_host_mutation_rejected(
        "fold_fail_mut_else",
        "lt(left, right)",
        "fail(\"i64 invariant violated\")",
        "111.0",
    );
}

#[test]
fn semantic_controls_reject_markers_hidden_in_dead_nested_branches() {
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    assert_host_mutation_rejected(
        "fold_fail_mut_dead_nested",
        "lt(left, right)",
        "if lt(left, left) then fail(\"i64 invariant violated\") else 111.0",
        "if lt(left, left) then 222.0 else 111.0",
    );
}

// ===========================================================================
// chelis#720 - the Cast arm deletes the IEEE-correct branch
// ===========================================================================

/// True f16 rounds cast(2049.0, f16) to 2048, so lt(2048, 2048) is false and
/// the answer is 222. Before the compiled Phase 3 fix, eval printed 222
/// (correct) while the compiled binary printed 111 because the correct branch
/// was deleted at compile time.
#[test]
fn f16_cast_condition_folds_with_f16_semantics() {
    let program = "def pick() -> f32 = if lt(cast(2048.0, f16), cast(2049.0, f16)) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    assert_eq!(
        eval_first_line(program).expect("eval"),
        "222.0",
        "eval is the correct lane here and must stay correct"
    );
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let run = build_and_run_c(program, "fold_f16").expect("C lane");
    assert!(run.ok, "compiled program failed: {}", run.stderr);
    assert_eq!(
        run.stdout, STATIC_PICK_222_STDOUT,
        "the compiled program must take the IEEE f16 branch"
    );
    assert_eq!(run.stderr, "");
}

/// bf16 sibling at threshold 257 (8-bit mantissa).
#[test]
fn bf16_cast_condition_folds_with_bf16_semantics() {
    let program = "def pick() -> f32 = if lt(cast(256.0, bf16), cast(257.0, bf16)) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    assert_eq!(eval_first_line(program).expect("eval"), "222.0");
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let run = build_and_run_c(program, "fold_bf16").expect("C lane");
    assert!(run.ok, "compiled program failed: {}", run.stderr);
    assert_eq!(
        run.stdout, STATIC_PICK_222_STDOUT,
        "the compiled program must take the IEEE bf16 branch"
    );
    assert_eq!(run.stderr, "");
}

// ===========================================================================
// chelis#718 - i8 conditions do NOT fold, but the runtime branch diverges
// through the int64_t widening (#714). Contract: the overflow must trap.
// ===========================================================================

/// `add(100i8, 100i8)` overflows i8. Today eval wraps (-56 < 0, prints
/// 111) and compiled C widens (200 < 0, prints 222) - opposite branches at
/// runtime. The decided contract (#680/#695) says the overflow itself must
/// trap in both lanes.
#[test]
fn int8_overflow_condition_traps_in_both_lanes() {
    let program = "def pick() -> f32 = if lt(add(100i8, 100i8), 0i8) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    match eval_first_line(program) {
        Ok(line) => panic!("eval must trap on the i8 overflow, got: {line}"),
        Err(stderr) => assert!(stderr.contains("overflow"), "got: {stderr}"),
    }
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let run = build_and_run_c(program, "fold_i8").expect("C lane");
    assert!(
        !run.ok && run.stderr.contains("overflow"),
        "compiled C must trap on the i8 overflow; got ok={}, stdout `{}`",
        run.ok,
        run.stdout
    );
}
