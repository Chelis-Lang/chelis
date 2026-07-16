//! dtype x op x lane precision matrix (chelis#680, #682, #684).
//!
//! ## What this file is
//!
//! A single table-driven statement of what Chelis arithmetic is supposed to do
//! at every dtype's precision boundary, in every execution lane, with each cell
//! labelled by its current status. Three questions get answered per row:
//!
//!   1. is the value CORRECT (matches exact integer / IEEE-754 float semantics)?
//!   2. do the LANES AGREE (`chelis eval` vs `chelis build --target c`)?
//!   3. is the current status a lock, a known bug, or correct-by-design?
//!
//! ## The subtlety this file exists to protect
//!
//! `add(2^53, 1)`:
//!   * at `f64`   -> `2^53` is **CORRECT**. `2^53 + 1` is not representable in
//!     an f64 mantissa (53 bits), so IEEE-754 ties-to-even rounds it down. Any
//!     "fix" that makes this return `2^53 + 1` has broken f64.
//!   * at `int64` -> `2^53` is a **BUG**. The value is exactly representable in
//!     an i64; it is only wrong because the evaluator laundered it through f64.
//!
//! Same literal inputs, opposite verdicts. That collision is precisely why the
//! bug survived: an f64-typed, tolerance-based oracle
//! (`crates/chelis-e2e/tests/eval_agreement.rs`, `assert_close(a: f64, b: f64,
//! tol)`) cannot distinguish the two cases, so it passed both.
//!
//! ## Status vocabulary
//!
//! * `Locked` - verified correct today; this row is a regression lock.
//! * `ByDesign` - lossy, and CORRECT to be lossy (float mantissa limits). Locks
//!   the boundary so a fix does not over-correct.
//! * `Broken` - verified wrong today; carries its issue number. These assert the
//!   correct behavior and fail until the fix lands, which is the point (write
//!   the failing test first, per the repo contract). They are `#[ignore]`d so CI
//!   stays green; run them with `-- --ignored`.
//!
//! Every `Broken` row below was confirmed by running the code, not by reading
//! it. Two claims that came from source inspection alone turned out to be false
//! during this investigation, so nothing here is asserted without execution.
//!
//! ## Precision boundaries used
//!
//! | dtype | mantissa | first non-representable integer | max |
//! |-------|----------|--------------------------------|-----|
//! | f16   | 11 bits  | 2049 (2^11+1)                  | 65504 |
//! | bf16  | 8 bits   | 257 (2^8+1)                    | ~3.4e38 |
//! | f32   | 24 bits  | 16777217 (2^24+1)              | ~3.4e38 |
//! | f64   | 53 bits  | 9007199254740993 (2^53+1)      | ~1.8e308 |
//! | int8  | exact    | n/a                            | 127 |
//! | int16 | exact    | n/a                            | 32767 |
//! | int32 | exact    | n/a                            | 2147483647 |
//! | int64 | exact    | n/a                            | 9223372036854775807 |
//!
//! Integer dtypes are exact by definition across their whole range; that is the
//! entire content of #680.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

// ---------------------------------------------------------------------------
// Boundary constants
// ---------------------------------------------------------------------------

/// `i64::MAX`. Rust's saturating `f64 as i64` clamps here on overflow, which is
/// why it is a misleading round-trip probe (see
/// `i64_max_tensor_round_trip_passes_by_luck_not_by_correctness`).
const I64_MAX: i64 = 9_223_372_036_854_775_807;

// ---------------------------------------------------------------------------
// Matrix vocabulary
// ---------------------------------------------------------------------------

/// Current status of a matrix cell.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Status {
    /// Verified correct today. Regression lock.
    Locked,
    /// Lossy AND correct to be lossy (float mantissa). Locks the boundary so a
    /// fix does not over-correct the float lane.
    ByDesign,
    /// Verified wrong today. Carries the tracking issue.
    Broken(&'static str),
}

/// Which lanes a row is asserted in.
#[derive(Clone, Copy, Debug, PartialEq)]
enum Lanes {
    /// Evaluator only (op has no compiled-lane form, or the fixture is
    /// eval-shaped).
    EvalOnly,
    /// Both lanes, and they must AGREE with each other as well as with the
    /// expected value. This is the invariant #680 turns on.
    Both,
}

struct Row<'a> {
    name: &'a str,
    /// Surf expression whose value is printed.
    expr: &'a str,
    /// Exact expected value, rendered as the printed string.
    expected: &'a str,
    status: Status,
    lanes: Lanes,
    /// Why this row exists / what it protects.
    note: &'a str,
}

// ---------------------------------------------------------------------------
// Lane drivers
// ---------------------------------------------------------------------------

/// Run a printed expression through `chelis eval --file`, return the first
/// printed line verbatim.
///
/// Deliberately returns the raw STRING, never a parsed `f64`: parsing through
/// f64 is the bug under test, and an f64-typed oracle cannot observe it. This
/// is the fix for the blind spot in `eval_agreement.rs`.
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

fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build + link + run an expression through the C lane, return the printed
/// `out = <value>` payload verbatim as a string.
fn c_lane_str(expr: &str, ret_ty: &str, name: &str) -> Result<String, String> {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(
        &path,
        &format!("def run() -> {ret_ty} = {expr}\nout = run()\n"),
    );
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
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    let line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("out ="))
        .ok_or_else(|| format!("no `out =` line in:\n{stdout}"))?;
    Ok(line.split('=').nth(1).unwrap_or("").trim().to_string())
}

/// Drive one matrix row.
fn check_row(row: &Row, ret_ty: &str) {
    let eval_got =
        eval_lane_str(row.expr).unwrap_or_else(|e| panic!("{}: eval lane failed: {e}", row.name));

    if row.lanes == Lanes::Both && c_toolchain_available() {
        match c_lane_str(row.expr, ret_ty, row.name) {
            Ok(c_got) => {
                assert_eq!(
                    eval_got, c_got,
                    "{}: LANE DIVERGENCE. eval={eval_got}, compiled C={c_got}, \
                     for `{}`. The same program must not produce different \
                     values in different lanes. {}",
                    row.name, row.expr, row.note
                );
            }
            Err(e) => panic!("{}: C lane failed: {e}", row.name),
        }
    }

    assert_eq!(
        eval_got, row.expected,
        "{}: got {eval_got}, expected {}. status={:?}. {}",
        row.name, row.expected, row.status, row.note
    );
}

// ===========================================================================
// FLOAT LANE: lossy AND correct. These must NOT change when the integer lane
// is fixed. A fix that "corrects" these has broken IEEE-754.
// ===========================================================================

/// f64 at its own mantissa boundary. `2^53 + 1` is not representable in f64, so
/// rounding down to `2^53` is the CORRECT IEEE-754 answer.
///
/// This is the exact numeric case that is a BUG at int64 (see
/// `int64_add_at_mantissa_boundary_is_exact` below). Same inputs, opposite
/// verdicts.
#[test]
fn f64_add_at_mantissa_boundary_is_correctly_lossy() {
    check_row(
        &Row {
            name: "f64_add_2p53",
            expr: "add(cast(9007199254740992.0, f64), cast(1.0, f64))",
            expected: "9007199254740992",
            status: Status::ByDesign,
            lanes: Lanes::EvalOnly,
            note: "f64 has a 53-bit mantissa; 2^53+1 is not representable. \
                   Rounding to 2^53 is correct. Do not 'fix' this.",
        },
        "f64",
    );
}

/// f32 at its own mantissa boundary (24 bits). `2^24 + 1 = 16777217` is not
/// representable; rounding to `16777216` is correct.
///
/// Also pins that the evaluator's f64 kernel does not accidentally give f32
/// EXTRA precision: computing in f64 and rounding once to f32 is correctly
/// rounded (f64 carries >= 2p+2 bits for p=24), so this must match native f32.
#[test]
fn f32_add_at_mantissa_boundary_is_correctly_lossy() {
    check_row(
        &Row {
            name: "f32_add_2p24",
            expr: "add(cast(16777216.0, f32), cast(1.0, f32))",
            expected: "16777216",
            status: Status::ByDesign,
            lanes: Lanes::EvalOnly,
            note: "f32 has a 24-bit mantissa; 2^24+1 is not representable. \
                   Also pins that the f64 kernel does not grant f32 extra \
                   precision (double rounding is safe here: 53 >= 2*24+2).",
        },
        "f32",
    );
}

/// f32 below its boundary must be exact. Negative parity for the row above:
/// proves the boundary is a real mantissa limit, not blanket float sloppiness.
#[test]
fn f32_add_below_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "f32_add_below_2p24",
            expr: "add(cast(16777215.0, f32), cast(1.0, f32))",
            expected: "16777216",
            status: Status::Locked,
            lanes: Lanes::EvalOnly,
            note: "2^24-1 + 1 = 2^24 is exactly representable in f32.",
        },
        "f32",
    );
}

// ===========================================================================
// INTEGER LANE: must be exact across the whole dtype range. This is #680.
// ===========================================================================

/// The headline #680 case. Verified: eval returns 9007199254740992 while
/// compiled C returns 9007199254740993, so this asserts BOTH correctness and
/// lane parity.
#[test]
#[ignore = "chelis#680: eval returns 9007199254740992, compiled C returns 9007199254740993. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_add_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_add_2p53",
            expr: "add(cast(9007199254740992, int64), cast(1, int64))",
            expected: "9007199254740993",
            status: Status::Broken("chelis#680"),
            lanes: Lanes::Both,
            note: "int64 is exact across its whole range. f64's mantissa limit \
                   is not int64's problem. Contrast f64_add_at_mantissa_boundary.",
        },
        "int64",
    );
}

/// int64 below the f64 mantissa boundary already works. Negative parity: proves
/// the failure is specifically the 2^53 mantissa limit leaking in, not that
/// integer add is broken generally.
#[test]
fn int64_add_below_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_add_small",
            expr: "add(cast(2, int64), cast(3, int64))",
            expected: "5",
            status: Status::Locked,
            lanes: Lanes::Both,
            note: "Small int64 add works today; locks the common path.",
        },
        "int64",
    );
}

/// Exactly representable in i64, above 2^53, and its f64 image happens to be
/// exact too. Guards against a fix that clamps everything above 2^53 rather
/// than computing exactly.
#[test]
fn int64_mul_above_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_mul_above_2p53",
            expr: "mul(cast(94906266, int64), cast(94906266, int64))",
            expected: "9007199326062756",
            status: Status::Locked,
            lanes: Lanes::Both,
            note: "94906266^2 = 9007199326062756, exact in i64 and above 2^53. \
                   Guards against a fix that refuses/clamps above 2^53.",
        },
        "int64",
    );
}

/// int64 subtraction at the boundary.
#[test]
#[ignore = "chelis#680: operand 2^53+1 corrupted via f64. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_sub_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_sub_2p53",
            expr: "sub(cast(9007199254740993, int64), cast(1, int64))",
            expected: "9007199254740992",
            status: Status::Broken("chelis#680"),
            lanes: Lanes::Both,
            note: "The operand 2^53+1 is corrupted on the way in via f64.",
        },
        "int64",
    );
}

/// `abs` on int64. Verified: eval returns 9007199254740992 while compiled C
/// returns 9007199254740993.
#[test]
#[ignore = "chelis#680: eval returns 9007199254740992, compiled C returns 9007199254740993. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_abs_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_abs_2p53",
            expr: "abs(cast(-9007199254740993, int64))",
            expected: "9007199254740993",
            status: Status::Broken("chelis#680"),
            lanes: Lanes::Both,
            note: "numeric_unop routes int through `op(as_f64()) as i64` \
                   (host_ops.rs:427).",
        },
        "int64",
    );
}

/// `neg` on int64 at the boundary.
#[test]
#[ignore = "chelis#680: numeric_unop routes int through f64. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_neg_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_neg_2p53",
            expr: "neg(cast(9007199254740993, int64))",
            expected: "-9007199254740993",
            status: Status::Broken("chelis#680"),
            lanes: Lanes::Both,
            note: "Same numeric_unop f64 path as abs.",
        },
        "int64",
    );
}

// ===========================================================================
// COMPARISONS: exactness matters for CONTROL FLOW, not just values.
// ===========================================================================

/// Verified: returns `false`. `ordered_compare` (`host_ops.rs:881-893`)
/// compares int scalars via `.as_f64()`, collapsing both operands to the same
/// f64. `compare_eq` immediately above it was fixed to use `.as_i64()` in #387;
/// `ordered_compare` never was.
#[test]
#[ignore = "chelis#680: ordered_compare compares via as_f64; returns false. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_lt_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_lt_2p53",
            expr: "lt(cast(9007199254740992, int64), cast(9007199254740993, int64))",
            expected: "true",
            status: Status::Broken("chelis#680"),
            lanes: Lanes::EvalOnly,
            note: "A wrong comparison changes CONTROL FLOW, not just a value.",
        },
        "bool",
    );
}

#[test]
#[ignore = "chelis#680: ordered_compare compares via as_f64; returns false. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_gt_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_gt_2p53",
            expr: "gt(cast(9007199254740993, int64), cast(9007199254740992, int64))",
            expected: "true",
            status: Status::Broken("chelis#680"),
            lanes: Lanes::EvalOnly,
            note: "Sibling of lt via ordered_compare.",
        },
        "bool",
    );
}

#[test]
#[ignore = "chelis#680: ordered_compare compares via as_f64. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_gte_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_gte_2p53",
            expr: "gte(cast(9007199254740992, int64), cast(9007199254740993, int64))",
            expected: "false",
            status: Status::Broken("chelis#680"),
            lanes: Lanes::EvalOnly,
            note: "2^53 >= 2^53+1 is false; f64 collapse makes it true.",
        },
        "bool",
    );
}

/// `eq` is the control: it was fixed in #387 to compare via `.as_i64()`, so it
/// is already correct. Its presence here documents that the fix pattern exists
/// and was simply not applied to its neighbours.
#[test]
fn int64_eq_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_eq_2p53",
            expr: "eq(cast(9007199254740992, int64), cast(9007199254740993, int64))",
            expected: "false",
            status: Status::Locked,
            lanes: Lanes::EvalOnly,
            note: "Already correct: compare_eq uses .as_i64() (host_ops.rs:844). \
                   Proof the fix pattern existed and was not applied to \
                   ordered_compare next door.",
        },
        "bool",
    );
}

/// Verified: returns the SMALLER operand. `max_elem` routes through
/// `numeric_binop(args, f64::max)` (`eval.rs:843`).
#[test]
#[ignore = "chelis#680: returns the SMALLER operand. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_max_elem_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_max_elem_2p53",
            expr: "max_elem(cast(9007199254740992, int64), cast(9007199254740993, int64))",
            expected: "9007199254740993",
            status: Status::Broken("chelis#680"),
            lanes: Lanes::Both,
            note: "Returns the SMALLER value today: both operands collapse to \
                   the same f64 and f64::max returns the first.",
        },
        "int64",
    );
}

/// `min_elem` passes today BY LUCK: both operands collapse to the same f64, and
/// `min` of two equal values happens to be the expected answer. Kept as a lock
/// and documented so nobody reads the green as evidence the lane is sound.
#[test]
#[ignore = "chelis#691: lane parity fails: C emits fmaxf/fminf on int64_t. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_min_elem_at_mantissa_boundary_is_exact() {
    check_row(
        &Row {
            name: "int64_min_elem_2p53",
            expr: "min_elem(cast(9007199254740992, int64), cast(9007199254740993, int64))",
            expected: "9007199254740992",
            status: Status::Locked,
            lanes: Lanes::Both,
            note: "PASSES BY LUCK: operands collapse to one f64 and min of two \
                   equal values is coincidentally right. Not evidence of \
                   soundness. Its sibling max_elem is wrong on the same inputs.",
        },
        "int64",
    );
}

// ===========================================================================
// CONTROL FLOW: the same program must not take different branches per lane.
// ===========================================================================

/// Verified: eval takes the else branch (222); compiled C takes the then
/// branch (111). This is the most severe observable form of #680: the bug does
/// not merely perturb a value, it selects a different code path.
#[test]
#[ignore = "chelis#680: eval takes else (222), compiled C takes then (111). This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int_condition_selects_the_same_branch_in_every_lane() {
    check_row(
        &Row {
            name: "branch_2p53",
            expr: "if lt(cast(9007199254740992, int64), cast(9007199254740993, int64)) \
                   then cast(111, int64) else cast(222, int64)",
            expected: "111",
            status: Status::Broken("chelis#680"),
            lanes: Lanes::Both,
            note: "eval=222, C=111. A wrong integer comparison changes which \
                   branch executes.",
        },
        "int64",
    );
}

// ===========================================================================
// BITWISE / SHIFT: #682. No large values; C emits the literal 0.
// ===========================================================================

macro_rules! bitwise_row {
    ($fn_name:ident, $label:literal, $expr:literal, $expected:literal) => {
        #[test]
        #[ignore = "chelis#682: the C backend has no match arm for this builtin \
                    and emits `/* unsupported builtin */ 0`, so eval and \
                    compiled C disagree. This test asserts the CORRECT behavior \
                    and fails until the fix lands. Run with `cargo test -p \
                    chelis-cli --test precision_matrix -- --ignored`."]
        fn $fn_name() {
            check_row(
                &Row {
                    name: $label,
                    expr: $expr,
                    expected: $expected,
                    status: Status::Broken("chelis#682"),
                    lanes: Lanes::Both,
                    note: "C backend has no match arm for this builtin \
                           (host_emit.rs) and falls through to \
                           `/* unsupported builtin */ 0`. Not a precision bug.",
                },
                "int64",
            );
        }
    };
}

bitwise_row!(
    bitand_is_correct_in_every_lane,
    "bitand",
    "bitand(cast(12, int64), cast(10, int64))",
    "8"
);
bitwise_row!(
    bitor_is_correct_in_every_lane,
    "bitor",
    "bitor(cast(12, int64), cast(10, int64))",
    "14"
);
bitwise_row!(
    bitxor_is_correct_in_every_lane,
    "bitxor",
    "bitxor(cast(12, int64), cast(10, int64))",
    "6"
);
bitwise_row!(
    shl_is_correct_in_every_lane,
    "shl",
    "shl(cast(1, int64), cast(10, int64))",
    "1024"
);
bitwise_row!(
    shr_is_correct_in_every_lane,
    "shr",
    "shr(cast(1024, int64), cast(3, int64))",
    "128"
);

// ===========================================================================
// LITERALS: verified CLEAN in both lanes. Regression locks.
// ===========================================================================

/// Verified exact in BOTH lanes. The front end (lexer/parser/AST/desugar/
/// formatter/decompiler) carries integer literals as exact i64.
#[test]
fn int64_literal_above_mantissa_boundary_is_exact_in_every_lane() {
    check_row(
        &Row {
            name: "lit_2p53_plus_1",
            expr: "cast(9007199254740993, int64)",
            expected: "9007199254740993",
            status: Status::Locked,
            lanes: Lanes::EvalOnly,
            note: "Front end is exact; corruption starts at the arithmetic ops. \
                   Locks that nobody 'simplifies' literal storage to f64.",
        },
        "int64",
    );
}

/// `i64::MAX` as a literal.
#[test]
fn i64_max_literal_is_exact() {
    check_row(
        &Row {
            name: "lit_i64_max",
            expr: &format!("cast({I64_MAX}, int64)"),
            expected: "9223372036854775807",
            status: Status::Locked,
            lanes: Lanes::EvalOnly,
            note: "i64::MAX round-trips as a literal.",
        },
        "int64",
    );
}

// ===========================================================================
// OVERFLOW: per #680, errors not wraps, at EVERY width.
//
// Today there are TWO different behaviors split by width, neither authored:
//   int8/16/32 WRAP  (ScalarBits::from_i64_as narrows with `as i8`, which
//                     truncates bits)
//   int64      SATURATES (the f64 round-trip's `as i64` saturates)
// The split is decided purely by which Rust `as` cast runs last.
// ===========================================================================

fn assert_traps_with_overflow(expr: &str, label: &str) {
    match eval_lane_str(expr) {
        Ok(v) => panic!(
            "{label}: `{expr}` must trap on integer overflow per chelis#680 \
             (errors, not wraps), but it succeeded and returned {v}"
        ),
        Err(stderr) => assert!(
            stderr.contains("overflow"),
            "{label}: `{expr}` failed but without a branded overflow \
             diagnostic. Got: {stderr}"
        ),
    }
}

/// Verified today: returns i64::MAX (saturates).
#[test]
#[ignore = "chelis#680: saturates to i64::MAX instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_add_overflow_traps() {
    assert_traps_with_overflow(
        &format!("add(cast({I64_MAX}, int64), cast(1, int64))"),
        "int64_add_overflow",
    );
}

/// Verified today: returns i64::MAX. Both operands are in range; only the
/// product overflows, so this cannot be dismissed as a bad literal.
#[test]
#[ignore = "chelis#680: saturates to i64::MAX instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_mul_overflow_traps() {
    assert_traps_with_overflow(
        "mul(cast(4000000000, int64), cast(4000000000, int64))",
        "int64_mul_overflow",
    );
}

/// Verified today: returns -128 (WRAPS, unlike int64 which saturates).
#[test]
#[ignore = "chelis#680: wraps to -128 instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int8_add_overflow_traps() {
    assert_traps_with_overflow("add(cast(127, int8), cast(1, int8))", "int8_add_overflow");
}

/// Verified today: returns -32768 (wraps).
#[test]
#[ignore = "chelis#680: wraps to -32768 instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int16_add_overflow_traps() {
    assert_traps_with_overflow(
        "add(cast(32767, int16), cast(1, int16))",
        "int16_add_overflow",
    );
}

/// Verified today: returns -2147483648 (wraps).
#[test]
#[ignore = "chelis#680: wraps to -2147483648 instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int32_add_overflow_traps() {
    assert_traps_with_overflow(
        "add(cast(2147483647, int32), cast(1, int32))",
        "int32_add_overflow",
    );
}

/// Negative parity: a computation that does NOT overflow must still succeed.
/// Guards against a fix that traps too eagerly.
#[test]
fn int64_add_just_below_overflow_does_not_trap() {
    check_row(
        &Row {
            name: "int64_add_no_overflow",
            expr: &format!("add(cast({}, int64), cast(1, int64))", I64_MAX - 1),
            expected: "9223372036854775807",
            status: Status::Locked,
            lanes: Lanes::EvalOnly,
            note: "i64::MAX-1 + 1 == i64::MAX exactly, and must NOT trap. \
                   Negative parity for the overflow trap.",
        },
        "int64",
    );
}

/// Negative parity at a narrow width.
#[test]
fn int8_add_just_below_overflow_does_not_trap() {
    check_row(
        &Row {
            name: "int8_add_no_overflow",
            expr: "add(cast(126, int8), cast(1, int8))",
            expected: "127",
            status: Status::Locked,
            lanes: Lanes::EvalOnly,
            note: "126 + 1 == 127 fits int8; must not trap.",
        },
        "int8",
    );
}

// ===========================================================================
// TENSOR STORAGE: #684. Lossy at rest in eval; exact in compiled C.
// ===========================================================================

/// Verified lane divergence with NO arithmetic at all:
///   eval -> [9007199254740992]   (IrTensorValue.data is Vec<f64>)
///   C    -> [9007199254740993]   (C runtime has dtype-tagged storage)
#[test]
#[ignore = "chelis#684: eval [9007199254740992], compiled C [9007199254740993]. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_tensor_round_trip_is_exact_in_every_lane() {
    let expr = "to_list(to_tensor([cast(9007199254740993, int64)]))";
    let eval_got = eval_lane_str(expr).expect("eval lane");
    if c_toolchain_available() {
        let c_got = c_lane_str(expr, "List[int64]", "tensor_rt").expect("c lane");
        assert_eq!(
            eval_got, c_got,
            "LANE DIVERGENCE on a pure to_tensor/to_list round-trip with no \
             arithmetic: eval={eval_got}, C={c_got}. chelis#684"
        );
    }
    assert_eq!(
        eval_got, "[9007199254740993]",
        "to_list(to_tensor([2^53+1])) must round-trip exactly. eval stores \
         tensor data as Vec<f64> (crates/chelis-ir/src/eval.rs:13). chelis#684"
    );
}

/// i64::MAX is a BAD round-trip probe and passes BY LUCK: f64 rounds it UP to
/// 2^63, then the saturating `as i64` clamps it back DOWN to i64::MAX. Two
/// errors cancel.
///
/// Locked deliberately, with the coincidence documented, so nobody uses
/// i64::MAX as evidence that tensor storage is exact. Use 2^53+1 (above).
#[test]
fn i64_max_tensor_round_trip_passes_by_luck_not_by_correctness() {
    let expr = &format!("to_list(to_tensor([cast({I64_MAX}, int64)]))");
    let got = eval_lane_str(expr).expect("eval lane");
    assert_eq!(
        got, "[9223372036854775807]",
        "i64::MAX round-trips through Vec<f64> storage only because f64 rounds \
         UP to 2^63 and the saturating cast clamps back DOWN to i64::MAX. This \
         green is a coincidence, NOT evidence of exact storage. See \
         int64_tensor_round_trip_is_exact_in_every_lane for the real probe."
    );
}

// ===========================================================================
// BINDING FORMS: where the value LIVES decides whether it survives (#684).
//
// A top-level binding promotes an int64 scalar to a RANK-0 TENSOR, which lands
// in the Vec<f64> storage and loses the value. A `def` body stays on the exact
// scalar lane. So the same literal is exact or corrupted depending purely on
// the syntactic form that holds it, with no cast and no arithmetic involved.
//
// This is the highest-reachability form of #684: every file under `examples/`
// is written with top-level bindings.
// ===========================================================================

/// Run a whole program (not a single expression) and return the first printed
/// line. Needed because these cases are ABOUT the binding form, so the program
/// shape is the thing under test.
fn eval_program_first_line(program: &str) -> Result<String, String> {
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

/// The control. A bare expression never becomes a tensor and is exact today.
#[test]
fn bare_expression_keeps_int64_exact() {
    let got =
        eval_program_first_line("module M.Main\nout = print(cast(9007199254740993, int64))\n")
            .expect("eval");
    assert_eq!(
        got, "9007199254740993",
        "a bare int64 expression must be exact (control for the binding cases)"
    );
}

/// Verified: prints `tensor(shape=[], data=[9007199254740992.0])`. An
/// UNannotated top-level binding is enough to promote and corrupt.
#[test]
#[ignore = "chelis#684: binding promotes the scalar to a rank-0 f64 tensor. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn unannotated_top_level_binding_keeps_int64_exact() {
    let got = eval_program_first_line(
        "module M.Main\nx = cast(9007199254740993, int64)\nout = print(x)\n",
    )
    .expect("eval");
    assert_eq!(
        got, "9007199254740993",
        "an int64 top-level binding must stay an exact int64 scalar. It is \
         currently promoted to a rank-0 tensor backed by Vec<f64> and prints as \
         `tensor(shape=[], data=[...])`. chelis#684"
    );
}

/// Verified: identical corruption to the unannotated form. Proves the promotion
/// is caused by the BINDING, not by the type annotation.
#[test]
#[ignore = "chelis#684: binding promotes the scalar to a rank-0 f64 tensor. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn annotated_top_level_binding_keeps_int64_exact() {
    let got = eval_program_first_line(
        "module M.Main\nx: int64 = cast(9007199254740993, int64)\nout = print(x)\n",
    )
    .expect("eval");
    assert_eq!(
        got, "9007199254740993",
        "annotating `: int64` must not change the value. Corrupts identically \
         to the unannotated form, so the promotion is the binding, not the \
         annotation. chelis#684"
    );
}

/// Verified EXACT today. A `def` body stays on the scalar lane. Locked as the
/// contrast case that localises #684 to the binding path.
#[test]
fn def_body_keeps_int64_exact() {
    let got = eval_program_first_line(
        "module M.Main\ndef f() -> int64 = cast(9007199254740993, int64)\nout = print(f())\n",
    )
    .expect("eval");
    assert_eq!(
        got, "9007199254740993",
        "a def body stays on the exact scalar lane; this is why the same value \
         is exact here and corrupted in a top-level binding"
    );
}

/// Fan-out: binding used twice (auto-copy per the implicit-linearity rules).
/// Both reads must agree with each other and with the true value.
#[test]
#[ignore = "chelis#684: binding promotes the scalar to a rank-0 f64 tensor. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_binding_fanout_keeps_both_reads_exact_and_equal() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("f.ch");
    write_file(
        &path,
        "module M.Main\nx = cast(9007199254740993, int64)\na = print(x)\nb = print(x)\n",
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(out.status.success(), "fan-out program should evaluate");
    let stdout = String::from_utf8_lossy(&out.stdout);
    let reads: Vec<&str> = stdout.lines().take(2).map(str::trim).collect();
    assert_eq!(
        reads[0], reads[1],
        "the two fan-out reads of one binding must agree with each other"
    );
    assert_eq!(
        reads[0], "9007199254740993",
        "fan-out (auto-copy) of an int64 binding must preserve the value. \
         chelis#684"
    );
}

// ===========================================================================
// LINEARITY SURFACE: copy/borrow are tensor-only. Loud rejections, locked so a
// future change does not quietly make them another silent f64 path.
// ===========================================================================

/// `copy(x)` on an int64 scalar is rejected at check time:
///   "copy requires tensor input, got int64"
/// Locked as a LOUD failure. If copy ever accepts scalars it must not route
/// them through f64.
#[test]
fn copy_of_int64_scalar_is_rejected_loudly() {
    let err = eval_program_first_line(
        "module M.Main\nx = cast(9007199254740993, int64)\ny = copy(x)\nout = print(y)\n",
    )
    .expect_err("copy of an int64 scalar should be rejected");
    assert!(
        err.contains("copy requires tensor input"),
        "copy(int64) must fail loudly with a clear diagnostic, got: {err}"
    );
}

/// `&x` on an int64 scalar is rejected at check time:
///   "borrow requires tensor or tensor-carrying input, got int64"
#[test]
fn borrow_of_int64_scalar_is_rejected_loudly() {
    let err = eval_program_first_line(
        "module M.Main\n\
         def ident(v: int64) -> int64 = v\n\
         x = cast(9007199254740993, int64)\n\
         out = print(ident(&x))\n",
    )
    .expect_err("borrow of an int64 scalar should be rejected");
    assert!(
        err.contains("borrow requires tensor"),
        "&int64 must fail loudly with a clear diagnostic, got: {err}"
    );
}

// ===========================================================================
// C BACKEND emits float32 math for int64 tensors (#691).
//
// Verified by compiling and running: the C backend types the pointers
// correctly as `int64_t*` but calls `fmaxf`, a float32 function:
//
//     int64_t* restrict __out_2 = (int64_t*)t2->data;
//     const int64_t* restrict __in_a_2 = (const int64_t*)t0->data;
//     __out_2[i] = fmaxf(__in_a_2[i], __in_b_2[i]);
//
// `double_math_fn` (crates/chelis-backend-c/src/emit.rs:1352-1370) remaps
// float->double only when `is_f64` (`:1344-1346`, `Prim::F64` only), so an
// Int64 operand keeps the float32 function.
//
// Direction matters: here EVAL is right and the COMPILED BACKEND is wrong -
// the opposite of #680. A fix must make both lanes exact, not just move the
// error to the other lane.
//
// Reachability: this needs a `def`, which is the ordinary way to write
// Chelis. A `def` takes the permissive host-lane precision gate
// (`reject_unsupported_c_precisions_host`, crates/chelis-cli/src/main.rs:
// 7054-7162), which admits int64 generically. Only a `def`-free program hits
// the strict bare-DAG gate (`reject_unsupported_c_precisions`, `:7174-7214`)
// that rejects int64 tensors. Both lanes emit through the same `CEmitter`.
// ===========================================================================

/// Verified: eval returns `[16777217, 1, 2, 3]`; compiled C returns
/// `[16777216.0, 1.0, 2.0, 3.0]`.
///
/// `16777217` is `2^24 + 1`: not representable in **float32**, trivially
/// representable in int64. The threshold is 2^24, not 2^53, so this is far
/// easier to reach than #680. Note the compiled output even renders as floats
/// for a `tensor[4, int64]`.
#[test]
#[ignore = "chelis#691: C backend emits fmaxf (float32) for int64 max_elem; \
            compiled C returns 16777216 where eval returns 16777217. This test \
            asserts the CORRECT behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test precision_matrix -- --ignored`."]
fn int64_max_elem_tensor_agrees_across_lanes_at_f32_boundary() {
    let program = "module M.Main\n\
         def pick(a: tensor[4, int64], b: tensor[4, int64]) -> tensor[4, int64] = max_elem(a, b)\n\
         out = print(to_list(pick(\
           to_tensor([cast(16777217, int64), cast(1, int64), cast(2, int64), cast(3, int64)]), \
           to_tensor([cast(1, int64), cast(1, int64), cast(1, int64), cast(1, int64)]))))\n";
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mx.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(out.status.success(), "eval lane should succeed");
    let eval_got = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    assert_eq!(
        eval_got, "[16777217, 1, 2, 3]",
        "eval lane must be exact for int64 max_elem"
    );

    if !c_toolchain_available() {
        eprintln!("skipping compiled lane: no host C toolchain");
        return;
    }
    let cdir = tempdir().expect("tempdir");
    let cpath = cdir.path().join("mx.ch");
    let cout = cdir.path().join("mx-out");
    write_file(
        &cpath,
        "def pick(a: tensor[4, int64], b: tensor[4, int64]) -> tensor[4, int64] = max_elem(a, b)\n\
         out = pick(\
           to_tensor([cast(16777217, int64), cast(1, int64), cast(2, int64), cast(3, int64)]), \
           to_tensor([cast(1, int64), cast(1, int64), cast(1, int64), cast(1, int64)]))\n",
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            cpath.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            cout.to_str().unwrap(),
        ])
        .assert()
        .success();
    let emitted = std::fs::read_to_string(cout.join("mx.c")).expect("emitted C");
    assert!(
        !emitted.contains("fmaxf"),
        "C backend emitted `fmaxf` (a float32 function) for an int64 max_elem. \
         `double_math_fn` remaps float->double only when is_f64, so Int64 keeps \
         the float32 function and both operands are narrowed to a 24-bit \
         mantissa. chelis#691"
    );
}

// ===========================================================================
// REFUTATION LOCKS: probes that found NOTHING.
//
// During the #680 audit, three claims derived from reading code (and code
// comments) were refuted by running it. Each is locked here, because a
// refuted claim is exactly as valuable as a confirmed one: it stops the next
// person re-deriving it from the same misleading source and scoping a fix
// around a bug that does not exist.
//
// These pass today. They are locks, not bug reports.
// ===========================================================================

/// REFUTES: "int32 tensors lose precision above 2^24 because the C runtime
/// stores them as f32."
///
/// Source of the claim: `crates/chelis-runtime/src/lib.rs:144-152`, verbatim:
///
/// > `CHELIS_I32` and `CHELIS_BOOL` tensors still store data as 4-byte f32 bit
/// > patterns [...] the trait impl for `i32` exists but reads i32 bytes, which
/// > is the wrong decode for the current f32-encoded storage convention [...]
/// > A future §5 follow-on migrates `CHELIS_I32` and `CHELIS_BOOL`.
///
/// That predicts `16777217` (2^24+1) corrupts to `16777216`. It does not:
/// verified exact in BOTH the eval and compiled-C lanes. Either the comment is
/// stale or this path does not touch the f32-encoded slot.
///
/// The comment is tracked by chelis#694. This test pins the actual behavior so
/// nobody "fixes" a bug the code does not have.
#[test]
fn int32_tensor_round_trip_is_exact_above_the_f32_boundary() {
    let expr = "to_list(to_tensor([cast(16777217, int32)]))";
    let eval_got = eval_lane_str(expr).expect("eval lane");
    assert_eq!(
        eval_got, "[16777217]",
        "int32 tensor round-trip must be exact at 2^24+1. If this fails, the \
         claim at crates/chelis-runtime/src/lib.rs:144-152 has become true and \
         chelis#694 needs revisiting."
    );
    if c_toolchain_available() {
        let c_got = c_lane_str(expr, "List[int32]", "i32_rt").expect("c lane");
        assert_eq!(
            c_got, eval_got,
            "int32 tensor round-trip must agree across lanes"
        );
    }
}

/// REFUTES: "the C backend corrupts int64 literals above 2^53 via
/// `RiscOp::Const { value: f64 }`."
///
/// The claim: `crates/chelis-ir/src/dag.rs:883` really is `Const { value: f64 }`
/// and `crates/chelis-ir/src/lower.rs:4841` really does `*n as f64`, so a
/// literal above 2^53 should be rounded before either lane sees it.
///
/// It is not. That is the DAG **tensor** lane, which rejects int64 outright;
/// int64 scalars flow through the host lane and are exact. Verified: the
/// emitted C contains `__arg0_1 = 9007199254740993;` and the binary prints it.
///
/// This matters for scoping: the corruption in #680 begins at the arithmetic
/// ops, NOT at the literal. A fix aimed at literal lowering would be aimed at
/// the wrong place.
#[test]
fn int64_literal_above_mantissa_boundary_is_exact_in_the_compiled_lane() {
    let expr = "add(cast(9007199254740993, int64), cast(0, int64))";
    let eval_got = eval_lane_str(expr).expect("eval lane");
    // eval corrupts this via the ADD (chelis#680), not via the literal - see
    // int64_literal_above_mantissa_boundary_is_exact_in_every_lane for the
    // literal-only probe, which is exact in eval too.
    let _ = eval_got;
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let c_got = c_lane_str(expr, "int64", "lit_c").expect("c lane");
    assert_eq!(
        c_got, "9007199254740993",
        "the compiled lane must carry an int64 literal above 2^53 exactly. \
         The DAG-lane `RiscOp::Const {{ value: f64 }}` path does not apply to \
         int64 scalars, which use the host lane."
    );
}

/// The int64 scalar `abs` host-lane path in the compiled backend is EXACT,
/// even though the eval lane is wrong on the same input (chelis#680) and the
/// DAG-lane `fabsf` arm is wrong for tensors (chelis#691).
///
/// Locked to keep the three cases distinct: same operation, three lanes, three
/// different verdicts. A fix must not collapse them by accident.
#[test]
fn int64_scalar_abs_is_exact_in_the_compiled_host_lane() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let c_got =
        c_lane_str("abs(cast(-9007199254740993, int64))", "int64", "abs_host").expect("c lane");
    assert_eq!(
        c_got, "9007199254740993",
        "compiled int64 scalar abs goes through the host lane and is exact"
    );
}

// ===========================================================================
// Std.Decimal: the surface that started this investigation.
//
// The original report was "Decimal doesn't work". It does: every one of these
// is exact today, and they are the cases a user actually writes. Locked so the
// #680 fix cannot regress them, and so the real defect stays correctly scoped
// (Decimal's own logic is sound; it is a victim of the evaluator's f64
// arithmetic, not the cause).
//
// These need the staged chelis-std reef fixture, so they carry the same
// manual-gate ignore as the rest of the std-dependent corpus.
// ===========================================================================

/// The classic fixed-point traps, all exact today. `0.1 + 0.2` renders `0.3`,
/// where f64 gives `0.30000000000000004`.
#[test]
#[ignore = "needs the staged chelis-std reef fixture; exceeds the inner-loop \
            budget. Run with `cargo test -p chelis-cli --test precision_matrix \
            -- --ignored`."]
fn decimal_classic_float_traps_are_exact() {
    let (_dir, reef_home, app_pkg) = common::make_app("precision-decimal-traps");
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\n\
         import Std.Decimal (decimal, decimal_add, decimal_sub, decimal_mul, \
         decimal_to_string, decimal_from_int)\n\
         a = print(decimal_to_string(decimal_add(decimal(\"0.1\"), decimal(\"0.2\"))))\n\
         b = print(decimal_to_string(decimal_sub(decimal(\"1.00\"), decimal(\"0.90\"))))\n\
         c = print(decimal_to_string(decimal_mul(decimal(\"1.1\"), decimal(\"1.1\"))))\n\
         d = print(decimal_to_string(decimal_mul(decimal(\"19.99\"), \
         decimal_from_int(cast(3, int64)))))\n",
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "decimal traps program should evaluate"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let got: Vec<&str> = stdout.lines().take(4).map(str::trim).collect();
    // f64 would give 0.30000000000000004, 0.09999999999999998, 1.2100000000000002.
    assert_eq!(got[0], "0.3", "0.1 + 0.2 must be exactly 0.3");
    assert_eq!(got[1], "0.1", "1.00 - 0.90 must be exactly 0.1");
    assert_eq!(got[2], "1.21", "1.1 * 1.1 must be exactly 1.21");
    assert_eq!(got[3], "59.97", "19.99 * 3 must be exactly 59.97");
}

/// All four rounding modes on the tie case `5 / 2`, including banker's
/// rounding. Exact today.
#[test]
#[ignore = "needs the staged chelis-std reef fixture; exceeds the inner-loop \
            budget. Run with `cargo test -p chelis-cli --test precision_matrix \
            -- --ignored`."]
fn decimal_rounding_modes_are_correct_on_the_tie_case() {
    let (_dir, reef_home, app_pkg) = common::make_app("precision-decimal-rounding");
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\n\
         import Std.Decimal (decimal, decimal_div, decimal_to_string, \
         round_half_up, round_half_even, round_down, round_up)\n\
         a = print(decimal_to_string(decimal_div(decimal(\"5\"), decimal(\"2\"), \
         cast(0, int64), round_half_up())))\n\
         b = print(decimal_to_string(decimal_div(decimal(\"5\"), decimal(\"2\"), \
         cast(0, int64), round_half_even())))\n\
         c = print(decimal_to_string(decimal_div(decimal(\"5\"), decimal(\"2\"), \
         cast(0, int64), round_down())))\n\
         d = print(decimal_to_string(decimal_div(decimal(\"5\"), decimal(\"2\"), \
         cast(0, int64), round_up())))\n\
         e = print(decimal_to_string(decimal_div(decimal(\"-5\"), decimal(\"2\"), \
         cast(0, int64), round_half_up())))\n",
    );
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args([
            "eval",
            "--file",
            app_pkg.join("src/main.ch").to_str().unwrap(),
        ])
        .output()
        .expect("chelis eval should run");
    assert!(
        out.status.success(),
        "decimal rounding program should evaluate"
    );
    let stdout = String::from_utf8_lossy(&out.stdout);
    let got: Vec<&str> = stdout.lines().take(5).map(str::trim).collect();
    assert_eq!(got[0], "3", "5/2 half-up == 3");
    assert_eq!(got[1], "2", "5/2 half-even == 2 (banker's)");
    assert_eq!(got[2], "2", "5/2 down == 2");
    assert_eq!(got[3], "3", "5/2 up == 3");
    assert_eq!(got[4], "-3", "-5/2 half-up == -3 (away from zero)");
}

// ===========================================================================
// SILENT ZERO PLACEHOLDERS (#699).
//
// A different failure class from precision: `lower_transcendental`
// (crates/chelis-ir/src/lower.rs:9834-9845) replaces a non-float operand with
// a zero constant and DROPS the operand:
//
//     if input_prec.is_float() {
//         self.dag.add_node(op, vec![x], out_ty, ...)
//     } else {
//         // Non-float input: produce a zero constant as error placeholder.
//         self.dag.add_node(RiscOp::Const { value: 0.0 }, vec![], out_ty, ...)
//     }
//
// The comment says "error placeholder" but no error is raised. `abs`, `floor`,
// `ceil` and `round` are routed through this helper (`lower.rs:6930` and
// siblings) yet are NOT in `TRANSCENDENTAL_FLOAT_ONLY_OPS`
// (crates/chelis-types/src/infer.rs:4192-4204), so they are well-typed on
// integer tensors, fail `is_float()`, and silently compile to zeros in EVERY
// compiled backend. The evaluator is correct, so the lanes disagree.
//
// Values here are 100-400. This has nothing to do with precision boundaries.
//
// Same design mistake as #682's `/* unsupported builtin */ 0`: unimplemented
// must fail the build, not evaluate to zero.
// ===========================================================================

/// Assert an int64-tensor unary op agrees across lanes.
///
/// Verified today: eval is correct, compiled C returns all zeros.
fn assert_int_tensor_unop_parity(op: &str, expected: &str, name: &str) {
    let eval_program = format!(
        "module M.Main\n\
         def run(x: tensor[4, int64]) -> tensor[4, int64] = {op}(x)\n\
         out = print(to_list(run(to_tensor([cast(-100, int64), cast(200, int64), \
         cast(-300, int64), cast(400, int64)]))))\n"
    );
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("u.ch");
    write_file(&path, &eval_program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    assert!(out.status.success(), "{name}: eval lane should succeed");
    let eval_got = String::from_utf8_lossy(&out.stdout)
        .lines()
        .next()
        .unwrap_or("")
        .trim()
        .to_string();
    assert_eq!(
        eval_got, expected,
        "{name}: eval lane must be correct for `{op}` on an int64 tensor"
    );

    if !c_toolchain_available() {
        eprintln!("skipping compiled lane for {name}: no host C toolchain");
        return;
    }
    let cdir = tempdir().expect("tempdir");
    let cpath = cdir.path().join(format!("{name}.ch"));
    let cout = cdir.path().join(format!("{name}-out"));
    write_file(
        &cpath,
        &format!(
            "def run(x: tensor[4, int64]) -> tensor[4, int64] = {op}(x)\n\
             out = run(to_tensor([cast(-100, int64), cast(200, int64), \
             cast(-300, int64), cast(400, int64)]))\n"
        ),
    );
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            cpath.to_str().unwrap(),
            "--target",
            "c",
            "--output",
            cout.to_str().unwrap(),
        ])
        .assert()
        .success();
    let emitted = std::fs::read_to_string(cout.join(format!("{name}.c"))).expect("emitted C");
    assert!(
        !emitted.contains("chelis_fill_i64(t0, (int64_t)0)"),
        "{name}: `{op}` on an int64 tensor lowered to a zero constant with the \
         operand dropped. lower_transcendental's non-float branch emits \
         `RiscOp::Const {{ value: 0.0 }}` with an empty input list and raises no \
         error, so the compiled program silently returns zeros while eval \
         returns {expected}. chelis#699"
    );
}

/// Verified: eval `[100, 200, 300, 400]`, compiled C `[0.0, 0.0, 0.0, 0.0]`.
#[test]
#[ignore = "chelis#699: abs on an int64 tensor silently compiles to zeros \
            (lower_transcendental's non-float 'error placeholder' raises no \
            error). This test asserts the CORRECT behavior and fails until the \
            fix lands. Run with `cargo test -p chelis-cli --test \
            precision_matrix -- --ignored`."]
fn int64_tensor_abs_agrees_across_lanes() {
    assert_int_tensor_unop_parity("abs", "[100, 200, 300, 400]", "abs_i64");
}

/// `floor` is identity on integers. Verified: compiled C returns zeros.
#[test]
#[ignore = "chelis#699: floor on an int64 tensor silently compiles to zeros. \
            Run with `cargo test -p chelis-cli --test precision_matrix -- \
            --ignored`."]
fn int64_tensor_floor_agrees_across_lanes() {
    assert_int_tensor_unop_parity("floor", "[-100, 200, -300, 400]", "floor_i64");
}

/// `ceil` is identity on integers. Verified: compiled C returns zeros.
#[test]
#[ignore = "chelis#699: ceil on an int64 tensor silently compiles to zeros. \
            Run with `cargo test -p chelis-cli --test precision_matrix -- \
            --ignored`."]
fn int64_tensor_ceil_agrees_across_lanes() {
    assert_int_tensor_unop_parity("ceil", "[-100, 200, -300, 400]", "ceil_i64");
}

/// `round` is identity on integers. Verified: compiled C returns zeros.
#[test]
#[ignore = "chelis#699: round on an int64 tensor silently compiles to zeros. \
            Run with `cargo test -p chelis-cli --test precision_matrix -- \
            --ignored`."]
fn int64_tensor_round_agrees_across_lanes() {
    assert_int_tensor_unop_parity("round", "[-100, 200, -300, 400]", "round_i64");
}

/// The zeroed subtree feeds the rest of the computation, which proceeds
/// correctly on the wrong operand and yields PLAUSIBLE output.
///
/// Verified: `add(abs(x), [1,1,1,1])` compiles to `[1.0, 1.0, 1.0, 1.0]` -
/// the `add` computed `0 + 1 = 1` perfectly correctly on a zeroed operand.
/// No crash, no NaN, no absurd magnitude. This is the property that makes the
/// placeholder worse than an unimplemented-op panic: it is not self-announcing.
#[test]
#[ignore = "chelis#699: the zeroed abs subtree poisons downstream arithmetic \
            and yields plausible output ([1,1,1,1] instead of [101,201,301,401]). \
            Run with `cargo test -p chelis-cli --test precision_matrix -- \
            --ignored`."]
fn zeroed_abs_does_not_silently_poison_downstream_arithmetic() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("poison.ch");
    let out_dir = dir.path().join("poison-out");
    write_file(
        &path,
        "def run(x: tensor[4, int64]) -> tensor[4, int64] = \
         add(abs(x), to_tensor([cast(1, int64), cast(1, int64), cast(1, int64), \
         cast(1, int64)]))\n\
         out = run(to_tensor([cast(-100, int64), cast(200, int64), \
         cast(-300, int64), cast(400, int64)]))\n",
    );
    Command::cargo_bin("chelis")
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
        .assert()
        .success();
    let status = common::link_generated(&out_dir, "poison.c", "poison");
    assert!(status.success(), "link failed: {status}");
    let run = std::process::Command::new(out_dir.join("poison"))
        .output()
        .expect("compiled binary should run");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        !stdout.contains("[1.0, 1.0, 1.0, 1.0]"),
        "add(abs(x), [1,1,1,1]) returned [1,1,1,1]: the abs subtree was zeroed \
         and the add computed 0+1 correctly on it. Expected [101, 201, 301, 401]. \
         Note the output is PLAUSIBLE, not obviously broken. chelis#699. \
         Got: {stdout}"
    );
}

/// f32 `abs` must stay correct: it satisfies `is_float()` and takes the real
/// lowering path. Locked so the #699 fix is aimed at the right branch, and to
/// prove the trigger is the `is_float()` guard rather than `abs` generally.
#[test]
fn f32_tensor_abs_is_correct_and_unaffected_by_the_placeholder() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("fabs.ch");
    let out_dir = dir.path().join("fabs-out");
    write_file(
        &path,
        "def run(y: tensor[4, f32]) -> tensor[4, f32] = abs(y)\n\
         out = run(to_tensor([-1.0, 2.0, -3.0, 4.0]))\n",
    );
    Command::cargo_bin("chelis")
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
        .assert()
        .success();
    let status = common::link_generated(&out_dir, "fabs.c", "fabs");
    assert!(status.success(), "link failed: {status}");
    let run = std::process::Command::new(out_dir.join("fabs"))
        .output()
        .expect("compiled binary should run");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        stdout.contains("1.0") && stdout.contains("2.0") && stdout.contains("3.0"),
        "f32 abs must be correct (it satisfies is_float() and takes the real \
         lowering path). Got: {stdout}"
    );
    assert!(
        !stdout.contains("[0.0, 0.0, 0.0, 0.0]"),
        "f32 abs must not be zeroed: got {stdout}"
    );
}

// ===========================================================================
// COMPILE-TIME BRANCH DELETION (#711).
//
// `fold_static_cond` (crates/chelis-ir/src/lower.rs:9095-9178) evaluates `if`
// conditions as f64 at lowering time, including integer comparisons, and
// `lower_if` (`:10333`) then lowers ONLY the taken branch:
//
//     if let Some(taken) = self.fold_static_cond(cond) {
//         return self.lower_expr(if taken { then_expr } else { else_expr });
//     }
//
// So a wrong fold does not merely miscompute a value - it REMOVES the untaken
// branch from the program. That makes this the most severe shape in the class:
// every other instance produces a wrong value a corrected runtime would fix;
// this one deletes code, and no downstream stage can recover it.
//
// The fold only fires when the condition is a pure `Const` DAG. `cast(...)`
// introduces a non-Const node and the fold declines, which is why an earlier
// probe using casts saw both branches and wrongly concluded no fold occurred.
// Bare suffixed literals (`9007199254740992i64`) keep it foldable.
//
// Note `fold_static_size` (`lower.rs:9032-9069`), ~900 lines earlier in the
// SAME file, folds the identical operators with `checked_i64` and correctly
// declines on overflow. The right implementation exists next door.
// ===========================================================================

/// Verified: the emitted C contains `chelis_fill_f32_bits(t0, 0x435e0000u)`
/// (= 222.0, the else branch) and `111.0`'s bit pattern `0x42de0000` appears
/// NOWHERE in the file. `2^53 < 2^53 + 1` is true, so the answer is 111.0.
#[test]
#[ignore = "chelis#711: fold_static_cond folds the int64 comparison in f64, \
            selects the wrong branch, and DELETES the then-branch at compile \
            time. This test asserts the CORRECT behavior and fails until the \
            fix lands. Run with `cargo test -p chelis-cli --test \
            precision_matrix -- --ignored`."]
fn static_int_condition_does_not_delete_the_correct_branch() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("fold.ch");
    let out_dir = dir.path().join("fold-out");
    write_file(
        &path,
        "def pick() -> f32 = \
         if lt(9007199254740992i64, 9007199254740993i64) then 111.0 else 222.0\n\
         out = pick()\n",
    );
    Command::cargo_bin("chelis")
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
        .assert()
        .success();
    let emitted = std::fs::read_to_string(out_dir.join("fold.c")).expect("emitted C");
    // 111.0 == 0x42de0000, 222.0 == 0x435e0000 as f32 bit patterns. Compare the
    // BITS, not the decimal text: the decimal `111` also occurs inside an
    // unrelated hash constant (0x94D049BB133111EBULL), which is exactly how an
    // earlier probe of this bug fooled itself.
    let has_correct = emitted.contains("0x42de0000");
    let has_wrong = emitted.contains("0x435e0000");
    assert!(
        has_correct || !has_wrong,
        "fold_static_cond selected the WRONG branch and deleted the correct one. \
         `if lt(2^53, 2^53+1) then 111.0 else 222.0` must yield 111.0 \
         (0x42de0000), but the emitted C contains only 222.0 (0x435e0000). The \
         then-branch was not miscomputed, it was REMOVED, so no runtime fix can \
         recover it. chelis#711"
    );
}

// ===========================================================================
// pad_sequences narrows int64 to int32 in the compiled lane (#713).
// ===========================================================================

/// Verified: eval returns `3000000000.0`; compiled C returns `2147483647.0`
/// (= i32::MAX). `chelis_pad_sequences` (crates/chelis-runtime/src/lib.rs:
/// 2318-2330) allocates a `CHELIS_I32` output whenever the pad value is int64,
/// and writes elements `i64 -> f64 -> i32`. The declared return type here is
/// `tensor[2, 2, int64]`.
#[test]
#[ignore = "chelis#713: pad_sequences allocates an int32 output for int64 \
            input, so a token id above i32::MAX saturates in the compiled lane \
            while eval is exact. Run with `cargo test -p chelis-cli --test \
            precision_matrix -- --ignored`."]
fn pad_sequences_preserves_int64_ids_above_i32_max() {
    let eval_expr = "pad_sequences([[cast(3000000000, int64), cast(1, int64)], \
                     [cast(2, int64)]], cast(0, int64))";
    let eval_got = eval_lane_str(eval_expr).expect("eval lane");
    assert!(
        eval_got.contains("3000000000"),
        "eval must preserve a token id above i32::MAX; got {eval_got}"
    );

    if !c_toolchain_available() {
        eprintln!("skipping compiled lane: no host C toolchain");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pads.ch");
    let out_dir = dir.path().join("pads-out");
    write_file(
        &path,
        "def run(rows: List[List[int64]]) -> tensor[2, 2, int64] = \
         pad_sequences(rows, cast(0, int64))\n\
         out = run([[cast(3000000000, int64), cast(1, int64)], [cast(2, int64)]])\n",
    );
    Command::cargo_bin("chelis")
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
        .assert()
        .success();
    let status = common::link_generated(&out_dir, "pads.c", "pads");
    assert!(status.success(), "link failed: {status}");
    let run = std::process::Command::new(out_dir.join("pads"))
        .output()
        .expect("compiled binary should run");
    let stdout = String::from_utf8_lossy(&run.stdout);
    assert!(
        !stdout.contains("2147483647"),
        "pad_sequences saturated a token id of 3000000000 to i32::MAX in the \
         compiled lane while eval returned it exactly. The declared return type \
         is tensor[2, 2, int64]. chelis#713. Got: {stdout}"
    );
}
