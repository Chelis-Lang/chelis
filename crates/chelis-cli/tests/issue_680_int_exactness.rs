//! Chelis-Lang/chelis#680 - integer arithmetic exactness and cross-lane parity.
//!
//! ## Why this file exists
//!
//! The evaluator computes integer scalar arithmetic in `f64` and casts back
//! with a saturating `as i64`
//! (`crates/chelis-compiler-api/src/runtime/host_ops.rs:362`, in
//! `dispatch_scalar_binop`):
//!
//! ```ignore
//! // Mirror pre-WS-A0 behavior: integer scalar ops compute the
//! // result in f64 and truncate.
//! let value = op(lb.as_f64(), rb.as_f64()) as i64;
//! ```
//!
//! `add`/`sub`/`mul` are declared as `f64` lambdas
//! (`crates/chelis-compiler-api/src/runtime/eval.rs:820-822`) and the shared
//! helpers `numeric_binop` / `numeric_unop` / `tensor_numeric_binop` are all
//! typed `Fn(f64, f64) -> f64`. Two defects fall out of that one line:
//!
//!   * precision loss above `2^53` (the f64 mantissa is 53 bits), and
//!   * saturation at `i64::MAX` (Rust's `f64 as i64` is a saturating cast).
//!
//! The C backend lowers `RiscOp::Add` to a native `+` on `int64_t`
//! (`crates/chelis-backend-c/src/emit.rs:441`), so it is exact for scalars.
//! The two lanes therefore disagree on the same program. Cross-lane parity is
//! the load-bearing invariant here, and it is what the tests below lock down.
//!
//! ## Why the existing oracles never caught it
//!
//! `crates/chelis-e2e/tests/eval_agreement.rs` is the eval-vs-C agreement
//! harness, but it is itself f64-typed and tolerance-based (`eval_last` ->
//! `f64`, `parse_c_output` -> `f64`, `assert_close(a, b, tol)`), so
//! `assert_close(9007199254740992.0, 9007199254740993.0, tol)` passes
//! trivially. An exact integer lane did not exist. This file supplies one.
//!
//! `packages/chelis-std/tests/decimal.ch` is likewise entirely in-range, and
//! runs under `chelis test` (the lossy lane), so `Std.Decimal` reports green
//! while silently returning wrong money answers.
//!
//! ## This is the third local fix of one systemic bug
//!
//! The same "integer represented as float" defect has been fixed twice before
//! without ever fixing the class:
//!
//!   1. the C runtime typed `chelis_tensor.data` as `*mut f32` regardless of
//!      dtype, causing four silent-corruption bugs; fixed with the
//!      `TensorElement` trait (see `crates/chelis-e2e/tests/dtype_op_matrix.rs`),
//!   2. integer `div`/`mod` round-tripped through f64 and yielded `i64::MAX`;
//!      fixed for those four ops with `checked_int_binop` (#387),
//!   3. `add`/`sub`/`mul`/`neg`/`abs`/`lt`/`gt`/`max_elem`/reductions were
//!      left on the f64 path (this issue).
//!
//! Every assertion below states an exactness or parity invariant rather than a
//! tolerance, so the class stays locked once fixed.
//!
//! ## Overflow contract
//!
//! Per the #680 decision, integer overflow **traps with a branded diagnostic**
//! rather than wrapping or saturating, mirroring the existing
//! `chelis_int_div_guard` / `integer division or remainder by zero` precedent
//! (`spec/05-risc-primitives.md`). It must trap identically at every integer
//! width; today int8/int16/int32 silently wrap while int64 saturates, because
//! Rust's int->int `as` truncates while float->int `as` saturates.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use std::path::Path;
use tempfile::{TempDir, tempdir};

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// `2^53`. The largest power of two where `n` and `n + 1` are both exactly
/// representable in f64. `2^53 + 1` is NOT representable and rounds to `2^53`,
/// which makes this the canonical boundary for every precision assertion here.
const TWO_POW_53: i64 = 9_007_199_254_740_992;
/// `2^53 + 1`. Not representable in f64; rounds to `TWO_POW_53`.
const TWO_POW_53_PLUS_1: i64 = 9_007_199_254_740_993;
/// `i64::MAX`. Rust's saturating `f64 as i64` clamps here on overflow.
const I64_MAX: i64 = 9_223_372_036_854_775_807;

// ---------------------------------------------------------------------------
// Lane harnesses
// ---------------------------------------------------------------------------

/// Run `program` under the evaluator lane (`chelis eval --file`) and return
/// stdout. The style gate is disabled because these fixtures synthesize ad-hoc
/// Surf to exercise numeric behavior, per the `CHELIS_STYLE_GATE_DISABLE`
/// carve-out in `CLAUDE.md`.
fn eval_lane(program: &str) -> (bool, String, String) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("probe.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .output()
        .expect("chelis eval should run");
    (
        out.status.success(),
        String::from_utf8_lossy(&out.stdout).into_owned(),
        String::from_utf8_lossy(&out.stderr).into_owned(),
    )
}

/// Evaluate a single `int64` expression under the evaluator lane and return the
/// printed value parsed as an exact `i64`.
///
/// Deliberately parses `i64`, never `f64`: parsing through `f64` is precisely
/// the bug under test, and an `f64`-typed oracle cannot observe it (see the
/// `eval_agreement.rs` note in the module docs).
fn eval_int(expr: &str) -> i64 {
    let program = format!("module Probe.Main\nout = print({expr})\n");
    let (ok, stdout, stderr) = eval_lane(&program);
    assert!(ok, "eval lane failed for `{expr}`:\nstderr: {stderr}");
    let first = stdout
        .lines()
        .next()
        .unwrap_or_else(|| panic!("eval produced no output for `{expr}`"));
    first
        .trim()
        .parse::<i64>()
        .unwrap_or_else(|e| panic!("eval output `{first}` for `{expr}` is not an i64: {e}"))
}

/// Evaluate a `bool` expression under the evaluator lane.
fn eval_bool(expr: &str) -> bool {
    let program = format!("module Probe.Main\nout = print({expr})\n");
    let (ok, stdout, stderr) = eval_lane(&program);
    assert!(ok, "eval lane failed for `{expr}`:\nstderr: {stderr}");
    let first = stdout.lines().next().unwrap_or("").trim().to_string();
    match first.as_str() {
        "true" => true,
        "false" => false,
        other => panic!("eval output `{other}` for `{expr}` is not a bool"),
    }
}

/// True when the host C toolchain is available for the compiled lane.
fn c_toolchain_available() -> bool {
    std::process::Command::new("cc")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}

/// Build `program` through `chelis build --target c`, link it, run it, and
/// return the value printed for the `out` binding parsed as an exact `i64`.
fn build_run_int(dir: &TempDir, program: &str, name: &str) -> i64 {
    let path = dir.path().join(format!("{name}.ch"));
    let out_dir = dir.path().join(format!("{name}-out"));
    write_file(&path, program);
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

    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    assert!(status.success(), "link failed for {name}: {status}");
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    let stdout = String::from_utf8_lossy(&run.stdout).into_owned();
    parse_out_binding(&stdout, name)
}

/// Parse `out = <i64>` from a compiled binary's stdout, exactly.
fn parse_out_binding(stdout: &str, label: &str) -> i64 {
    let line = stdout
        .lines()
        .find(|l| l.trim_start().starts_with("out ="))
        .unwrap_or_else(|| panic!("{label}: no `out =` line in compiled output:\n{stdout}"));
    let rhs = line.split('=').nth(1).unwrap_or("").trim();
    rhs.parse::<i64>()
        .unwrap_or_else(|e| panic!("{label}: compiled output `{rhs}` is not an i64: {e}"))
}

/// Assert both lanes produce `expected` for a program whose `out` binding is
/// `expr`. This is the invariant the whole issue turns on: a program must not
/// depend on whether it was interpreted or compiled.
fn assert_lane_parity(expr: &str, expected: i64, name: &str) {
    let eval_result = eval_int(expr);
    if !c_toolchain_available() {
        eprintln!("skipping compiled lane for {name}: no host C toolchain");
        assert_eq!(
            eval_result, expected,
            "{name}: eval lane returned {eval_result}, expected {expected}"
        );
        return;
    }
    let dir = tempdir().expect("tempdir");
    let program = format!("def run() -> int64 = {expr}\nout = run()\n");
    let c_result = build_run_int(&dir, &program, name);
    assert_eq!(
        eval_result, c_result,
        "{name}: LANE DIVERGENCE. `chelis eval` returned {eval_result} but \
         compiled C returned {c_result} for the same program `{expr}`. The \
         same source must not produce different values in different lanes."
    );
    assert_eq!(
        eval_result, expected,
        "{name}: both lanes returned {eval_result}, expected {expected}"
    );
}

// ---------------------------------------------------------------------------
// Group 1: cross-lane parity. The invariant that would have caught this.
// ---------------------------------------------------------------------------

/// Verified during investigation: eval returns 9007199254740992, compiled C
/// returns 9007199254740993. `2^53` and `1` are both exactly representable in
/// f64; their true sum is not. This is the minimal reproduction of #680.
#[test]
#[ignore = "chelis#680: eval 9007199254740992, compiled C 9007199254740993. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn add_at_two_pow_53_agrees_across_lanes() {
    assert_lane_parity(
        &format!("add(cast({TWO_POW_53}, int64), cast(1, int64))"),
        TWO_POW_53_PLUS_1,
        "add_two_pow_53",
    );
}

/// The wrong-branch case. Verified: eval takes the else branch (222) while
/// compiled C takes the then branch (111), because eval's `lt` compares in f64
/// and reports `2^53 < 2^53 + 1` as false. A program must not change control
/// flow based on its execution lane.
#[test]
#[ignore = "chelis#680: eval takes else (222), compiled C takes then (111). This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn static_int_condition_selects_same_branch_across_lanes() {
    let expr = format!(
        "if lt(cast({TWO_POW_53}, int64), cast({TWO_POW_53_PLUS_1}, int64)) \
         then cast(111, int64) else cast(222, int64)"
    );
    assert_lane_parity(&expr, 111, "static_cond_branch");
}

/// Multiplication is where Std.Decimal's coefficient scaling lives.
#[test]
fn mul_above_two_pow_53_agrees_across_lanes() {
    // 94906266^2 = 9007199326062756, exactly representable and above 2^53.
    // Guards against a fix that clamps everything above 2^53 rather than
    // computing exactly.
    assert_lane_parity(
        "mul(cast(94906266, int64), cast(94906266, int64))",
        9_007_199_326_062_756,
        "mul_above_2p53",
    );
}

// ---------------------------------------------------------------------------
// Group 2: literal fidelity. Verified CLEAN - these are regression locks.
// ---------------------------------------------------------------------------

/// Verified during investigation: the Surf/Deep front end carries integer
/// literals as exact `i64` through lexer, parser, AST, desugar, formatter and
/// decompiler. Locks that in so a future "just store literals as f64" change
/// cannot regress it silently.
#[test]
fn int64_literal_survives_the_front_end_exactly() {
    assert_eq!(
        eval_int(&format!("cast({TWO_POW_53_PLUS_1}, int64)")),
        TWO_POW_53_PLUS_1,
        "an int64 literal above 2^53 must not be rounded by the front end"
    );
}

/// `i64::MAX` as a bare literal. Verified exact today.
#[test]
fn i64_max_literal_survives_the_front_end_exactly() {
    assert_eq!(
        eval_int(&format!("cast({I64_MAX}, int64)")),
        I64_MAX,
        "i64::MAX must round-trip as a literal"
    );
}

// ---------------------------------------------------------------------------
// Group 3: comparison ops. Verified WRONG today.
// ---------------------------------------------------------------------------

/// Verified: returns `false`. `ordered_compare`
/// (`host_ops.rs:881-893`) compares integer scalars via `.as_f64()` even
/// though `ScalarBits::I64` holds them exactly. Note `compare_eq` directly
/// above it was fixed to use `.as_i64()`; `ordered_compare` never was.
#[test]
#[ignore = "chelis#680: ordered_compare compares via as_f64; returns false. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn lt_is_exact_at_two_pow_53_boundary() {
    assert!(
        eval_bool(&format!(
            "lt(cast({TWO_POW_53}, int64), cast({TWO_POW_53_PLUS_1}, int64))"
        )),
        "lt(2^53, 2^53+1) must be true; f64 comparison collapses both operands"
    );
}

/// Verified: returns `false`.
#[test]
#[ignore = "chelis#680: ordered_compare compares via as_f64; returns false. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn gt_is_exact_at_two_pow_53_boundary() {
    assert!(
        eval_bool(&format!(
            "gt(cast({TWO_POW_53_PLUS_1}, int64), cast({TWO_POW_53}, int64))"
        )),
        "gt(2^53+1, 2^53) must be true"
    );
}

/// Negative parity for `lt`: distinct large values must not compare equal.
#[test]
fn eq_is_exact_at_two_pow_53_boundary() {
    assert!(
        !eval_bool(&format!(
            "eq(cast({TWO_POW_53}, int64), cast({TWO_POW_53_PLUS_1}, int64))"
        )),
        "eq(2^53, 2^53+1) must be false"
    );
}

/// Verified: returns 9007199254740992, i.e. the SMALLER operand. `max_elem`
/// routes through `numeric_binop(args, f64::max)` (`eval.rs:843`).
#[test]
#[ignore = "chelis#680: returns the SMALLER operand. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn max_elem_is_exact_at_two_pow_53_boundary() {
    assert_eq!(
        eval_int(&format!(
            "max_elem(cast({TWO_POW_53}, int64), cast({TWO_POW_53_PLUS_1}, int64))"
        )),
        TWO_POW_53_PLUS_1,
        "max_elem must return the larger operand, not the f64-collapsed one"
    );
}

/// Sibling of `max_elem`; `min_elem` lowers via `neg(max_elem(neg, neg))`.
#[test]
fn min_elem_is_exact_at_two_pow_53_boundary() {
    assert_eq!(
        eval_int(&format!(
            "min_elem(cast({TWO_POW_53}, int64), cast({TWO_POW_53_PLUS_1}, int64))"
        )),
        TWO_POW_53,
        "min_elem must return the smaller operand"
    );
}

// ---------------------------------------------------------------------------
// Group 4: overflow traps. Per #680: errors, not wraps, at EVERY width.
// ---------------------------------------------------------------------------

/// Assert an expression fails with a branded integer-overflow diagnostic.
fn assert_overflow_traps(expr: &str, label: &str) {
    let program = format!("module Probe.Main\nout = print({expr})\n");
    let (ok, stdout, stderr) = eval_lane(&program);
    assert!(
        !ok,
        "{label}: `{expr}` must trap on integer overflow, but it succeeded \
         and printed: {stdout}"
    );
    let combined = format!("{stdout}{stderr}");
    assert!(
        combined.contains("overflow"),
        "{label}: `{expr}` failed but without a branded overflow diagnostic. \
         Got: {combined}"
    );
}

/// Verified today: returns `9223372036854775807` (saturates), because the f64
/// round-trip's `as i64` saturates.
#[test]
#[ignore = "chelis#680: saturates to i64::MAX instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn int64_add_overflow_traps() {
    assert_overflow_traps(
        &format!("add(cast({I64_MAX}, int64), cast(1, int64))"),
        "int64_add_overflow",
    );
}

/// Verified today: returns `9223372036854775807`. Both operands are in range;
/// only the product overflows, so this cannot be dismissed as a bad literal.
#[test]
#[ignore = "chelis#680: saturates to i64::MAX instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn int64_mul_overflow_traps() {
    assert_overflow_traps(
        "mul(cast(4000000000, int64), cast(4000000000, int64))",
        "int64_mul_overflow",
    );
}

/// Verified today: returns `-128` (wraps). int8/16/32 wrap while int64
/// saturates, decided purely by which Rust `as` cast runs last:
/// `ScalarBits::from_i64_as` narrows with `value as i8`
/// (`crates/chelis-compiler-api/src/runtime/mod.rs:145`), and int->int `as`
/// truncates bits.
#[test]
#[ignore = "chelis#680: wraps to -128 instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn int8_add_overflow_traps() {
    assert_overflow_traps("add(cast(127, int8), cast(1, int8))", "int8_add_overflow");
}

/// Verified today: returns `-32768` (wraps).
#[test]
#[ignore = "chelis#680: wraps to -32768 instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn int16_add_overflow_traps() {
    assert_overflow_traps(
        "add(cast(32767, int16), cast(1, int16))",
        "int16_add_overflow",
    );
}

/// Verified today: returns `-2147483648` (wraps).
#[test]
#[ignore = "chelis#680: wraps to -2147483648 instead of trapping. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn int32_add_overflow_traps() {
    assert_overflow_traps(
        "add(cast(2147483647, int32), cast(1, int32))",
        "int32_add_overflow",
    );
}

/// Negative parity for the overflow trap: a computation that does NOT overflow
/// must still succeed. Guards against a fix that traps too eagerly.
#[test]
fn int64_add_at_max_minus_one_does_not_trap() {
    assert_eq!(
        eval_int(&format!(
            "add(cast({}, int64), cast(1, int64))",
            I64_MAX - 1
        )),
        I64_MAX,
        "i64::MAX - 1 + 1 == i64::MAX must not trap"
    );
}

/// Negative parity at the narrow width.
#[test]
fn int8_add_at_max_does_not_trap() {
    assert_eq!(
        eval_int("add(cast(126, int8), cast(1, int8))"),
        127,
        "126 + 1 == 127 fits in int8 and must not trap"
    );
}

// ---------------------------------------------------------------------------
// Group 5: tensor storage fidelity.
//
// `IrTensorValue` stores `data: Vec<f64>` for every dtype
// (`crates/chelis-ir/src/eval.rs:13`), so int64 tensors are lossy AT REST,
// before any op runs. Verified: `to_list(to_tensor([2^53+1]))` returns
// `[9007199254740992]`.
//
// This needs a dtype-tagged tensor storage change rather than an arithmetic
// fix, so it is tracked separately and ignored here rather than silently
// omitted (see the "no silent caps" rule in CLAUDE.md).
// ---------------------------------------------------------------------------

#[test]
#[ignore = "chelis#680 follow-up: int64 tensor storage is Vec<f64> \
            (crates/chelis-ir/src/eval.rs:13) and is lossy at rest. Needs \
            dtype-tagged storage, not an arithmetic fix. Run with \
            `cargo test -p chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn int64_tensor_survives_to_tensor_round_trip() {
    let program = format!(
        "module Probe.Main\n\
         xs: List[int64] = [cast({TWO_POW_53_PLUS_1}, int64)]\n\
         out = print(to_list(to_tensor(xs)))\n"
    );
    let (ok, stdout, stderr) = eval_lane(&program);
    assert!(ok, "tensor round-trip failed: {stderr}");
    let first = stdout.lines().next().unwrap_or("").trim().to_string();
    assert_eq!(
        first,
        format!("[{TWO_POW_53_PLUS_1}]"),
        "to_list(to_tensor([2^53+1])) must round-trip exactly; \
         Vec<f64> storage rounds it to 2^53"
    );
}

// ---------------------------------------------------------------------------
// Group 6: Std.Decimal, the user-visible victim (chelis#680, chelis#681).
//
// Std.Decimal stores its coefficient as int64 and scales via repeated
// `mul(coefficient, 10)`, so it inherits the f64 arithmetic bug directly.
// These run through the staged-std harness rather than a bare file because
// they need the Std.Decimal import to resolve.
// ---------------------------------------------------------------------------

/// Verified: returns `922337203685477.5807` instead of `1219326320073159.3435`.
/// These are ordinary 8-digit dollar amounts with cents, so routine money math
/// reaches this. Under the #680 trap contract this must now fail loudly rather
/// than return a wrong number.
#[test]
#[ignore = "chelis#680: needs the staged chelis-std reef fixture; exceeds the \
            inner-loop budget. Run with `cargo test -p chelis-cli --test \
            issue_680_int_exactness -- --ignored`."]
fn decimal_mul_does_not_silently_saturate() {
    let (_dir, reef_home, app_pkg) = common::make_app("issue-680-decimal-mul");
    write_file(
        &app_pkg.join("src/main.ch"),
        "module Demo.Main\n\
         import Std.Decimal (decimal, decimal_mul, decimal_to_string)\n\
         out = print(decimal_to_string(decimal_mul(decimal(\"12345678.99\"), \
         decimal(\"98765432.11\"))))\n",
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
    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);
    assert!(
        !stdout.contains("922337203685477.5807"),
        "decimal_mul silently saturated to i64::MAX: {stdout}"
    );
    if out.status.success() {
        assert!(
            stdout.contains("1219326320073159.3435"),
            "decimal_mul returned an unexpected value: {stdout}"
        );
    } else {
        assert!(
            format!("{stdout}{stderr}").contains("overflow"),
            "decimal_mul failed without a branded overflow diagnostic: {stderr}"
        );
    }
}

// ---------------------------------------------------------------------------
// Group 7: bitwise / shift codegen parity (chelis#682).
//
// `chelis build --target c` has no match arm for bitand/bitor/bitxor/shl/shr
// in `crates/chelis-backend-c/src/host_emit.rs`, so all five fall through to
// the generic fallback `other => format!("/* unsupported builtin {other} */ 0")`
// and silently compile to the literal 0. The evaluator implements all five
// correctly via `int_binop` / `int_shift_binop`
// (`crates/chelis-compiler-api/src/runtime/eval.rs:942-946`), so the lanes
// disagree on every one of them.
//
// Verified by compiling and running: eval gives 8/14/6/1024/128 while the
// compiled binary gives 0 for all five. No large values are involved; this is
// a missing-codegen bug, not a precision bug.
// ---------------------------------------------------------------------------

/// `bitand(12, 10) == 8`. Verified: eval 8, compiled C 0.
#[test]
#[ignore = "chelis#682: eval 8, compiled C 0. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn bitand_agrees_across_lanes() {
    assert_lane_parity(
        "bitand(cast(12, int64), cast(10, int64))",
        8,
        "bitand_parity",
    );
}

/// `bitor(12, 10) == 14`. Verified: eval 14, compiled C 0.
#[test]
#[ignore = "chelis#682: eval 14, compiled C 0. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn bitor_agrees_across_lanes() {
    assert_lane_parity(
        "bitor(cast(12, int64), cast(10, int64))",
        14,
        "bitor_parity",
    );
}

/// `bitxor(12, 10) == 6`. Verified: eval 6, compiled C 0.
#[test]
#[ignore = "chelis#682: eval 6, compiled C 0. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn bitxor_agrees_across_lanes() {
    assert_lane_parity(
        "bitxor(cast(12, int64), cast(10, int64))",
        6,
        "bitxor_parity",
    );
}

/// `shl(1, 10) == 1024`. Verified: eval 1024, compiled C 0.
#[test]
#[ignore = "chelis#682: eval 1024, compiled C 0. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn shl_agrees_across_lanes() {
    assert_lane_parity("shl(cast(1, int64), cast(10, int64))", 1024, "shl_parity");
}

/// `shr(1024, 3) == 128`. Verified: eval 128, compiled C 0.
#[test]
#[ignore = "chelis#682: eval 128, compiled C 0. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with `cargo test -p \
            chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn shr_agrees_across_lanes() {
    assert_lane_parity("shr(cast(1024, int64), cast(3, int64))", 128, "shr_parity");
}

/// The class-level guard, not just the five instances. `reject_host_only_builtins`
/// (`crates/chelis-compiler-api/src/compiler.rs:2232`) exists to stop a builtin
/// from silently becoming a C stub, but `HOST_ONLY_BUILTINS` is `&["tensor_scan"]`
/// (`compiler.rs:2230`), so it caught one builtin and missed five.
///
/// An unimplemented builtin must fail the BUILD, not compile to `0` and return
/// garbage at runtime. This test pins that no emitted C ever contains the
/// silent-stub marker.
#[test]
#[ignore = "chelis#682: build emits a silent `unsupported builtin` stub. This test asserts the CORRECT \
            behavior and fails until the fix lands. Run with \
            `cargo test -p chelis-cli --test issue_680_int_exactness -- --ignored`."]
fn unsupported_builtin_never_silently_emits_a_zero_stub() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let dir = tempdir().expect("tempdir");
    let program = "def run() -> int64 = bitand(cast(12, int64), cast(10, int64))\nout = run()\n";
    let path = dir.path().join("stub.ch");
    let out_dir = dir.path().join("stub-out");
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
        // A clean build error naming the builtin is the ACCEPTABLE outcome:
        // unimplemented must fail loudly rather than miscompile.
        return;
    }
    let emitted = std::fs::read_to_string(out_dir.join("stub.c")).expect("emitted C");
    assert!(
        !emitted.contains("unsupported builtin"),
        "chelis build succeeded but emitted a silent `unsupported builtin` stub \
         that compiles to 0 and returns garbage at runtime. An unimplemented \
         builtin must fail the build instead."
    );
}

// ---------------------------------------------------------------------------
// Group 8: front-end literal range.
// ---------------------------------------------------------------------------

/// Verified: `cast(-9223372036854775808, int64)` dies with
/// `lex error: invalid number '9223372036854775808'`, because the lexer parses
/// the magnitude as a positive `i64` before the parser applies negation, and
/// `2^63` overflows `i64`. Fails loudly rather than silently, so it is a
/// usability gap rather than a correctness hole.
#[test]
#[ignore = "i64::MIN is not writable as a literal (lexer parses the magnitude \
            as a positive i64 before negation, crates/chelis-surf/src/lexer.rs:717). \
            Not yet filed; unignore when it is fixed."]
fn i64_min_is_writable_as_a_literal() {
    let program = "module Probe.Main\nout = print(cast(-9223372036854775808, int64))\n";
    let (ok, stdout, stderr) = eval_lane(program);
    assert!(ok, "i64::MIN must be writable as a literal, got: {stderr}");
    assert_eq!(
        stdout.lines().next().unwrap_or("").trim(),
        "-9223372036854775808"
    );
}

/// The documented workaround must keep working regardless of #682, and it
/// pins that `sub` is exact at the i64::MIN boundary.
#[test]
fn i64_min_workaround_is_exact() {
    assert_eq!(
        eval_int("sub(cast(-9223372036854775807, int64), cast(1, int64))"),
        i64::MIN,
        "i64::MIN + 1 - 1 must be exactly i64::MIN"
    );
}

// ---------------------------------------------------------------------------
// Group 9: float lanes must NOT regress.
//
// The f64 kernel is mathematically sound for floats: f64 carries at least
// 2p+2 bits for f32 (p=24), f16 (p=11) and bf16 (p=8), so add/sub/mul through
// f64 are correctly rounded and match native f32. These tests exist so a fix
// that splits the integer lane out does not disturb float behavior.
// ---------------------------------------------------------------------------

/// f32 addition at its own precision boundary (`2^24 + 1` is not representable
/// in f32). Pins that the float lane keeps native-f32 rounding.
#[test]
fn f32_add_rounds_like_native_f32() {
    let program = "module Probe.Main\n\
                   out = print(add(cast(16777216.0, f32), cast(1.0, f32)))\n";
    let (ok, stdout, stderr) = eval_lane(program);
    assert!(ok, "f32 add failed: {stderr}");
    let first = stdout.lines().next().unwrap_or("").trim().to_string();
    let parsed: f64 = first
        .parse()
        .unwrap_or_else(|e| panic!("f32 output `{first}` unparseable: {e}"));
    // Native f32: 16777216.0 + 1.0 == 16777216.0 (ties-to-even).
    assert_eq!(
        parsed, 16_777_216.0,
        "f32 add must round like native f32, not accumulate f64 precision"
    );
}

/// f64 addition at its own boundary. `2^53 + 1` is not representable in f64,
/// so the correct f64 answer IS 2^53. This is the float-lane mirror of the
/// integer bug and documents that the same numeric input is correct here and
/// wrong for int64.
#[test]
fn f64_add_rounds_like_native_f64() {
    let program = "module Probe.Main\n\
                   out = print(add(cast(9007199254740992.0, f64), cast(1.0, f64)))\n";
    let (ok, stdout, stderr) = eval_lane(program);
    assert!(ok, "f64 add failed: {stderr}");
    let first = stdout.lines().next().unwrap_or("").trim().to_string();
    let parsed: f64 = first
        .parse()
        .unwrap_or_else(|e| panic!("f64 output `{first}` unparseable: {e}"));
    assert_eq!(
        parsed, 9_007_199_254_740_992.0,
        "f64 add at 2^53 must round to 2^53; this is correct for f64 and is \
         exactly why the same value is WRONG for int64"
    );
}

fn _assert_helpers_used(_p: &Path) {}
