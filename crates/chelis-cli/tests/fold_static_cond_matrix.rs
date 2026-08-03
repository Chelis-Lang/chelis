//! chelis#720 regression matrix for `fold_static_cond`: cast operands must be
//! finalized as sealed f16/bf16 scalars before comparison. Folding through an
//! f32 memo deletes the branch IEEE f16/bf16 semantics require.
//!
//! Sibling of chelis#711 (the integer Const arm of the same fold, threshold
//! 2^53); this one fires at 2049 (f16) / 257 (bf16). The fixes do not
//! overlap: #711's checked-i64 folding does not touch the Cast arm.
//!
//! All branch-presence checks compare f32 BIT PATTERNS in the emitted C
//! (111.0 = 0x42de0000, 222.0 = 0x435e0000), never decimal text - a decimal
//! `111` grep matches the hash constant 0x94D049BB133111EBULL (the exact
//! false-positive the #711 audit recorded).
//!
//! Bounding controls: conditions with an effectful branch (`fail`) route
//! host-lane, do not fold, and the compiled int64 comparison there is exact
//! (locked below). The ignored int8 row belongs to the Phase 3 compiled-C
//! trap work; this Phase 2 fold must at least decline when its typed kernel
//! detects the overflow.

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

/// Build to C; return `(emitted_c, run_stdout, run_stderr, run_ok)`.
fn build_and_run_c(program: &str, name: &str) -> Result<(String, String, String, bool), String> {
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
    let emitted = std::fs::read_to_string(out_dir.join(format!("{name}.c")))
        .map_err(|e| format!("read emitted C: {e}"))?;
    let status = common::link_generated(&out_dir, &format!("{name}.c"), name);
    if !status.success() {
        return Err(format!("link failed: {status}"));
    }
    let run = std::process::Command::new(out_dir.join(name))
        .output()
        .expect("compiled binary should run");
    Ok((
        emitted,
        String::from_utf8_lossy(&run.stdout).into_owned(),
        String::from_utf8_lossy(&run.stderr).into_owned(),
        run.status.success(),
    ))
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

/// f32 bit pattern of the 222.0 branch payload as it appears in emitted
/// `chelis_fill_f32_bits` calls (111.0 is 0x42de0000; the broken rows
/// assert on the DELETED branch's bits, which is 222.0's).
const BITS_222: &str = "435e0000";

// ===========================================================================
// chelis#720 - the Cast arm deletes the IEEE-correct branch
// ===========================================================================

/// True f16 rounds cast(2049.0, f16) to 2048, so lt(2048, 2048) is false and
/// the answer is 222. Observed today: eval prints 222 (correct); the
/// compiled binary prints 111, and 222's bit pattern is ABSENT from the
/// emitted C - the correct branch was deleted at compile time.
#[test]
#[ignore = "chelis#720 fold semantics are now covered in chelis-ir; the end-to-end C row \
            remains blocked because C host ABI selection rejects f16 before dead-condition \
            elimination. Artifact routing and C host ABI support are outside this Phase 2 \
            slice. Run with `cargo test -p chelis-cli --test fold_static_cond_matrix -- --ignored`."]
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
    let (emitted, stdout, _, ok) = build_and_run_c(program, "fold_f16").expect("C lane");
    assert!(ok);
    assert!(
        emitted.to_lowercase().contains(BITS_222),
        "the 222 branch (0x435e0000) must exist in the emitted C; it was deleted"
    );
    assert!(
        stdout.lines().next().unwrap_or("").trim() == "222.0",
        "the compiled program must take the IEEE f16 branch; got: {stdout}"
    );
}

/// bf16 sibling at threshold 257 (8-bit mantissa).
#[test]
#[ignore = "chelis#720 fold semantics are now covered in chelis-ir; the end-to-end C row \
            remains blocked because C host ABI selection rejects bf16 before dead-condition \
            elimination. Artifact routing and C host ABI support are outside this Phase 2 \
            slice. Run with `cargo test -p chelis-cli --test fold_static_cond_matrix -- --ignored`."]
fn bf16_cast_condition_folds_with_bf16_semantics() {
    let program = "def pick() -> f32 = if lt(cast(256.0, bf16), cast(257.0, bf16)) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    assert_eq!(eval_first_line(program).expect("eval"), "222.0");
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (emitted, stdout, _, ok) = build_and_run_c(program, "fold_bf16").expect("C lane");
    assert!(ok);
    assert!(
        emitted.to_lowercase().contains(BITS_222),
        "the 222 branch (0x435e0000) must exist in the emitted C; it was deleted"
    );
    assert!(
        stdout.lines().next().unwrap_or("").trim() == "222.0",
        "the compiled program must take the IEEE bf16 branch; got: {stdout}"
    );
}

// ===========================================================================
// chelis#718 - int8 conditions do NOT fold, but the runtime branch diverges
// through the int64_t widening (#714). Contract: the overflow must trap.
// ===========================================================================

/// `add(100i8, 100i8)` overflows int8. Today eval wraps (-56 < 0, prints
/// 111) and compiled C widens (200 < 0, prints 222) - opposite branches at
/// runtime, no deletion (both bit patterns present in the emitted C,
/// verified when this row was probed). The decided contract (#680/#695)
/// says the overflow itself must trap in both lanes.
#[test]
#[ignore = "chelis#718: an int8 overflow used as a branch condition sends eval and C down \
            opposite branches (eval wraps to -56 and prints 111; C widens to 200 and prints \
            222); the contract says the overflow must trap in both lanes. Run with \
            `cargo test -p chelis-cli --test fold_static_cond_matrix -- --ignored`."]
fn int8_overflow_condition_traps_in_both_lanes() {
    let program = "def pick() -> f32 = if lt(add(100i8, 100i8), 0i8) \
                   then 111.0 else 222.0\nout = print(pick())\n";
    match eval_first_line(program) {
        Ok(line) => panic!("eval must trap on the int8 overflow, got: {line}"),
        Err(stderr) => assert!(stderr.contains("overflow"), "got: {stderr}"),
    }
    if !c_toolchain_available() {
        panic!("needs a host C toolchain");
    }
    let (_, stdout, stderr, ok) = build_and_run_c(program, "fold_i8").expect("C lane");
    assert!(
        !ok && stderr.contains("overflow"),
        "compiled C must trap on the int8 overflow; got ok={ok}, stdout `{stdout}`"
    );
}

// ===========================================================================
// CONTROLS
// ===========================================================================

/// An effectful branch (`fail`) keeps the def in the host lane: no fold,
/// both branches present in the emitted C, and the compiled int64 comparison
/// is EXACT - `lt(2^53, 2^53 + 1)` is true, so the binary must trap with the
/// fail message. It does. (eval takes the wrong branch on the same program -
/// that is chelis#680's known f64 comparison bug, asserted nowhere here.)
#[test]
fn host_lane_fail_branch_survives_and_c_comparison_is_exact() {
    if !c_toolchain_available() {
        eprintln!("skipping: no host C toolchain");
        return;
    }
    let program = "def pick() -> f32 = if lt(9007199254740992i64, 9007199254740993i64) \
                   then fail(\"int64 invariant violated\") else 222.0\nout = print(pick())\n";
    let (emitted, _, stderr, ok) = build_and_run_c(program, "fold_fail").expect("C lane");
    assert!(
        emitted.contains("int64 invariant violated"),
        "the fail branch must survive lowering (host lane, no fold)"
    );
    assert!(
        !ok && stderr.contains("int64 invariant violated"),
        "2^53 < 2^53 + 1 is true in exact integers; the compiled host lane \
         must take the fail branch. got ok={ok}, stderr: {stderr}"
    );
}
