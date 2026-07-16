//! chelis#688 - the integer-via-f64 class (#695) reaching `chelis prove`.
//!
//! The issue's chokepoint (`flatten_field_value` / `read_produced_field`
//! collapsing produced int64 fields to f64) needs the smt build plus an
//! @opaque int64-field type to drive. But the CLASS is live in the default
//! build: the fuzz tier's concrete interpreter computes int64 arithmetic
//! through the #680 f64 evaluator, so `chelis prove` reports a TRUE int64
//! theorem as failed, with a "counterexample" that witnesses nothing.
//!
//! `shift_a(x) = x + (2^53 + 1)` and `shift_b(x) = x + 2^53` differ by
//! exactly 1 at every int64 input; `shift_a(x) != shift_b(x)` is a theorem.
//! Observed today: `property failure`, `counterexample: {"x": 0}` - and
//! shift_a(0) = 9007199254740993 vs shift_b(0) = 9007199254740992 are
//! distinct, so the witness is spurious. A formal-methods surface handing
//! out wrong counterexamples is the sharpest form of the class.

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

/// Run `chelis prove` on a program; return (stdout, success).
fn prove(program: &str) -> (String, bool) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "prove",
            path.to_str().unwrap(),
            "--samples",
            "8",
            "--seed",
            "7",
        ])
        .output()
        .expect("chelis prove should run");
    (
        String::from_utf8_lossy(&out.stdout).into_owned() + &String::from_utf8_lossy(&out.stderr),
        out.status.success(),
    )
}

/// Observed today: `property failure: distinct_offsets_stay_distinct` with
/// counterexample x = 0, which does not violate the property.
#[test]
#[ignore = "chelis#688: prove's fuzz-tier interpreter collapses the int64 offsets 2^53+1 \
            and 2^53 to one f64 and reports a true theorem failed with a spurious \
            counterexample (x = 0). Run with \
            `cargo test -p chelis-cli --test prove_int64_exactness -- --ignored`."]
fn prove_does_not_refute_a_true_int64_theorem() {
    let (stdout, _) = prove(
        "module Audit.ProveInt\n\
         export (shift_a, shift_b)\n\
         def shift_a(x: int64) -> int64 = add(x, 9007199254740993i64)\n\
         def shift_b(x: int64) -> int64 = add(x, 9007199254740992i64)\n\
         @property distinct_offsets_stay_distinct forall(x: int64):\n\
           (shift_a(x) != shift_b(x))\n",
    );
    assert!(
        stdout.contains("1 passed, 0 failed"),
        "shift_a and shift_b differ by exactly 1 at every int64 input; the \
         property is a theorem and must pass. Got:\n{stdout}"
    );
}

/// Control: the identical property shape at small offsets passes 8/8, so
/// the failure above is about magnitude (the f64 mantissa boundary), not
/// about the property machinery.
#[test]
fn prove_accepts_the_small_offset_control() {
    let (stdout, _) = prove(
        "module Audit.ProveIntCtl\n\
         export (bump_a, bump_b)\n\
         def bump_a(x: int64) -> int64 = add(x, 1i64)\n\
         def bump_b(x: int64) -> int64 = add(x, 2i64)\n\
         @property small_offsets_stay_distinct forall(x: int64):\n\
           (bump_a(x) != bump_b(x))\n",
    );
    assert!(
        stdout.contains("1 passed, 0 failed"),
        "the small-offset control must pass; got:\n{stdout}"
    );
}
