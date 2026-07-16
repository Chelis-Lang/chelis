//! Phase 3t.A1 follow-up — captured type-checker gaps (#39).
//!
//! These tests pin the locked behavior of #39 — defsig dim enforcement
//! through wildcard-bearing bodies — so a regression doesn't silently
//! re-open the soundness gap.
//!
//! ## #39 — defsig dim enforcement leaks through wildcard inference (FIXED)
//!
//! When a function declared with concrete tensor dims has a body whose
//! inferred type contains `Dim::Wildcard` (e.g. `pad_sequences_to`,
//! `to_tensor` produce wildcard dims), the unify between the body type
//! and the declared signature succeeds because `unify_dim` treats Wildcard
//! as a matches-anything sentinel. The fix in `infer_top_level` narrows
//! the body's wildcards against the declared template before generalizing,
//! so callers see the declared concrete shape and dim mismatches surface
//! as `DimensionMismatch` errors.
//!
//! Pure-sig case (no body) also catches the mismatch — see the second test.

use assert_cmd::Command;
use predicates::prelude::*;

#[path = "common/mod.rs"]
mod common;

use common::{make_app, write_file};

#[test]
#[ignore = "manual gate: Phase 3t type-checker gap CLI regression suite exceeds the default inner-loop budget"]
fn pure_sig_dim_mismatch_is_caught() {
    // Sanity check: when the value being passed is itself a sig (no body),
    // the dim mismatch IS caught. Pin this so a regression doesn't silently
    // weaken the working case alongside a fix for the broken one below.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-tc-pure-sig");
    write_file(
        &app_pkg.join("src/main.ch"),
        r"module Demo.Main

sig do_thing: tensor[32, 128, f32] -> f32
sig make_3: tensor[1, 3, f32]

result = do_thing(make_3)
",
    );
    // Issue #207: `chelis check` now exits non-zero (exit 2) when the
    // JSON `errors` array is non-empty. We assert on the JSON
    // content via stdout; the dedicated invariant test in
    // `issue_207_check_exit_code_invariant.rs` covers the exit
    // code separately.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .stdout(predicate::str::contains("DimensionMismatch"));
}

#[test]
#[ignore = "manual gate: Phase 3t type-checker gap CLI regression suite exceeds the default inner-loop budget"]
fn defsig_dim_enforcement_leaks_through_wildcard_body_in_callers() {
    // `make` is declared to return `tensor[1, 3, f32]`, but its body
    // inferred type carries Wildcard dims from `pad_sequences_to`.
    // `do_thing` expects `tensor[32, 128, f32]`. The call must fail with
    // `DimensionMismatch: Lit(32) vs Lit(1)`. Before the #39 fix,
    // `chelis check` silently reported score ~1 with no errors because
    // wildcards leaked into the generalized scheme.
    let (_dir, reef_home, app_pkg) = make_app("phase3t-tc-defsig-leak");
    write_file(
        &app_pkg.join("src/main.ch"),
        r"module Demo.Main

sig do_thing: tensor[32, 128, f32] -> f32

def make() -> tensor[1, 3, f32] =
  pad_sequences_to([[cast(1.0, f32), cast(2.0, f32), cast(3.0, f32)]], cast(3, int64), cast(0.0, f32))

result = do_thing(make())
",
    );
    // Issue #207: `chelis check` exits non-zero (exit 2) when the
    // JSON `errors` array is non-empty. We assert on the JSON
    // content; the dedicated invariant test covers the exit code.
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .env("CHELIS_REEF_HOME", &reef_home)
        .current_dir(&app_pkg)
        .args(["check", app_pkg.join("src/main.ch").to_str().unwrap()])
        .assert()
        .stdout(predicate::str::contains("DimensionMismatch"))
        .stdout(predicate::str::contains("Lit(32)"))
        .stdout(predicate::str::contains("Lit(1)"));
}
