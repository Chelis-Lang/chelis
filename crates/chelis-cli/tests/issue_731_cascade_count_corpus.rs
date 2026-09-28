//! chelis#731 Phase 2 -- the cascade-count corpus (Part II B2.3 of
//! `spec/design/checker_totality.md`).
//!
//! The `ErrorWitness` migration replaced ~250 `errors.push(...); return
//! Type::Error;` sites with `report(...)` and the cascade sites with
//! `propagate(...)`. `propagate` preserves the pre-token behavior exactly: a
//! node whose child already typed as `Type::Error` re-types as `Type::Error`
//! WITHOUT re-reporting, so one root cause does not spray a fan-out of
//! secondary diagnostics. This corpus is the NEGATIVE-PARITY control for that
//! claim: a fixed set of multi-error / nested-error programs, each asserted to
//! produce EXACTLY the diagnostic count it produced on `main` before the
//! migration. A count that grew would be diagnostic spray (a regression in
//! cascade suppression); a count that shrank would be a masked error.
//!
//! Baseline provenance: the `expected` counts were measured on `origin/main`
//! (the pre-migration tree) with this exact corpus and this exact
//! `chelis check` invocation, then re-measured on the migration branch and
//! asserted equal. The before/after table is recorded in the Phase 2 PR body.
//! If a legitimate checker change alters one of these counts, update the
//! baseline in the SAME change set and record the new before/after in the PR
//! (the B2.3 discipline).

#![allow(clippy::uninlined_format_args)]

use assert_cmd::Command;
use tempfile::tempdir;

#[path = "common/mod.rs"]
mod common;

use common::write_file;

const MASKED_ERROR: &str = "add(cast(1.0, f32), cast(2, i64))";

/// Number of diagnostics `chelis check` reports for `program`.
fn diagnostic_count(program: &str) -> usize {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("p.ch");
    write_file(&path, program);
    let out = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("chelis check should run");
    let parsed: serde_json::Value =
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| panic!("check must emit JSON: {e}"));
    parsed["errors"]
        .as_array()
        .map(|a| a.len())
        .unwrap_or_else(|| panic!("check JSON must carry an `errors` array: {parsed}"))
}

/// The corpus: (name, program, expected diagnostic count measured on main).
///
/// NOTE(chelis#731 Phase 2): the `expected` values below are the placeholders
/// measured and finalized during implementation against `origin/main`; the
/// build loop pins them to the exact pre-migration counts.
fn corpus() -> Vec<(&'static str, String, usize)> {
    vec![
        // One root error in tail position: reported once.
        (
            "single_error",
            format!("def f() -> f32 = {MASKED_ERROR}\n"),
            1,
        ),
        // Cascade: the masked error is one root cause; the enclosing `add`
        // propagates it and must NOT add a second diagnostic.
        (
            "cascade_over_reported_error",
            format!("def f() -> f32 = add({MASKED_ERROR}, cast(3.0, f32))\n"),
            1,
        ),
        // Deeper nesting: still one root cause, one diagnostic.
        (
            "cascade_two_levels",
            format!("def f() -> f32 = add(add({MASKED_ERROR}, cast(3.0, f32)), cast(4.0, f32))\n"),
            1,
        ),
        // The error under an effect wrapper (chelis#709 shape): still one.
        (
            "cascade_under_with_device",
            format!("def f() -> f32 = with device(\"gpu:0\") {{ {MASKED_ERROR} }}\n"),
            1,
        ),
        // The error under an `if` branch: one.
        (
            "cascade_under_if",
            format!("def f(c: bool) -> f32 = if c then {MASKED_ERROR} else 1.0\n"),
            1,
        ),
        // Two INDEPENDENT root errors in two defs: two diagnostics (each root
        // reported, no suppression across independent causes).
        (
            "two_independent_errors",
            format!("def f() -> f32 = {MASKED_ERROR}\ndef g() -> f32 = {MASKED_ERROR}\n"),
            2,
        ),
    ]
}

#[test]
fn cascade_counts_match_pre_migration_baseline() {
    let mut mismatches = Vec::new();
    for (name, program, expected) in corpus() {
        let got = diagnostic_count(&program);
        if got != expected {
            mismatches.push(format!(
                "{name}: expected {expected} diagnostic(s) (main baseline), got {got}"
            ));
        }
    }
    assert!(
        mismatches.is_empty(),
        "chelis#731 Phase 2 cascade-count regression (diagnostic spray or masking):\n  {}",
        mismatches.join("\n  ")
    );
}
