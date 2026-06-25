//! Issue #463 regression-lock: a property whose top-level goal is a
//! boolean CONNECTIVE (conjunction `&&` / disjunction `||`) lowers to the
//! SMT tier (Tier B) and reaches a DEFINITE verdict — proven when the
//! connective holds, disproved-with-counterexample when it does not. The
//! identical bounds also lower individually and a `not (...)` goal lowers,
//! so this pins that the connective at the goal site is no weaker than its
//! operands.
//!
//! ## What #463 actually was
//!
//! The issue was filed against the keyword forms `(a) and (b)` /
//! `(a) or (b)`. But the SPEC-CANONICAL Surf boolean operators are `&&`
//! and `||` (`spec/02-surf-syntax.md` §2, the Final operator-precedence
//! table; they desugar to Deep `(app (var and) ...)` / `(app (var or)
//! ...)`). The bare keywords `and`/`or` are NOT Surf operators — they are
//! only the Deep `var` names. The RT6 total cvc5 lowering (the
//! `BoolOp::And`/`BoolOp::Or` arity-normalizing arm in
//! `chelis_prove::tier_b::lower_to_cvc5`) already lowers the connective at
//! the goal site, so `&&`/`||` (and the call-form `and(a, b)` / `or(a, b)`)
//! prove and disprove at SMT today.
//!
//! Per the maintainer decision (option A), #463 is closed as
//! already-fixed for the canonical `&&`/`||`; this file is the regression
//! lock plus a negative test that the non-spec keyword-infix form
//! `(a) and (b)` is a clean parse/type error — never a false pass and
//! never a panic. We deliberately do NOT teach the parser to accept
//! `and`/`or` as infix operators: that would contradict the §2 Final
//! operator table.

use assert_cmd::Command;
use serde_json::Value;
use tempfile::{TempDir, tempdir};

fn write_prop(source: &str) -> TempDir {
    let dir = tempdir().expect("tempdir");
    std::fs::write(dir.path().join("prop.ch"), source).expect("write property");
    dir
}

fn property_records(output: &[u8]) -> Vec<Value> {
    String::from_utf8_lossy(output)
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| serde_json::from_str::<Value>(line).expect("json line"))
        .filter(|record| record.get("kind").and_then(Value::as_str) == Some("property"))
        .collect()
}

fn run_prove_json(source: &str, tier: &str) -> std::process::Output {
    let dir = write_prop(source);
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "prove",
            dir.path().join("prop.ch").to_str().unwrap(),
            "--json",
            "--tier",
            tier,
        ])
        .output()
        .expect("run prove")
}

// ---------------------------------------------------------------------------
// Positive: a connective goal lowers to Tier B and PROVES.
// ---------------------------------------------------------------------------

/// A top-level `&&` conjunction of two true bounds proves at the SMT tier
/// (not Tier C fuzz, not unsupported). This is the conjunctive-certificate
/// shape downstream verification (WS-5) relies on.
#[cfg(feature = "smt")]
#[test]
fn conjunction_goal_proves_at_smt() {
    let output = run_prove_json(
        "@property conj forall(x: f32):\n  ((x * x >= 0.0) && (x * x >= 0.0))\n",
        "smt-only",
    );
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "conj");
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(props[0]["proof_tier"], "smt");
    assert_eq!(props[0]["arith_model"], "real");
}

/// A top-level `||` disjunction with one always-true clause proves at SMT.
#[cfg(feature = "smt")]
#[test]
fn disjunction_goal_proves_at_smt() {
    let output = run_prove_json(
        "@property disj forall(x: f32):\n  ((x * x >= 0.0) || (x * x >= 1.0))\n",
        "smt-only",
    );
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "disj");
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(props[0]["proof_tier"], "smt");
}

/// A NESTED connective (`(a && b) || c`) lowers and proves — pins that the
/// lowering recurses through the connective tree, not just a flat top node.
#[cfg(feature = "smt")]
#[test]
fn nested_connective_goal_proves_at_smt() {
    let output = run_prove_json(
        "@property nested forall(x: f32):\n  (((x * x >= 0.0) && (x * x >= 0.0)) || (x >= 1.0))\n",
        "smt-only",
    );
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "nested");
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(props[0]["proof_tier"], "smt");
}

/// The call-form `and(a, b)` connective lowers like the operator form
/// (`a && b`) — pins parity between the two spellings the lowering accepts.
#[cfg(feature = "smt")]
#[test]
fn call_form_conjunction_proves_at_smt() {
    let output = run_prove_json(
        "@property conj_call forall(x: f32):\n  and(x * x >= 0.0, x * x >= 0.0)\n",
        "smt-only",
    );
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "conj_call");
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(props[0]["proof_tier"], "smt");
}

// ---------------------------------------------------------------------------
// Negative parity: a connective goal that is FALSE is DISPROVED at Tier B,
// with a counterexample — never a false green.
// ---------------------------------------------------------------------------

/// A conjunction with one FALSE clause (`x*x >= 1.0` fails at x=0) is
/// disproved at SMT with a counterexample. This is the negative twin of
/// `conjunction_goal_proves_at_smt`: the connective must be solved, not
/// silently dropped to a fuzz pass.
#[cfg(feature = "smt")]
#[test]
fn false_conjunction_goal_disproved_at_smt() {
    let output = run_prove_json(
        "@property conj_false forall(x: f32):\n  ((x * x >= 0.0) && (x * x >= 1.0))\n",
        "smt-only",
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "conj_false");
    assert_eq!(props[0]["status"], "failed");
    assert_eq!(props[0]["proof_tier"], "smt");
    assert!(
        props[0].get("counterexample").is_some(),
        "expected a counterexample, got {:?}",
        props[0]
    );
}

/// A disjunction with BOTH clauses false is disproved at SMT.
#[cfg(feature = "smt")]
#[test]
fn false_disjunction_goal_disproved_at_smt() {
    let output = run_prove_json(
        "@property disj_false forall(x: f32):\n  ((x >= 100.0) || (x >= 200.0))\n",
        "smt-only",
    );
    assert_eq!(
        output.status.code(),
        Some(1),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "disj_false");
    assert_eq!(props[0]["status"], "failed");
    assert_eq!(props[0]["proof_tier"], "smt");
    assert!(
        props[0].get("counterexample").is_some(),
        "expected a counterexample, got {:?}",
        props[0]
    );
}

// ---------------------------------------------------------------------------
// The connective is no weaker than its operands: under `--tier auto` a
// connective goal proves at SMT exactly like a single bound.
// ---------------------------------------------------------------------------

/// Under `--tier auto`, the conjunction proves at the SMT tier (it does not
/// fall to Tier C fuzz, and it does not surface an `error` status — the two
/// warts the issue flagged for the *malformed* keyword form do not occur
/// for the canonical `&&` form).
#[cfg(feature = "smt")]
#[test]
fn conjunction_goal_auto_proves_at_smt_not_fuzz_not_error() {
    let output = run_prove_json(
        "@property conj_auto forall(x: f32):\n  ((x * x >= 0.0) && (x * x >= 0.0))\n",
        "auto",
    );
    assert!(
        output.status.success(),
        "stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    assert_eq!(props.len(), 1, "records: {props:?}");
    assert_eq!(props[0]["name"], "conj_auto");
    assert_eq!(props[0]["status"], "passed");
    assert_eq!(props[0]["proof_tier"], "smt");
}

// ---------------------------------------------------------------------------
// Negative: the non-spec KEYWORD-INFIX form `(a) and (b)` is a clean
// parse/type error — never a false pass, never a panic. Surf has no
// `and`/`or` infix operator (only `&&`/`||`), so `(a) and (b)` parses as a
// juxtaposition application and fails to type-check. This must hold in BOTH
// the default and the `--features smt` build, so it is NOT smt-gated.
// ---------------------------------------------------------------------------

/// `(a) and (b)` keyword-infix is rejected, not silently passed. The exit
/// code is non-zero and the status is never `passed`.
#[test]
fn keyword_infix_and_is_a_clean_error_not_a_false_pass() {
    let output = run_prove_json(
        "@property kw_and forall(x: f32):\n  ((x * x >= 0.0) and (x * x >= 0.0))\n",
        "auto",
    );
    assert_ne!(
        output.status.code(),
        Some(0),
        "keyword-infix `and` must not pass; stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    // Either a property record with a non-pass status, or no property
    // record at all (a discovery/check error) — but NEVER a `passed`.
    for record in &props {
        assert_ne!(
            record["status"], "passed",
            "keyword-infix `and` must never prove: {record:?}"
        );
    }
    // The combined stdout/stderr must not look like a clean success.
    let combined = format!(
        "{}{}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    assert!(
        !combined.contains("\"status\":\"passed\""),
        "keyword-infix `and` must never prove; combined={combined}"
    );
}

/// `(a) or (b)` keyword-infix is likewise rejected, not silently passed.
#[test]
fn keyword_infix_or_is_a_clean_error_not_a_false_pass() {
    let output = run_prove_json(
        "@property kw_or forall(x: f32):\n  ((x * x >= 0.0) or (x * x >= 1.0))\n",
        "auto",
    );
    assert_ne!(
        output.status.code(),
        Some(0),
        "keyword-infix `or` must not pass; stdout={}\nstderr={}",
        String::from_utf8_lossy(&output.stdout),
        String::from_utf8_lossy(&output.stderr)
    );
    let props = property_records(&output.stdout);
    for record in &props {
        assert_ne!(
            record["status"], "passed",
            "keyword-infix `or` must never prove: {record:?}"
        );
    }
}
