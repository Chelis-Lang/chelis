//! WS-A5 acceptance tests: precision polymorphism in the type system.
//!
//! Pin the bug surfaced during the WS-C investigation (poly_id +
//! use_mismatch silently type-checking) and the new positive shapes
//! enabled by adding a precision-variable slot to `Type::Tensor`.
//!
//! See `spec/04-type-system.md` §5.8 and `spec/02-surf-syntax.md` for
//! the contextual desugar rule that turns a lowercase non-primitive
//! identifier in the precision slot of a `tensor[...]` type into a
//! sig-quantified type variable.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

// Issue #207: `chelis check` now exits non-zero when the JSON
// `errors` array is non-empty. The WS-C blocker test below
// deliberately expects errors, so this helper drops the exit-code
// assertion and captures stdout regardless.
fn run_json_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

/// The WS-C blocker reproducer: a sig with a polymorphic precision
/// slot must enforce that callers actually instantiate it consistently.
/// Before WS-A5, `poly_id`'s sig accepted any precision and the
/// `use_mismatch` body silently downgraded the precision invariant to
/// a wildcard. After WS-A5, the precision slot is a real type variable
/// that unifies with the call-site precisions, so the shape mismatch
/// (input i32, output declared f32) must surface.
#[test]
fn ws_c_blocker_polymorphic_precision_does_not_silently_accept_mismatch() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("ws_c_blocker.ch");
    write_file(
        &path,
        r#"sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]
def poly_id(x) = x
def use_mismatch(x: tensor[3, i32]) -> tensor[3, f32] = poly_id(x)
"#,
    );

    let json = run_json_check(&path);
    let errors = json["errors"]
        .as_array()
        .expect("errors should be a json array");
    assert!(
        !errors.is_empty(),
        "WS-C blocker: poly_id with mismatched precision must produce \
         a check error, got clean output {json}"
    );
    assert!(
        json["score"].as_f64().unwrap_or(1.0) < 1.0,
        "WS-C blocker: score must be below 1.0 for the precision \
         mismatch case, got {json}"
    );
    // The diagnostic kind may be either PrecisionMismatch (raised by
    // the unify path that hits the precision-var directly) or
    // TypeMismatch (raised by the def-body-vs-declared-sig check
    // path that observes the inferred body type carries i32 once
    // the precision var is bound to i32 by the input). Both shapes
    // mention `f32` and `i32` in the message, which is what we
    // actually want a user to see; the kind tag is a downstream
    // implementation detail.
    let any_mentions_both_precisions = errors.iter().any(|e| {
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        msg.contains("f32") && msg.contains("i32")
    });
    assert!(
        any_mentions_both_precisions,
        "WS-C blocker: at least one error should mention both f32 and \
         i32 so the user can see the mismatched precisions, got {errors:?}"
    );
}

/// Positive case: the same polymorphic sig accepts two distinct
/// concrete instantiations across different call sites. Each call
/// internally has matching precisions; only the precisions vary
/// between calls.
#[test]
fn polymorphic_precision_accepts_distinct_consistent_instantiations() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("poly_distinct.ch");
    write_file(
        &path,
        r#"sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]
def poly_id(x) = x
def use_f32(x: tensor[3, f32]) -> tensor[3, f32] = poly_id(x)
def use_int32(x: tensor[3, i32]) -> tensor[3, i32] = poly_id(x)
"#,
    );

    let json = run_json_check(&path);
    let errors = json["errors"]
        .as_array()
        .expect("errors should be a json array");
    assert!(
        errors.is_empty(),
        "polymorphic sig with two consistent instantiations should \
         type-check, got errors {errors:?}"
    );
    assert!(
        (json["score"].as_f64().unwrap_or(0.0) - 1.0).abs() < f64::EPSILON,
        "score must be 1.0 for the consistent-instantiations case, got {json}"
    );
}

/// Positive case: the precision variable can be the only polymorphic
/// component; the dimensions stay concrete.
#[test]
fn polymorphic_precision_only_with_concrete_dims() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("poly_prec_only.ch");
    write_file(
        &path,
        r#"sig same_prec[p]: tensor[3, p] -> tensor[3, p]
def same_prec(x) = x
def use_bf16(x: tensor[3, bf16]) -> tensor[3, bf16] = same_prec(x)
"#,
    );

    let json = run_json_check(&path);
    let errors = json["errors"]
        .as_array()
        .expect("errors should be a json array");
    assert!(
        errors.is_empty(),
        "polymorphic-precision-only sig should type-check, got {errors:?}"
    );
}

/// Negative case: a sig that uses two distinct precision variables
/// followed by a call that instantiates them inconsistently must
/// surface the inconsistency.
#[test]
fn polymorphic_precision_rejects_inconsistent_within_call() {
    // p and q are independent type variables in the sig. The first
    // arg is tensor[d, p], the result is tensor[d, p]; both must
    // share the SAME precision. Calling with i32 input but
    // declaring an f32 output forces p to bind to two different
    // precisions, which must fail.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("poly_inconsistent.ch");
    write_file(
        &path,
        r#"sig poly_id[d, p]: tensor[d, p] -> tensor[d, p]
def poly_id(x) = x
def break_it(x: tensor[3, i32]) -> tensor[3, f32] = poly_id(x)
"#,
    );

    let json = run_json_check(&path);
    let errors = json["errors"]
        .as_array()
        .expect("errors should be a json array");
    assert!(
        !errors.is_empty(),
        "inconsistent precision instantiation must error, got {json}"
    );
}

/// Concrete-precision baseline: the equivalent fully-concrete sig
/// already errored before WS-A5; verify it still does, so we know the
/// existing PrecisionMismatch path is intact.
#[test]
fn concrete_precision_still_rejects_mismatch_baseline() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concrete_baseline.ch");
    write_file(
        &path,
        r#"sig f32_id[d]: tensor[d, f32] -> tensor[d, f32]
def f32_id(x) = x
def break_it(x: tensor[3, i32]) -> tensor[3, f32] = f32_id(x)
"#,
    );

    let json = run_json_check(&path);
    let errors = json["errors"]
        .as_array()
        .expect("errors should be a json array");
    assert!(
        !errors.is_empty(),
        "concrete-precision sig must still error on precision mismatch, got {json}"
    );
}
