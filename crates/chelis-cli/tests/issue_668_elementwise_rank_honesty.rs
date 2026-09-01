//! chelis#668: a checked elementwise application must not combine two
//! positive-rank tensors whose ranks differ.
//!
//! The source rule is spec/05's hard no-broadcast contract.  The checker is
//! the primary boundary; the C emitter retains a defensive runtime guard for
//! malformed or dynamically-bound IR that reaches it without this source
//! check.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use tempfile::tempdir;

fn check(source: &str) -> (std::process::ExitStatus, Value) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rank_honesty.ch");
    fs::write(&path, source).expect("write source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run chelis check");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "check must emit JSON ({error}); stdout={} stderr={}",
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        )
    });
    (output.status, report)
}

fn rank_divergent_source(rhs: &str) -> String {
    format!(
        "module Repro.RankDivergent\n\
         sig f: tensor[n, f32] -> tensor[u, f32]\n\
         def f(x) = {{\n\
           s = stride(x, 2i64)\n\
           e = expand(x, 0i32, 2i64)\n\
           {rhs}\n\
         }}\n\
         out = f(to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))\n"
    )
}

#[test]
fn provable_positive_rank_mismatch_is_a_dimension_error() {
    let (status, report) = check(&rank_divergent_source("add(s, e)"));
    assert!(!status.success(), "rank mismatch must fail check: {report}");
    assert!(
        report["score"].as_f64().is_some_and(|score| score < 1.0),
        "rank mismatch must not receive a perfect score: {report}"
    );
    let errors = report["errors"].as_array().expect("errors array");
    assert!(
        errors.iter().any(|error| {
            error["kind"] == "DimensionMismatch"
                && error["message"].as_str().is_some_and(|message| {
                    message.contains("elementwise") && message.contains("rank")
                })
        }),
        "expected a located elementwise rank diagnostic: {errors:#?}"
    );
}

#[test]
fn matching_runtime_rank_control_still_checks() {
    let (status, report) = check(&rank_divergent_source("add(s, s)"));
    assert!(
        status.success(),
        "matching ranks must remain valid: {report}"
    );
    assert_eq!(report["score"], 1, "clean control must score perfectly");
    assert_eq!(
        report["errors"],
        serde_json::json!([]),
        "perfect success must have no errors"
    );
}

#[test]
fn unary_identity_cannot_erase_rank_evidence() {
    let source = rank_divergent_source("sn = neg(s)\n  add(sn, e)");
    let (status, report) = check(&source);
    assert!(
        !status.success(),
        "rank evidence must survive neg: {report}"
    );
    assert!(
        report["errors"].as_array().is_some_and(|errors| errors
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch")),
        "expected a dimension mismatch after neg: {report}"
    );
}

#[test]
fn binary_identity_cannot_erase_rank_evidence() {
    let source = rank_divergent_source("ss = add(s, s)\n  add(ss, e)");
    let (status, report) = check(&source);
    assert!(
        !status.success(),
        "rank evidence must survive an equal-rank add: {report}"
    );
    assert!(
        report["errors"].as_array().is_some_and(|errors| errors
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch")),
        "expected a dimension mismatch after add: {report}"
    );
}

#[test]
fn matching_rank_identity_chain_still_checks() {
    let source = rank_divergent_source("ss = add(s, s)\n  neg(ss)");
    let (status, report) = check(&source);
    assert!(
        status.success(),
        "matching-rank identity chain failed: {report}"
    );
    assert_eq!(report["score"], 1, "clean control must score perfectly");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn inline_unary_identity_preserves_rank_in_both_argument_orders() {
    for rhs in ["mul(relu(s), e)", "mul(e, relu(s))", "add(e, neg(s))"] {
        let (status, report) = check(&rank_divergent_source(rhs));
        assert!(
            !status.success(),
            "inline unary rank evidence must survive in `{rhs}`: {report}"
        );
        assert!(
            report["errors"].as_array().is_some_and(|errors| errors
                .iter()
                .any(|error| error["kind"] == "DimensionMismatch")),
            "expected a dimension mismatch for `{rhs}`: {report}"
        );
    }
}

#[test]
fn nested_inline_identity_chain_preserves_rank() {
    let (status, report) = check(&rank_divergent_source("add(relu(neg(s)), e)"));
    assert!(
        !status.success(),
        "nested inline identity rank evidence must survive: {report}"
    );
    assert!(
        report["errors"].as_array().is_some_and(|errors| errors
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch")),
        "expected a dimension mismatch after nested identities: {report}"
    );
}

#[test]
fn inline_binary_identity_preserves_rank() {
    let (status, report) = check(&rank_divergent_source("add(mul(s, s), e)"));
    assert!(
        !status.success(),
        "inline binary identity rank evidence must survive: {report}"
    );
    assert!(
        report["errors"].as_array().is_some_and(|errors| errors
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch")),
        "expected a dimension mismatch after inline mul: {report}"
    );
}

#[test]
fn central_identity_registry_drives_inline_floor_div_rank_in_both_orders() {
    for rhs in ["add(floor_div(s, s), e)", "add(e, floor_div(s, s))"] {
        let (status, report) = check(&rank_divergent_source(rhs));
        assert!(
            !status.success(),
            "the central identity registry must preserve floor_div rank in `{rhs}`: {report}"
        );
        assert!(
            report["errors"].as_array().is_some_and(|errors| errors
                .iter()
                .any(|error| error["kind"] == "DimensionMismatch")),
            "expected a dimension mismatch for `{rhs}`: {report}"
        );
    }
}

#[test]
fn central_identity_registry_drives_let_bound_floor_div_rank() {
    let source = rank_divergent_source("fd = floor_div(s, s)\n  add(fd, e)");
    let (status, report) = check(&source);
    assert!(
        !status.success(),
        "let-bound floor_div must retain the registry-owned rank fact: {report}"
    );
    assert!(
        report["errors"].as_array().is_some_and(|errors| errors
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch")),
        "expected a dimension mismatch after let-bound floor_div: {report}"
    );
}

#[test]
fn inline_comparison_cannot_hide_rank_mismatch_in_discarded_binding() {
    let source = rank_divergent_source("ignored = eq(e, neg(s))\n  s");
    let (status, report) = check(&source);
    assert!(
        !status.success(),
        "a discarded comparison must still validate its operand ranks: {report}"
    );
    assert!(
        report["errors"].as_array().is_some_and(|errors| errors
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch")),
        "expected a dimension mismatch in discarded comparison: {report}"
    );
}

#[test]
fn matching_inline_identity_chain_still_checks() {
    let (status, report) = check(&rank_divergent_source("add(relu(neg(s)), mul(s, s))"));
    assert!(
        status.success(),
        "matching inline ranks must remain valid: {report}"
    );
    assert_eq!(report["score"], 1, "clean control must score perfectly");
    assert_eq!(report["errors"], serde_json::json!([]));
}
