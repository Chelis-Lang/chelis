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

fn check_deep(source: &str) -> (std::process::ExitStatus, Value) {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rank_honesty.dp");
    fs::write(&path, source).expect("write Deep source");
    let output = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run chelis check on Deep source");
    let report = serde_json::from_slice(&output.stdout).unwrap_or_else(|error| {
        panic!(
            "Deep check must emit JSON ({error}); stdout={} stderr={}",
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

#[test]
fn lexical_floor_div_parameter_is_not_reclassified_by_identity_registry() {
    let source = "module Repro.ShadowedIdentity\n\
def lift(x: tensor[n, f32]) -> tensor[2, n, f32] = expand(x, 0i32, 2i64)\n\
def apply(floor_div: (tensor[n, f32] -> tensor[2, n, f32]), x: tensor[n, f32]) -> tensor[2, n, f32] = {\n\
  s = stride(x, 2i64)\n\
  e = expand(x, 0i32, 2i64)\n\
  add(floor_div(s), e)\n\
}\n\
out = apply(lift, to_tensor([1.0, 2.0, 3.0, 4.0]))\n";
    let (status, report) = check(source);
    assert!(
        status.success(),
        "lexical precedence must win over the central builtin registry: {report}"
    );
    assert_eq!(report["score"], 1, "clean control must score perfectly");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn authored_rank_only_prefix_dimension_is_not_internal_validator_state() {
    let source = "module Repro.AuthoredRankName\n\
def convolve(\n\
  x: tensor[__chelis_rank_only_axis_0, 3, 8, 8, f32],\n\
  k: tensor[8, 3, 3, 3, f32],\n\
) -> tensor[__chelis_rank_only_axis_0, 8, 6, 6, f32] = conv2d(&x, &k, 1i32, 0)\n";
    let (status, report) = check(source);
    assert!(
        status.success(),
        "a legal authored dimension name must not alias private rank facts: {report}"
    );
    assert_eq!(report["score"], 1, "clean control must score perfectly");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn authored_deep_type_metadata_cannot_override_checker_owned_rank_facts() {
    let control = "(def {}
  f
  (fn {}
    (params {} (x {type: (t-tensor {} (d-name {} n) (t-prim {} f32))}))
    (let {}
      (bind {}
        s
        (app {} (var {} stride) (var {} x) (lit {type: (t-prim {} int64)} 2)))
      (let {}
        (bind {}
          e
          (app {}
            (var {} expand)
            (var {} x)
            (lit {type: (t-prim {} int32)} 0)
            (lit {type: (t-prim {} int64)} 2)))
        (app {} (var {} add) (var {} s) (var {} e))))))
";
    let forged = control.replace(
        "(app {} (var {} add) (var {} s) (var {} e))",
        "(app {}
          (var {} add)
          (var {type: (t-tensor {} (d-lit {} 2) (d-name {} n) (t-prim {} f32))} s)
          (var {} e))",
    );

    for (label, source) in [("control", control), ("forged", forged.as_str())] {
        let (status, report) = check_deep(source);
        assert!(
            !status.success(),
            "{label} rank mismatch must fail even when authored metadata claims another rank: {report}"
        );
        assert!(
            report["errors"].as_array().is_some_and(|errors| errors
                .iter()
                .any(|error| error["kind"] == "DimensionMismatch")),
            "{label} must retain the checker-owned dimension mismatch: {report}"
        );
    }
}

/// Rebinding a name to a value whose shape the validator cannot derive must
/// clear the name's previous fact, not inherit it.
///
/// `IrTypeEnv` holds only top-level defs, so any entry standing under a
/// let-bound name describes a different binding. Leaving it in place made a
/// rebinding read the earlier binding's rank, and the identity-rank validator
/// then rejected a valid program (chelis#668 round-6 F1). The rebound `a` here
/// is rank-2 like `b`; before the repair the stale rank-1 fact from the dead
/// first binding produced "got ranks 1 and 2".
#[test]
fn rebinding_to_a_nonderivable_value_drops_the_stale_rank_fact() {
    let source = "module Repro.Rebind\n\
                  def f(x: tensor[n, f32]) = {\n\
                    a = stride(x, 2i64)\n\
                    a = normalize(expand(x, 0i32, 2i64))\n\
                    b = expand(x, 0i32, 2i64)\n\
                    add(a, b)\n\
                  }\n\
                  out = f(to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))\n";
    let (status, report) = check(source);
    assert!(
        status.success(),
        "a rebinding must not inherit the previous binding's rank: {report}"
    );
    assert_eq!(
        report["score"], 1,
        "a valid rebinding must score perfectly: {report}"
    );
    assert_eq!(
        report["errors"],
        serde_json::json!([]),
        "perfect success must have no errors: {report}"
    );
}

/// Negative parity for the repair above: clearing the stale fact must not
/// become "stop checking rebound names". Here the rebound `a` IS derivable,
/// at rank 1, and disagrees with the rank-2 `b`, so the mismatch must still
/// be reported.
#[test]
fn rebinding_to_a_derivable_rank_still_rejects_a_genuine_mismatch() {
    let source = "module Repro.RebindNegative\n\
                  def f(x: tensor[n, f32]) = {\n\
                    a = expand(x, 0i32, 2i64)\n\
                    a = stride(x, 2i64)\n\
                    b = expand(x, 0i32, 2i64)\n\
                    add(a, b)\n\
                  }\n\
                  out = f(to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))\n";
    let (status, report) = check(source);
    assert!(
        !status.success(),
        "a rebinding to a provably divergent rank must still fail: {report}"
    );
    assert!(
        report["score"].as_f64().is_some_and(|score| score < 1.0),
        "a genuine mismatch must not receive a perfect score: {report}"
    );
    assert!(
        report["errors"].as_array().is_some_and(|errors| errors
            .iter()
            .any(|error| error["kind"] == "DimensionMismatch")),
        "expected the checker-owned dimension mismatch: {report}"
    );
}
