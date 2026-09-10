//! chelis#668 at the `chelis check` command: a checked elementwise
//! application must not combine two positive-rank tensors whose ranks differ.
//!
//! The source rule is `spec/05-risc-primitives.md` section 1.2's hard
//! no-broadcast contract, and one mechanism implements it: unification. The
//! post-inference identity-rank validator that used to compare ranks beside it
//! is gone (`spec/design/checker_totality.md` section PP5, D6 (d)); it derived
//! a rank for `stride`, `expand`, and `insert` in a private carrier, ran on
//! this ingress only, and duplicated a verdict unification had already
//! reached. The C emitters keep their runtime guard for malformed or
//! dynamically bound IR (R3, R4).
//!
//! Every row here reaches the checker through the CLI, so what it pins is the
//! command's exit status, score, and error kinds. The diagnostic texts and the
//! `check_typed_program` ingress belong to
//! `chelis-types/tests/issue_668_rank_agreement_is_unification.rs`, which is
//! the completion oracle.
//!
//! # Evidentiary status
//!
//! DISPOSITION LOCKS, all of them, and measured as such: the deletion changed
//! no verdict in this file. Each rejection was reported twice on this ingress
//! before it and once after, so what these rows lock is that the surviving
//! mechanism still refuses the same programs. The regression assertion for the
//! duplication itself is the ingress-agreement row in the `chelis-types` file,
//! which is red on the pre-deletion tree.
//!
//! The exception is `the_expand_built_reproducer_is_loud_at_run_time`, which
//! is new and covers the other half of the disposition: the programs PP5 used
//! to reject are well typed, and their loud outcome is section 2.4.1's
//! `Domain` trap.

mod common;

use assert_cmd::Command;
use common::{gcc_available, link_generated};
use serde_json::Value;
use std::fs;
use std::path::Path;
use std::process::Command as StdCommand;
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
           e = insert(x, 0i32, 2i64)\n\
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
                    // chelis#668's subject is that a PROVABLE rank mismatch is a
                    // dimension error rather than a soft score. Since chelis#1277
                    // fixed `insert`'s rank at the call, the operand's rank is
                    // known before the elementwise op, so unification reports the
                    // mismatch there. The disjunction survives the deletion of the
                    // identity-rank validator, whose phrasing the first arm
                    // matched: `spec/04` \u{00a7}4.7 asks for the earliest proof, not a
                    // particular phrasing.
                    (message.contains("elementwise") && message.contains("rank"))
                        || message.contains("tensor rank mismatch")
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
    assert_eq!(report["score"], 1.0, "clean control must score perfectly");
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
    assert_eq!(report["score"], 1.0, "clean control must score perfectly");
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
fn inline_floor_div_rank_is_unified_in_both_argument_orders() {
    for rhs in ["add(floor_div(s, s), e)", "add(e, floor_div(s, s))"] {
        let (status, report) = check(&rank_divergent_source(rhs));
        assert!(
            !status.success(),
            "unification must refuse the divergent floor_div pair in `{rhs}`: {report}"
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
fn let_bound_floor_div_rank_is_unified() {
    let source = rank_divergent_source("fd = floor_div(s, s)\n  add(fd, e)");
    let (status, report) = check(&source);
    assert!(
        !status.success(),
        "unification must refuse the divergent let-bound floor_div pair: {report}"
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
    assert_eq!(report["score"], 1.0, "clean control must score perfectly");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn a_lexically_shadowed_floor_div_parameter_keeps_its_declared_type() {
    let source = "module Repro.ShadowedIdentity\n\
def lift(x: tensor[n, f32]) -> tensor[2, n, f32] = insert(x, 0i32, 2i64)\n\
def apply(floor_div: (tensor[n, f32] -> tensor[2, n, f32]), x: tensor[n, f32]) -> tensor[2, n, f32] = {\n\
  s = stride(x, 2i64)\n\
  e = insert(x, 0i32, 2i64)\n\
  add(floor_div(s), e)\n\
}\n\
out = apply(lift, to_tensor([1.0, 2.0, 3.0, 4.0]))\n";
    let (status, report) = check(source);
    assert!(
        status.success(),
        "a parameter shadowing a builtin name keeps its declared type: {report}"
    );
    assert_eq!(report["score"], 1.0, "clean control must score perfectly");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn an_authored_rank_only_prefix_dimension_is_an_ordinary_name() {
    let source = "module Repro.AuthoredRankName\n\
def convolve(\n\
  x: tensor[__chelis_rank_only_axis_0, 3, 8, 8, f32],\n\
  k: tensor[8, 3, 3, 3, f32],\n\
) -> tensor[__chelis_rank_only_axis_0, 8, 6, 6, f32] = conv(&x, &k, [1i64, 1i64], [(0i64, 0i64), (0i64, 0i64)])\n";
    let (status, report) = check(source);
    assert!(
        status.success(),
        "an unusual authored dimension name is an ordinary name: {report}"
    );
    assert_eq!(report["score"], 1.0, "clean control must score perfectly");
    assert_eq!(report["errors"], serde_json::json!([]));
}

#[test]
fn authored_deep_type_metadata_cannot_override_the_inferred_rank() {
    // The rank disagreement is `stride`'s rank-1 result against `insert`'s
    // rank-2 one. It used to be spelled `expand`, which had two candidate
    // result shapes; under `spec/04-type-system.md` section 4.7.2 `expand` is
    // the same-rank broadcast, so that spelling now agrees with `stride` at
    // rank 1 and the program has no rank disagreement left to forge. `insert`
    // is the operation that raises the rank, and the forged metadata claims
    // exactly the rank-2 shape `insert` produces.
    //
    // Both halves are MEASURED, not predicted: authored `type:` metadata on a
    // `var` that disagrees with its binding does not displace the inferred
    // type, so the forged program is refused for the same reason as the
    // control. That was the completion design's one open question about this
    // row ("Not established", second bullet).
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
            (var {} insert)
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
/// clear the name's previous entry, not inherit it.
///
/// `IrTypeEnv` holds only top-level defs, so any entry standing under a
/// let-bound name describes a different binding. Leaving it in place made a
/// rebinding read the earlier binding's rank, and the identity-rank validator
/// then rejected a valid program (chelis#668 round-6 F1). That consumer is
/// gone, so this row can no longer go red the way it originally did; the
/// environment it exercises still feeds the exact-shape `conv` validators,
/// where a stale entry would be the same defect with a different consumer.
#[test]
fn rebinding_to_a_nonderivable_value_does_not_inherit_the_previous_shape() {
    let source = "module Repro.Rebind\n\
                  def f(x: tensor[n, f32]) = {\n\
                    a = stride(x, 2i64)\n\
                    a = relu(insert(x, 0i32, 2i64))\n\
                    b = insert(x, 0i32, 2i64)\n\
                    add(a, b)\n\
                  }\n\
                  out = f(to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0]))\n";
    let (status, report) = check(source);
    assert!(
        status.success(),
        "a rebinding must not inherit the previous binding's rank: {report}"
    );
    assert_eq!(
        report["score"], 1.0,
        "a valid rebinding must score perfectly: {report}"
    );
    assert_eq!(
        report["errors"],
        serde_json::json!([]),
        "perfect success must have no errors: {report}"
    );
}

/// Negative parity for the row above: clearing the stale entry must not
/// become "stop checking rebound names". Here the rebound `a` is rank 1 and
/// disagrees with the rank-2 `b`, so unification must still refuse it.
#[test]
fn rebinding_to_a_derivable_rank_still_rejects_a_genuine_mismatch() {
    let source = "module Repro.RebindNegative\n\
                  def f(x: tensor[n, f32]) = {\n\
                    a = insert(x, 0i32, 2i64)\n\
                    a = stride(x, 2i64)\n\
                    b = insert(x, 0i32, 2i64)\n\
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

// ── The other half of the disposition: the loud outcome ───────────────────

/// A chelis#668 reproducer built from `expand` rather than `insert`, and the
/// run-time guard that makes it safe to accept.
///
/// `spec/04-type-system.md` section 4.7.2 gives `expand` one result shape and
/// `spec/05-risc-primitives.md` section 2.4 makes it the same-rank unit-extent
/// broadcast, so `stride(e, 1i64)` and `e` are both rank 1 and the program is
/// well typed. That is not a hole: the operation is a claim that the operand's
/// extent at the axis is 1, and section 2.4.1 states the outcome when the
/// claim is false. A literal violation is a check-time type error; a symbolic
/// one, which is this program, is the section 4.7 runtime extent guard, and it
/// traps `Domain`.
///
/// The `stride` operand is taken off `e` rather than off `x` so that the two
/// operand shapes agree by construction. Taken off `x` they agree only when
/// the claim holds, and a failure could then be an ordinary shape mismatch
/// rather than the guard, which is the confusion this row exists to remove.
///
/// REGRESSION coverage for the acceptance, not for the deletion: this program
/// checks clean on both the pre- and post-deletion trees (the deletion changed
/// no verdict), and the assertion that has never held on any older tree is the
/// trap, which chelis#1277 S2b landed. What the row adds is the pairing: the
/// exact program PP5's validator used to reject is accepted, and its loudness
/// is named and executed on both lanes rather than asserted in prose.
///
/// If either lane stops trapping, this row fails rather than being softened:
/// the acceptance in `issue_668_rank_agreement_is_unification` depends on it.
#[test]
fn the_expand_built_reproducer_is_loud_at_run_time() {
    let program = |input: &str| {
        format!(
            "module Repro.Issue668Loud\n\
             sig f: tensor[n, f32] -> tensor[u, f32]\n\
             def f(x) = {{\n\
               e = expand(x, 0i32, 3i64)\n\
               s = stride(e, 1i64)\n\
               add(s, e)\n\
             }}\n\
             out = f(to_tensor([{input}]))\n"
        )
    };
    // The operand's extent at axis 0 is 1, so the claim holds.
    let satisfied = program("cast(5.0, f32)");
    // The operand's extent at axis 0 is 2, so the claim is refuted at run time.
    let refuted = program("cast(1.0, f32), cast(2.0, f32)");

    // Both are well typed: this is the program PP5's validator used to refuse.
    for (label, source) in [("satisfied", &satisfied), ("refuted", &refuted)] {
        let (status, report) = check(source);
        assert!(
            status.success(),
            "{label}: a same-rank `expand` beside a rank-1 operand is well \
             typed; a refuted unit-extent claim is a run-time outcome, not a \
             check-time one, unless the operand extent is a literal \
             (`spec/05-risc-primitives.md` section 2.4.1): {report}"
        );
        assert_eq!(report["score"], 1.0, "{label}: must score perfectly");
        assert_eq!(report["errors"], serde_json::json!([]));
    }

    let dir = tempdir().expect("tempdir");

    // The evaluator lane. A `def main`-shaped program reaches the host
    // interpreter, so this is the guard a `chelis eval` of user source meets.
    let ok_path = dir.path().join("satisfied.ch");
    fs::write(&ok_path, &satisfied).expect("write");
    let ok_eval = eval(&ok_path);
    assert!(
        ok_eval.status.success(),
        "the satisfied claim must execute: {}",
        String::from_utf8_lossy(&ok_eval.stderr)
    );
    assert!(
        String::from_utf8_lossy(&ok_eval.stdout).contains("shape=[3], data=[10.0, 10.0, 10.0]"),
        "the satisfied claim broadcasts exactly: {}",
        String::from_utf8_lossy(&ok_eval.stdout)
    );

    let bad_path = dir.path().join("refuted.ch");
    fs::write(&bad_path, &refuted).expect("write");
    let bad_eval = eval(&bad_path);
    let bad_stderr = String::from_utf8_lossy(&bad_eval.stderr).to_string();
    assert!(
        !bad_eval.status.success(),
        "a refuted unit-extent claim must not evaluate: {}",
        String::from_utf8_lossy(&bad_eval.stdout)
    );
    assert!(
        bad_stderr.contains("numeric trap: domain in load at int64"),
        "the [04-NUM-9] rendering, at the extent's own dtype, with the \
         section 4.7 slot for an all-interface guard: {bad_stderr}"
    );
    assert!(
        bad_stderr.contains("claimed = 1") && bad_stderr.contains("axis 0 = 2"),
        "the guard names the claim and the value observed: {bad_stderr}"
    );

    // The C lane. Same two programs, built and executed.
    if !gcc_available() {
        return;
    }
    let (ok_ran, ok_output) = build_link_run(&dir, "satisfied", &satisfied);
    assert!(ok_ran, "the satisfied claim must run on C: {ok_output}");
    assert!(
        ok_output.contains("shape=[3], data=[10.0, 10.0, 10.0]"),
        "the C lane agrees with the evaluator: {ok_output}"
    );
    let (bad_ran, bad_output) = build_link_run(&dir, "refuted", &refuted);
    assert!(
        !bad_ran,
        "a refuted unit-extent claim must abort on C: {bad_output}"
    );
    assert!(
        bad_output.contains("numeric trap: domain in load at int64")
            && bad_output.contains("claimed = 1")
            && bad_output.contains("axis 0 = 2"),
        "the C lane renders the same guard: {bad_output}"
    );
}

/// Build a fixture to C, link it against the emitted runtime, run it, and
/// return whether it exited zero plus its combined output. A build or link
/// failure panics: those are defects in the fixture or the emitter, never the
/// behaviour under test.
fn build_link_run(dir: &tempfile::TempDir, stem: &str, source: &str) -> (bool, String) {
    let path = dir.path().join(format!("{stem}.ch"));
    fs::write(&path, source).expect("write fixture");
    let out_dir = dir.path().join(format!("{stem}-out"));
    let build = Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args([
            "build",
            "--allow-style-violations",
            path.to_str().expect("UTF-8 path"),
            "--target",
            "c",
            "-o",
            out_dir.to_str().expect("UTF-8 path"),
        ])
        .output()
        .expect("run chelis build");
    assert!(
        build.status.success(),
        "build failed: {}",
        String::from_utf8_lossy(&build.stderr)
    );
    let linked = link_generated(&out_dir, &format!("{stem}.c"), stem);
    assert!(linked.success(), "link failed: {linked}");
    let run = StdCommand::new(out_dir.join(stem))
        .output()
        .expect("run the linked binary");
    let mut combined = String::from_utf8_lossy(&run.stdout).to_string();
    combined.push_str(&String::from_utf8_lossy(&run.stderr));
    (run.status.success(), combined)
}

fn eval(path: &Path) -> std::process::Output {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().expect("UTF-8 path")])
        .output()
        .expect("run chelis eval")
}
