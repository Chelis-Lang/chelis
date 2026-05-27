//! Issue Chelis-Lang/chelis#255: `List[tensor[k, f32]]` with mixed-rank
//! elements rejects with `DimensionMismatch: list element rank mismatch`.
//!
//! The dim slot `k` in `tensor[k, f32]` is a dimension variable, not a
//! shape-vector variable, so the list type fixes a single rank for every
//! element (spec/04-type-system.md §4.5.1 "Rank-Uniform List[tensor[...]]
//! Elements"). The fix landed in this issue is documentation +
//! diagnostic-quality only: the rule is now spelled out in §4.5.1 and the
//! `DimensionMismatch` message carries an actionable hint pointing at the
//! reshape/flatten remediation.
//!
//! Positive test: a rank-uniform `List[tensor[k, f32]]` with matching ranks
//! type-checks cleanly (and continues to do so after this change).
//!
//! Negative test: a mixed-rank `List[tensor[k, f32]]` is rejected with
//! `DimensionMismatch`, and the message carries the new hint pointing at
//! the reshape/flatten remediation so users are not left guessing.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn run_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .output()
        .expect("run chelis check");
    serde_json::from_slice(&output.stdout).expect("check output should be json")
}

fn errors(json: &Value) -> Vec<&Value> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .collect()
}

fn error_messages(json: &Value) -> Vec<String> {
    errors(json)
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

#[test]
fn issue_255_rank_uniform_list_of_tensor_type_checks() {
    // Positive case: two rank-1 tensors in a `List[tensor[k, f32]]`.
    // The dim variable `k` binds to a (wildcard / matching) single
    // dimension across both elements; the type checker accepts it.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("rank_uniform.ch");
    write_file(
        &path,
        "module Repro.RankUniformList\n\
         def make_uniform() -> List[tensor[k, f32]] = {\n  \
           a = to_tensor([cast(1.0, f32), cast(2.0, f32)])\n  \
           b = to_tensor([cast(3.0, f32), cast(4.0, f32)])\n  \
           [a, b]\n\
         }\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "rank-uniform List[tensor[k, f32]] should type-check; got {errs:?}",
    );
}

#[test]
fn issue_255_mixed_rank_list_of_tensor_rejects_with_actionable_hint() {
    // Negative case: a rank-1 and a rank-2 tensor in the same
    // `List[tensor[k, f32]]`. The checker must continue to reject
    // this as a `DimensionMismatch`, AND the message must carry the
    // hint that `List[tensor[...]]` requires rank-uniform elements
    // and that the remediation is to reshape/flatten elements to a
    // common rank.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("mixed_rank.ch");
    write_file(
        &path,
        "module Repro.MixedRankList\n\
         def make_mixed() -> List[tensor[k, f32]] = {\n  \
           a = to_tensor([cast(1.0, f32), cast(2.0, f32)])\n  \
           b = to_tensor([[cast(1.0, f32), cast(2.0, f32)], [cast(3.0, f32), cast(4.0, f32)]])\n  \
           [a, b]\n\
         }\n",
    );
    let json = run_check(&path);
    let errs = errors(&json);
    assert!(
        !errs.is_empty(),
        "mixed-rank List[tensor[k, f32]] must still reject",
    );
    let kinds: Vec<String> = errs
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "mixed-rank list should surface a DimensionMismatch; got kinds={kinds:?}",
    );

    let msgs = error_messages(&json);
    let rank_msg = msgs
        .iter()
        .find(|m| m.contains("list element rank mismatch"))
        .unwrap_or_else(|| {
            panic!("expected a 'list element rank mismatch' diagnostic; got messages={msgs:?}",)
        });

    // The original wording ("1 dims vs 2 dims") must remain so existing
    // downstream consumers and tests keep parsing.
    assert!(
        rank_msg.contains("1 dims vs 2 dims"),
        "rank-mismatch message should still report the per-side rank counts; got {rank_msg:?}",
    );
    // The new actionable hint must appear in the same message.
    assert!(
        rank_msg.contains("rank-uniform"),
        "rank-mismatch message must include the 'rank-uniform' hint; got {rank_msg:?}",
    );
    assert!(
        rank_msg.to_lowercase().contains("reshape"),
        "rank-mismatch message must point at reshape/flatten as the remediation; got {rank_msg:?}",
    );
    // The hint should cite the spec section that owns the rule, so
    // a user / agent reading the diagnostic can find the docs.
    assert!(
        rank_msg.contains("§4.5.1") || rank_msg.contains("4.5.1"),
        "rank-mismatch message should cite the owning spec section (§4.5.1); got {rank_msg:?}",
    );
}
