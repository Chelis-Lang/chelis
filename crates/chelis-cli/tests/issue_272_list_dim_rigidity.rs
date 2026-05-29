//! Issue Chelis-Lang/chelis#272: `List[tensor[...]]` Cons-join erased
//! named dims, defeating the §4.4 rigid declared-dim-parameter guard.
//!
//! Building a list literal of tensors (`[a, b]`) desugars to a
//! `Cons(a, Cons(b, Nil))` chain. The Cons element-typing path (added
//! for #218 concat ergonomics) produced a per-axis *join* that widened
//! any mismatched axis to `Wildcard`. For two genuinely-mismatched
//! *concrete literals* that widening is the deliberate #218 behavior
//! (so `concat([a, b], axis)` accepts ragged concrete axes). But the
//! same `_ => Wildcard` arm also fired for:
//!
//!   * two named dim *variables* (`tensor[k]` joined with `tensor[m]`),
//!     erasing the evidence the rigidity guard needs, and
//!   * a `(named-var, concrete-literal)` pair.
//!
//! The wildcard then satisfied an explicit return annotation naming a
//! rigid dim (`List[tensor[k, f32]]`), so a body that violates the
//! §4.4 rigid-distinct-dim guarantee type-checked — but *only* when the
//! violation was wrapped in a list. The equivalent non-list code is
//! correctly rejected (`check_declared_dvars_rigid`).
//!
//! These tests lock the asymmetry closed while keeping the #218
//! bare-`concat` ergonomics (mismatched concrete axes still widen so
//! `concat` accepts them — see `issue_218_to_tensor_in_grad_body.rs`).
//!
//! Spec: `spec/04-type-system.md` §4.4 (rigid dims) and §4.5.2
//! (list-literal dimension joining; sibling of §4.5.1 rank-uniformity
//! from chelis#255).

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

fn error_messages(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["message"].as_str().unwrap_or("").to_string())
        .collect()
}

fn error_kinds(json: &Value) -> Vec<String> {
    json["errors"]
        .as_array()
        .expect("errors should be a json array")
        .iter()
        .map(|e| e["kind"].as_str().unwrap_or("").to_string())
        .collect()
}

// =================================================================
// Baseline anchor: the non-list case is (and stays) rejected.
// =================================================================

#[test]
fn issue_272_baseline_nonlist_distinct_rigid_dims_rejects() {
    // The §4.4 reference TYPE ERROR. The list cases below must reject
    // with the same diagnostic *class* (DimensionMismatch from the
    // rigid-dim guard).
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nonlist.ch");
    write_file(
        &path,
        "def f[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = y\n",
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "baseline non-list distinct-rigid-dim case must reject with \
         DimensionMismatch; got {:?}",
        error_messages(&json),
    );
}

// =================================================================
// Scenario A (#272 headline): distinct rigid dims, body violates the
// declared element dim. Wrapped in a list it used to be accepted.
// =================================================================

#[test]
fn issue_272_scenario_a_distinct_rigid_dims_in_list_rejects() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scenario_a.ch");
    write_file(
        &path,
        "def make[k, m](a: tensor[k, f32], b: tensor[m, f32]) -> List[tensor[k, f32]] = [a, b]\n",
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "scenario A: distinct rigid dims k != m in a list-literal body \
         must reject with DimensionMismatch (same class as the non-list \
         case); got messages {:?}",
        error_messages(&json),
    );
}

// =================================================================
// Scenario B (#272): heterogeneous *concrete* element lengths cannot
// satisfy a declared uniform rigid dim `k`. The bare-concat join
// widens 2/3 to Wildcard, but that wildcard must not silently satisfy
// the uniformity promise of `List[tensor[k]]`.
// =================================================================

#[test]
fn issue_272_scenario_b_heterogeneous_concrete_vs_rigid_dim_rejects() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("scenario_b.ch");
    write_file(
        &path,
        "def make[k](a: tensor[2, f32], b: tensor[3, f32]) -> List[tensor[k, f32]] = [a, b]\n",
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "scenario B: a heterogeneous concrete-length list body \
         (tensor[2], tensor[3]) cannot satisfy a declared uniform \
         element dim `k`; must reject with DimensionMismatch; got \
         messages {:?}",
        error_messages(&json),
    );
}

// =================================================================
// Positive parity: a genuinely uniform list still type-checks.
// =================================================================

#[test]
fn issue_272_uniform_rigid_dim_list_type_checks() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("uniform.ch");
    write_file(
        &path,
        "def make[k](a: tensor[k, f32], b: tensor[k, f32]) -> List[tensor[k, f32]] = [a, b]\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "a genuinely uniform list `[a, b]` over a single rigid dim `k` \
         must still type-check; got {errs:?}",
    );
}

#[test]
fn issue_272_uniform_concrete_dim_list_type_checks() {
    // Matching concrete lengths must still type-check (the join keeps
    // the shared Lit, no widening, no rigid-dim violation).
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("uniform_concrete.ch");
    write_file(
        &path,
        "def make(a: tensor[2, f32], b: tensor[2, f32]) -> List[tensor[2, f32]] = [a, b]\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "a uniform concrete-length list must still type-check; got {errs:?}",
    );
}

// =================================================================
// Positive parity (#218 lock): bare `concat([...], axis)` over
// differing concrete axes still type-checks. This is the deliberate
// #218 ergonomic the fix must NOT re-break. No annotation, no dim
// params -> the join widens the differing concrete axis to Wildcard
// and nothing names a rigid dim, so no #272 check fires.
// =================================================================

#[test]
fn issue_272_bare_concat_differing_concrete_axes_still_type_checks() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bare_concat.ch");
    write_file(
        &path,
        "out = concat([to_tensor([[1.0, 2.0, 3.0]]),\n\
                       to_tensor([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])], 0)\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "#218 lock: bare concat over differing concrete axes must still \
         type-check; got {errs:?}",
    );
}

#[test]
fn issue_272_bare_heterogeneous_list_without_annotation_still_type_checks() {
    // Without a return annotation naming a rigid dim, a heterogeneous
    // bare list is a defensible "I don't know the shape" wildcard
    // result. The #272 fix only bites when an annotation promises
    // uniformity via a rigid/named dim. Here the differing concrete
    // axis simply widens to Wildcard and the binding has no rigid-dim
    // promise to violate.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bare_list.ch");
    write_file(
        &path,
        "out = concat([to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0, 5.0])], 0)\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "bare heterogeneous concrete list (no rigid-dim annotation) must \
         still type-check; got {errs:?}",
    );
}
