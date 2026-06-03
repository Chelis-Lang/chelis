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

// =================================================================
// Join side-effect: the tightened Cons-join resolves a `(concrete,
// wildcard)` pair to whatever the HEAD (first list element) dim
// resolves to, rather than the old unconditional `Wildcard`. The join
// returns `subst.apply_dim(head)`, so the result is head-biased:
//
//   * a concrete/named head ABSORBS a wildcard tail
//     (`[tensor[2], tensor[*]]` -> element `tensor[2]`); but
//   * a wildcard head ERASES a concrete tail
//     (`[tensor[*], tensor[2]]` -> element `tensor[*]`).
//
// This asymmetry is a deliberate-but-narrow consequence of pinning the
// fix to the head slot; these two tests lock it so a future change to
// the join orientation is a conscious decision, not a silent drift.
// (Mismatched *concrete* heads/tails still widen to Wildcard via the
// dedicated arm — see scenario B and the #218 ragged-axis lock.)
// =================================================================

#[test]
fn issue_272_join_concrete_head_absorbs_wildcard_tail_type_checks() {
    // `[tensor[2], tensor[*]]` under a rigid-`k` return.
    //   BEFORE: join -> Wildcard (old `_ => Wildcard`); no list-uniformity
    //           check existed, so the wildcard element was accepted. ACCEPT.
    //   AFTER:  join -> Lit(2) (head wins); element is concrete, not a
    //           wildcard, so the #272 check does not fire. Still ACCEPT.
    // Same verdict, but the inferred element dim tightened from `*` to `2`.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("concrete_head_wildcard_tail.ch");
    write_file(
        &path,
        "def f[k](a: tensor[2, f32], b: tensor[*, f32]) -> List[tensor[k, f32]] = [a, b]\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "a concrete head dim must absorb a wildcard tail and keep the list \
         element concrete (so it satisfies rigid `k`); got {errs:?}",
    );
}

#[test]
fn issue_272_join_wildcard_head_erases_concrete_tail_rejects() {
    // `[tensor[*], tensor[2]]` under a rigid-`k` return: the SAME element
    // multiset as the test above, only reordered.
    //   BEFORE: join -> Wildcard; no check existed. ACCEPT.
    //   AFTER:  join -> Wildcard (head is the wildcard, so the concrete tail
    //           is erased); the wildcard element cannot satisfy a declared
    //           rigid `k`. REJECT (DimensionMismatch).
    // Locks the head-position asymmetry: ordering flips the verdict.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wildcard_head_concrete_tail.ch");
    write_file(
        &path,
        "def f[k](a: tensor[*, f32], b: tensor[2, f32]) -> List[tensor[k, f32]] = [a, b]\n",
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "a wildcard head dim erases a concrete tail, so the list element is a \
         wildcard that cannot satisfy rigid `k`; must reject with \
         DimensionMismatch; got {:?}",
        error_messages(&json),
    );
}

// =================================================================
// Conservative false-positive surface of
// `check_list_elem_rigid_dim_vs_wildcard`: it treats ANY declared
// `Dim::Var` (rigid param) or `Dim::Name` (named symbolic dim) list
// element axis as a uniformity promise, and rejects a wildcard-element
// body against it -- even a single-element list, which is trivially
// "uniform". The escape hatch is an explicit `tensor[*, ..]` element
// annotation. The check is also scoped to a single `List[tensor[..]]`
// level; it does NOT recurse into nested `List[List[tensor[..]]]`.
// =================================================================

#[test]
fn issue_272_single_wildcard_elem_under_rigid_dim_rejects() {
    // `[b]` where `b: tensor[*, f32]`, under a rigid-`k` return.
    //   BEFORE: ACCEPT (wildcard satisfied `k` permissively; no check).
    //   AFTER:  REJECT (DimensionMismatch) -- a wildcard element may not
    //           satisfy a declared rigid `k`, even for a one-element list.
    // This is the conservative narrowing: a genuinely-uniform single
    // wildcard element is now rejected; the author must annotate `tensor[*]`.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("single_wildcard_rigid.ch");
    write_file(
        &path,
        "def f[k](b: tensor[*, f32]) -> List[tensor[k, f32]] = [b]\n",
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "a single wildcard-typed list element must not satisfy a declared \
         rigid `k`; must reject with DimensionMismatch; got {:?}",
        error_messages(&json),
    );
}

#[test]
fn issue_272_single_wildcard_elem_under_named_dim_rejects() {
    // Same body, but the declared element names a symbolic dim `batch`
    // (`Dim::Name`) rather than a rigid param. The check treats a named
    // dim as a uniformity promise too.
    //   BEFORE: ACCEPT.   AFTER: REJECT (DimensionMismatch).
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("single_wildcard_named.ch");
    write_file(
        &path,
        "def f(b: tensor[*, f32]) -> List[tensor[batch, f32]] = [b]\n",
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "a wildcard list element must not satisfy a declared named dim \
         `batch`; must reject with DimensionMismatch; got {:?}",
        error_messages(&json),
    );
}

#[test]
fn issue_272_wildcard_elem_satisfies_explicit_wildcard_annotation() {
    // The escape hatch. Declaring the element axis as an explicit wildcard
    // makes NO uniformity promise, so a wildcard element is accepted.
    //   BEFORE: ACCEPT.   AFTER: ACCEPT (unchanged -- the check only fires
    //   when the declared element names a rigid/named dim).
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("single_wildcard_wildcard_annot.ch");
    write_file(
        &path,
        "def f(b: tensor[*, f32]) -> List[tensor[*, f32]] = [b]\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "an explicit `List[tensor[*, f32]]` annotation makes no uniformity \
         promise, so a wildcard element must type-check; got {errs:?}",
    );
}

#[test]
fn issue_272_nested_list_wildcard_under_rigid_dim_not_checked() {
    // Documented boundary: `check_list_elem_rigid_dim_vs_wildcard` matches
    // a single `List[tensor[..]]` level and does NOT recurse into the
    // element when it is itself a `List`. A wildcard tensor nested under
    // `List[List[tensor[k, f32]]]` is therefore NOT protected.
    //   BEFORE: ACCEPT.   AFTER: ACCEPT (the check does not reach the inner
    //   rigid dim).
    // If nested-list rigidity ever needs enforcing, this test is the
    // canary: it should flip to REJECT and be moved to the rejecting group.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested_list_wildcard.ch");
    write_file(
        &path,
        "def f[k](b: tensor[*, f32]) -> List[List[tensor[k, f32]]] = [[b]]\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "documented gap: nested-list element rigidity is not checked, so \
         this currently type-checks; got {errs:?}",
    );
}
