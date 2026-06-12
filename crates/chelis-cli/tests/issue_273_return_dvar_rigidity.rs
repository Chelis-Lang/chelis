//! Issue Chelis-Lang/chelis#273: `check_declared_dvars_rigid` ignored
//! return-position dim vars, so a declared dim parameter appearing
//! *only in the return type* could be silently pinned by the body.
//!
//! The §4.4 rigidity guard collected declared dim parameters from the
//! signature's *parameter* positions only. A dim var occurring only in
//! the return type was therefore never handed to the guard, and
//! `def f[k](a: tensor[2, f32]) -> tensor[k, f32] = a` type-checked
//! while pinning `k := 2` — a promise of "for all k" the body cannot
//! keep.
//!
//! The fix distinguishes two roles for a return-only dim parameter
//! (spec/04-type-system.md §4.4.1):
//!
//!   * **output-inferred** — the body *produces* the dimension, either
//!     leaving the dim var unbound (generalize) or resolving it to a
//!     body-internal concrete dim (`examples/hello_tensor.ch`'s
//!     `def main() -> tensor[n, f32]` whose body builds a
//!     `tensor[3, f32]`). Accepted; the registered scheme resolves to
//!     the produced dim.
//!   * **input-coupled** — the body derives the return dim from the
//!     caller-visible parameter world: it pins the dim var to a
//!     concrete literal that occurs in a declared parameter position,
//!     or collapses it with a param-position declared dim parameter.
//!     Rejected with `DimensionMismatch`, the same diagnostic class as
//!     the param-position §4.4 violations.
//!
//! Sibling soundness work: chelis#272 closed the list-literal Cons-join
//! erasure of the same guard (`issue_272_list_dim_rigidity.rs`); this
//! issue predates it and is independent of lists.

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
// NEGATIVE: the #273 headline. A return-only rigid dim pinned to a
// concrete literal flowing from a declared parameter dimension.
// =================================================================

#[test]
fn issue_273_return_only_dvar_pinned_by_param_dim_rejects() {
    // The signature promises "for all k, returns tensor[k]"; the body
    // returns the tensor[2] parameter, pinning k := 2.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("headline.ch");
    write_file(&path, "def f[k](a: tensor[2, f32]) -> tensor[k, f32] = a\n");
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    let msgs = error_messages(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "return-only dim parameter pinned to a declared parameter's \
         concrete dim must reject with DimensionMismatch; got {msgs:?}",
    );
    assert!(
        msgs.iter()
            .any(|m| m.contains("return") && m.contains("Lit(2)")),
        "the diagnostic must name the return-position pin to Lit(2); \
         got {msgs:?}",
    );
}

#[test]
fn issue_273_return_only_dvar_pinned_by_other_param_dim_rejects() {
    // Same coupling through a different parameter: the body returns
    // `y: tensor[5, f32]`, pinning k := 5 (a declared param dim).
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("other_param.ch");
    write_file(
        &path,
        "def f[k](x: tensor[3, f32], y: tensor[5, f32]) -> tensor[k, f32] = y\n",
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "return-only dim parameter pinned to another parameter's \
         concrete dim must reject with DimensionMismatch; got {:?}",
        error_messages(&json),
    );
}

// =================================================================
// NEGATIVE: a return-only declared dim collapsed with a distinct
// param-position declared dim.
// =================================================================

#[test]
fn issue_273_return_only_dvar_collapsed_with_param_dvar_rejects() {
    // The body returns `x: tensor[n]` against a declared return
    // `tensor[m]`, unifying the distinct rigid dims n and m.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("collapse.ch");
    write_file(
        &path,
        "def f[n, m](x: tensor[n, f32]) -> tensor[m, f32] = x\n",
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    let msgs = error_messages(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "return-only dim parameter collapsed with a param-position dim \
         parameter must reject with DimensionMismatch; got {msgs:?}",
    );
    assert!(
        msgs.iter().any(|m| m.contains("return")),
        "the diagnostic must identify the return-position dim parameter; \
         got {msgs:?}",
    );
}

// =================================================================
// POSITIVE anchor: the param-position §4.4 baseline stays rejected
// (pre-existing behavior, must not regress).
// =================================================================

#[test]
fn issue_273_param_position_baseline_still_rejects() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("baseline.ch");
    write_file(
        &path,
        "def g[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = y\n",
    );
    let json = run_check(&path);
    let kinds = error_kinds(&json);
    assert!(
        kinds.iter().any(|k| k == "DimensionMismatch"),
        "the param-position §4.4 baseline must keep rejecting; got {:?}",
        error_messages(&json),
    );
}

// =================================================================
// POSITIVE: legitimate output-inferred return dims stay green.
// =================================================================

#[test]
fn issue_273_hello_tensor_style_output_inferred_dim_type_checks() {
    // Minimal in-test equivalent of examples/hello_tensor.ch: the body
    // genuinely produces the output dimension from body-internal data
    // (to_tensor static shape => tensor[3, f32]); no parameter dim is
    // involved. The return-only `n` resolves to the produced dim.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("output_inferred.ch");
    write_file(
        &path,
        "def make() -> tensor[n, f32] = to_tensor([1.0, 2.0, 3.0])\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "an output-inferred return dim (body-internal concrete shape, \
         no parameter coupling) must type-check; got {errs:?}",
    );
}

#[test]
fn issue_273_examples_hello_tensor_still_type_checks() {
    // The shipped example named by the issue as the regression risk for
    // the naive fix. Its `def main() -> tensor[n, f32]` body produces a
    // concrete tensor[3, f32]; the return-only `n` must stay accepted.
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../examples/hello_tensor.ch");
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "examples/hello_tensor.ch must keep type-checking; got {errs:?}",
    );
}

#[test]
fn issue_273_unbound_return_dvar_generalizes_type_checks() {
    // Variable-fed to_tensor produces wildcard dims; the wildcard
    // unifies permissively without binding, so the return-only `n`
    // stays a clean, generalizable dim var.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("generalize.ch");
    write_file(
        &path,
        "def f(items: List[f32]) -> tensor[n, f32] = to_tensor(items)\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "a return-only dim var left unbound by the body must generalize \
         and type-check; got {errs:?}",
    );
}

#[test]
fn issue_273_shared_param_return_dvar_identity_type_checks() {
    // A dim parameter shared between a param position and the return
    // is NOT return-only; the classic polymorphic identity stays green.
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("identity.ch");
    write_file(
        &path,
        "def id[n](x: tensor[n, f32]) -> tensor[n, f32] = x\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "the polymorphic identity (shared param/return dim) must keep \
         type-checking; got {errs:?}",
    );
}

// =================================================================
// POSITIVE boundary lock: a body-internal pin whose literal does NOT
// occur in any declared parameter position is tolerated (the
// deliberate output-inferred carve-out that keeps hello_tensor-style
// defs green). If this verdict ever flips, it must be a conscious
// spec change to §4.4, not silent drift.
// =================================================================

#[test]
fn issue_273_body_internal_pin_not_coinciding_with_params_type_checks() {
    // k := 3 from body-internal data; the only declared param dim is 2.
    // No caller-visible coupling => accepted (output-inferred).
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("body_internal.ch");
    write_file(
        &path,
        "def f(x: tensor[2, f32]) -> tensor[k, f32] = to_tensor([1.0, 2.0, 3.0])\n",
    );
    let json = run_check(&path);
    let errs = error_messages(&json);
    assert!(
        errs.is_empty(),
        "a body-internal pin (literal not occurring in any declared \
         parameter position) is the documented output-inferred \
         tolerance and must type-check; got {errs:?}",
    );
}
