// Regression for chelis#143: `sig` dim variables were not unified
// across parameter positions at type-check.
//
// A function with `sig f[n]: &tensor[n, f32] -> &tensor[n, f32] -> ...`
// should reject any call that passes tensors with different concrete
// values for `n`. At 0.7.10 the check silently passed because the
// `Wildcard ↔ Var` case in `unify_dim` bound the dim variable to
// `Wildcard`, after which any further `Var` occurrence resolved to
// `Wildcard` and silently unified with whatever the next arg provided.
//
// Fix in `crates/chelis-types/src/unify.rs`: when a `Wildcard` meets a
// `Var(v)`, succeed without binding `v`. Binding `v` to `Wildcard` is
// semantically equivalent to leaving `v` free (Wildcard matches
// anything), but the bind made the variable "stuck" so a later
// concrete arg couldn't constrain it. Leaving `v` free lets a later
// arg in any position bind it to a concrete `Lit`, and a subsequent
// concrete arg with a different value then trips the `Lit ↔ Lit`
// dimension mismatch as the sig demands.

use assert_cmd::Command;
use predicates::prelude::*;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn check_file(path: &Path) -> assert_cmd::assert::Assert {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["check", path.to_str().unwrap()])
        .assert()
}

fn fmt_inplace(path: &Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
}

#[test]
fn sig_dim_unification_mismatched_concrete_args_is_caught() {
    // Call site passes concrete-typed args with mismatched dims for
    // the same sig-quantified `n`. Pre-fix: `chelis check` returned
    // score=1.0, zero errors. Post-fix: DimensionMismatch.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("dim_mismatch.ch");
    write_file(
        &fixture,
        "module DimUnify\n\
         sig pair_id[n]: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]\n\
         def pair_id(x, y) = x\n\
         def call_it(a: &tensor[2, f32], b: &tensor[3, f32]) -> tensor[2, f32] = pair_id(a, b)\n",
    );
    fmt_inplace(&fixture);

    // `chelis check` writes its diagnostic JSON to stdout; the
    // presence of a `DimensionMismatch` entry in the `errors` array
    // is the failure signal. Issue #207 ties the exit code to the
    // errors array (exit 2 here), but this test asserts on the JSON
    // content; the dedicated invariant test covers the exit code.
    check_file(&fixture)
        .stdout(predicate::str::contains("DimensionMismatch"))
        .stdout(predicate::str::contains("Lit(2)"))
        .stdout(predicate::str::contains("Lit(3)"));
}

#[test]
fn sig_dim_unification_matched_concrete_args_passes_dim_check() {
    // Same shape as above but matched dims — the dim-check portion
    // must continue to pass (there is an unrelated Ref-vs-owned-return
    // mismatch that the inferred body's `x` triggers; this test only
    // confirms no DimensionMismatch is reported for the call-site).
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("dim_matched.ch");
    write_file(
        &fixture,
        "module DimUnify\n\
         sig pair_id[n]: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]\n\
         def pair_id(x, y) = x\n\
         def call_it(a: &tensor[4, f32], b: &tensor[4, f32]) -> tensor[4, f32] = pair_id(a, b)\n",
    );
    fmt_inplace(&fixture);

    check_file(&fixture).stdout(predicate::str::contains("DimensionMismatch").not());
}

#[test]
fn sig_dim_unification_wildcard_arg_does_not_mask_downstream_mismatch() {
    // When one arg has `Wildcard` (e.g. from `to_tensor`'s untyped
    // result) and the other has a concrete `Lit`, the `Wildcard` must
    // not bind the sig's dim var. The concrete arg's `Lit` should bind
    // the var so the call's return type carries the concrete dim
    // through, and a downstream sig that demands a *different*
    // concrete dim then trips DimensionMismatch.
    //
    // Pre-fix: the wildcard locked the sig var, the return type became
    // tensor[Wildcard, _], and the downstream sig's `Lit(3)` silently
    // unified with Wildcard — the cross-arg contract was dropped end
    // to end, not just at the immediate call site.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("wildcard_then_concrete.ch");
    write_file(
        &fixture,
        "module DimUnify\n\
         sig pair_id[n]: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]\n\
         def pair_id(x, y) = x\n\
         sig expect_three: &tensor[3, f32] -> f32\n\
         def expect_three(t) = cast(0.0, f32)\n\
         def call_it(a: &tensor[2, f32]) -> f32 = {\n\
           wild = to_tensor([cast(1.0, f32), cast(2.0, f32)])\n\
           result = pair_id(&wild, a)\n\
           expect_three(&result)\n}\n",
    );
    fmt_inplace(&fixture);

    // Post-fix: `a` binds the sig var to `Lit(2)`, `result` is
    // `tensor[2, f32]`, and the `expect_three(&result)` call unifies
    // `Lit(2)` against `Lit(3)` — DimensionMismatch with both literals
    // in the diagnostic. Pre-fix: zero DimensionMismatch entries here,
    // because every concrete dim went through a Wildcard sentinel that
    // matched-anything.
    check_file(&fixture)
        .stdout(predicate::str::contains("DimensionMismatch"))
        .stdout(predicate::str::contains("Lit(2)"))
        .stdout(predicate::str::contains("Lit(3)"));
}
