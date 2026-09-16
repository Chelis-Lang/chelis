// Regression for chelis#293: a `def` generic over a *general* type
// variable `P` (declared in the explicit `[..]` quantifier list, not a
// dim var and not a precision var) that threads `P` through a
// function-typed parameter type-checked in isolation but failed at
// every call site with `type mismatch: P vs tensor[..]`.
//
// Root cause was in the Surf desugarer
// (`crates/chelis-surf/src/desugar.rs`): the `TypeExpr::Named` case
// applied a pure lexical case-split — any uppercase name became a
// rigid ADT `(t-adt {} P)` — and ignored the def's explicit `[..]`
// quantifier set. With `P` lowered to a rigid ADT it had no free type
// variable to generalize, so `Env::generalize` left it un-quantified
// in the def's scheme; at the call site `Env::instantiate` produced no
// fresh variable for it and unification hit the `Adt` vs concrete-arg
// branch, emitting `type mismatch: P vs tensor[Lit(N), f32]`.
//
// Fix: in `desugar_type_with_scope`, a `TypeExpr::Named` whose name is
// in the enclosing quantifier set (`tvar_set` — the def's `[..]`
// clause) lowers to `(t-var {} <name>)` regardless of case. The `[..]`
// clause is the authoritative, unkinded quantifier source per
// `spec/02-surf-syntax.md` §P4b, so it overrides the case-split. The
// name then carries a real `TypeVar`, generalizes into the def's
// scheme, and instantiates to a fresh var at each call site that
// unifies through the arrow argument position.

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
fn general_tvar_through_callback_param_checks_clean() {
    // The reproducer from issue #293. Pre-fix: the `use_it` call site
    // failed with `TypeMismatch` / `type mismatch: P vs tensor[Lit(3),
    // f32]`. Post-fix: `chelis check` succeeds with no TypeMismatch.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("generic_callback.ch");
    write_file(
        &fixture,
        "module Repro.GenericCallback\n\
         def apply_resid[n, P](x: tensor[n, f32], inner_p: P, f: tensor[n, f32] -> P -> tensor[n, f32]) -> tensor[n, f32] =\n  \
           add(x, f(x, inner_p))\n\
         def use_it(x: tensor[3, f32], w: tensor[3, f32]) -> tensor[3, f32] =\n  \
           apply_resid(x, w, fn (t, q) -> mul(t, q))\n",
    );
    fmt_inplace(&fixture);

    check_file(&fixture)
        .success()
        .stdout(predicate::str::contains("TypeMismatch").not())
        .stdout(predicate::str::contains("type mismatch: P").not());
}

#[test]
fn general_tvar_def_checks_clean_in_isolation() {
    // Control: the generic def alone already checked clean before the
    // fix and must continue to.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("apply_resid_only.ch");
    write_file(
        &fixture,
        "module Repro.GenericCallback\n\
         def apply_resid[n, P](x: tensor[n, f32], inner_p: P, f: tensor[n, f32] -> P -> tensor[n, f32]) -> tensor[n, f32] =\n  \
           add(x, f(x, inner_p))\n",
    );
    fmt_inplace(&fixture);

    check_file(&fixture)
        .success()
        .stdout(predicate::str::contains("TypeMismatch").not());
}

#[test]
fn incompatible_callback_argument_still_rejected() {
    // Negative parity: the fix must not over-loosen unification. The
    // callback body `mul(t, q)` forces `q` (which is `P`) to be the
    // same tensor type as `t` (`tensor[n, f32]`), but the `inner_p`
    // argument supplied at the call site is a scalar `i32` literal,
    // bound to the same `P`. `P` cannot be both, so a unification
    // mismatch must still be reported.
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("bad_callback.ch");
    write_file(
        &fixture,
        "module Repro.BadCallback\n\
         def apply_resid[n, P](x: tensor[n, f32], inner_p: P, f: tensor[n, f32] -> P -> tensor[n, f32]) -> tensor[n, f32] =\n  \
           add(x, f(x, inner_p))\n\
         def use_it(x: tensor[3, f32]) -> tensor[3, f32] =\n  \
           apply_resid(x, 1, fn (t, q) -> mul(t, q))\n",
    );
    fmt_inplace(&fixture);

    // A unification failure surfaces as one of TypeMismatch /
    // PrecisionMismatch / DimensionMismatch in the diagnostic JSON.
    check_file(&fixture).stdout(
        predicate::str::contains("TypeMismatch")
            .or(predicate::str::contains("PrecisionMismatch"))
            .or(predicate::str::contains("DimensionMismatch")),
    );
}
