//! WS-A7 acceptance tests: bare-arg `def` + sig-with-borrows return-type
//! inference.
//!
//! Pin the bug surfaced during the WS-C v2 escalation: a `def` whose
//! body's parameters carry no annotations and whose declared signature
//! borrows the inputs (`&tensor[..]`) but returns an owned tensor was
//! incorrectly reporting that the body's return position was a borrow.
//!
//! Reproducer:
//!
//! ```chelis
//! sig tadd: &tensor[n, p] -> &tensor[n, p] -> tensor[n, p]
//! def tadd(lhs, rhs) = add(lhs, rhs)
//! ```
//!
//! Root cause (see `crates/chelis-types/src/infer.rs::infer_def_body_with_sig`):
//! the bare-arg `(fn (params lhs rhs) (app add lhs rhs))` body was inferred
//! with fresh, unconstrained type variables for each param. The call to
//! `add` (whose scheme is `(&t, &t) -> t`) auto-borrowed the unbound param
//! tvars at the call site, which collapsed the param-side and return-side
//! of `add` into the same equivalence class. The post-body sig-unify then
//! drove the return position to `&tensor[..]` instead of the declared
//! `tensor[..]`. Seeding bare body params with the declared sig types
//! before body inference makes the bare-arg path behave the same as the
//! annotated-arg path.
//!
//! Triangulated against two passing baselines:
//! - `matmul` (whose scheme `(&t1, &t2) -> out` keeps the return tvar
//!   independent of the param tvars; the collapse never happens) still
//!   type-checks unchanged.
//! - The annotated-arg variant of the reproducer (`def tadd(lhs:
//!   &tensor[n, p], rhs: &tensor[n, p]) = add(lhs, rhs)`) still type-checks
//!   unchanged.

use assert_cmd::Command;
use serde_json::Value;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    fs::write(path, contents).expect("write file");
}

fn run_json_check(path: &Path) -> Value {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check", path.to_str().unwrap()])
        .assert()
        .success()
        .get_output()
        .stdout
        .clone();
    serde_json::from_slice(&output).expect("check output should be json")
}

fn assert_clean(json: &Value, label: &str) {
    let errors = json["errors"]
        .as_array()
        .unwrap_or_else(|| panic!("{label}: errors should be a json array, got {json}"));
    assert!(
        errors.is_empty(),
        "{label}: expected no check errors, got {errors:?}",
    );
    let score = json["score"].as_f64().unwrap_or(0.0);
    assert!(
        (score - 1.0).abs() < 1e-9,
        "{label}: expected score 1.0, got {score} ({json})"
    );
}

/// The WS-A7 reproducer with sig-quantified precision (`p`): the bug
/// pre-dated WS-A5, so it triggers under both polymorphic-precision and
/// concrete-precision sigs.
#[test]
fn bare_arg_add_with_borrow_sig_polymorphic_precision_type_checks() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wsa7_repro_poly.ch");
    write_file(
        &path,
        "sig tadd: &tensor[n, p] -> &tensor[n, p] -> tensor[n, p]\n\
         def tadd(lhs, rhs) = add(lhs, rhs)\n",
    );
    let json = run_json_check(&path);
    assert_clean(&json, "polymorphic-precision sig + bare-arg def + add body");
}

/// Same shape as the polymorphic case but with concrete `f32` precision.
/// Confirms the fix is not specific to WS-A5 precision polymorphism.
#[test]
fn bare_arg_add_with_borrow_sig_concrete_precision_type_checks() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wsa7_repro_concrete.ch");
    write_file(
        &path,
        "sig tadd: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]\n\
         def tadd(lhs, rhs) = add(lhs, rhs)\n",
    );
    let json = run_json_check(&path);
    assert_clean(&json, "concrete-precision sig + bare-arg def + add body");
}

/// Arithmetic-dtype matrix: the WS-A7 reproducer shape (sig with
/// borrowed inputs + owned output, bare-arg def delegating to `add`)
/// must type-check when instantiated at every active arithmetic dtype.
/// This loop was consolidated here from `stdlib_precision_generalization_followups.rs`
/// (formerly `wsa7_bare_def_with_sig_having_borrows_typechecks`) in the
/// e2e parsimony pass so the dtype-matrix coverage lives with the
/// invariant's owning file.
#[test]
fn bare_arg_add_with_borrow_sig_typechecks_at_every_arithmetic_dtype() {
    const ARITHMETIC_DTYPES: &[&str] = &[
        "f32", "f64", "bf16", "f16", "int8", "int16", "int32", "int64",
    ];
    for dtype in ARITHMETIC_DTYPES {
        let dir = tempdir().expect("tempdir");
        let path = dir.path().join("wsa7_dtype_matrix.ch");
        let src = format!(
            "sig add_bare: &tensor[n, p] -> &tensor[n, p] -> tensor[n, p]\n\
             def add_bare(lhs, rhs) = add(lhs, rhs)\n\
             def use_at_dtype(xs: &tensor[3, {dtype}]) -> tensor[3, {dtype}] = add_bare(xs, xs)\n"
        );
        write_file(&path, &src);
        let json = run_json_check(&path);
        assert_clean(&json, &format!("wsa7 sig+bare-def at {dtype}"));
    }
}

/// Baseline 1 (must remain green): `matmul` shape, where the callee
/// scheme uses an independent return tvar (`(&t1, &t2) -> out`). The
/// collapse that broke the `add` shape never fires here, so this case
/// type-checked even pre-fix; it is included as a regression guard.
#[test]
fn baseline_bare_arg_matmul_with_borrow_sig_still_type_checks() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wsa7_baseline_matmul.ch");
    write_file(
        &path,
        "sig tmatmul: &tensor[m, k, p] -> &tensor[k, n, p] -> tensor[m, n, p]\n\
         def tmatmul(lhs, rhs) = matmul(lhs, rhs)\n",
    );
    let json = run_json_check(&path);
    assert_clean(
        &json,
        "baseline: bare-arg def + matmul body (independent return tvar)",
    );
}

/// Baseline 2 (must remain green): the annotated-arg variant of the
/// reproducer. With explicit `&tensor[..]` annotations on `lhs`/`rhs`,
/// the call-site auto-borrow had nothing to do (the actuals were
/// already concrete `&tensor`), so the collapse never fired; this case
/// type-checked even pre-fix and is included as a regression guard.
#[test]
fn baseline_annotated_args_with_borrow_sig_still_type_checks() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wsa7_baseline_annotated.ch");
    write_file(
        &path,
        "sig tadd: &tensor[n, f32] -> &tensor[n, f32] -> tensor[n, f32]\n\
         def tadd(lhs: &tensor[n, f32], rhs: &tensor[n, f32]) = add(lhs, rhs)\n",
    );
    let json = run_json_check(&path);
    assert_clean(
        &json,
        "baseline: annotated-arg def + add body (no auto-borrow at call site)",
    );
}

/// Negative coverage: a bare-arg def whose body returns an owned tensor
/// when the sig demands a borrowed return must error clearly. This
/// exercises the post-body sig-unify path on the structurally-distinct
/// case (body's return position is `Tensor`, declared is `Ref(Tensor)`).
#[test]
fn bare_arg_owned_body_with_borrow_return_sig_errors() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wsa7_neg_owned_to_borrow.ch");
    write_file(
        &path,
        "sig owned_to_borrow: &tensor[n, f32] -> &tensor[n, f32]\n\
         def owned_to_borrow(x) = add(x, x)\n",
    );
    let json = run_json_check(&path);
    let errors = json["errors"]
        .as_array()
        .expect("errors should be a json array");
    assert!(
        !errors.is_empty(),
        "owned-body vs borrowed-return-sig must produce a check error, got clean output {json}"
    );
    let any_mentions_mismatch = errors.iter().any(|e| {
        let msg = e.get("message").and_then(|m| m.as_str()).unwrap_or("");
        msg.contains("doesn't match declared signature")
    });
    assert!(
        any_mentions_mismatch,
        "expected a body-vs-declared-signature diagnostic, got {errors:?}",
    );
}

/// Negative coverage: a bare-arg def whose body returns a borrow when
/// the sig demands an owned return must error clearly. This is the
/// other direction of the borrow-vs-owned mismatch and confirms the fix
/// did not flip the polarity (i.e., it doesn't silently accept bodies
/// that genuinely produce a borrowed return when the sig says owned).
#[test]
fn bare_arg_borrow_body_with_owned_return_sig_errors() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("wsa7_neg_borrow_to_owned.ch");
    write_file(
        &path,
        "sig borrow_to_owned: tensor[n, f32] -> tensor[n, f32]\n\
         def borrow_to_owned(x) = &x\n",
    );
    let json = run_json_check(&path);
    let errors = json["errors"]
        .as_array()
        .expect("errors should be a json array");
    assert!(
        !errors.is_empty(),
        "borrow-body vs owned-return-sig must produce a check error, got clean output {json}"
    );
}
