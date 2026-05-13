//! Wave 3 terminal red team for the 0.7.9 lint precision + Shape A broader
//! return cleanup workstream.
//!
//! Adversarial fixtures for the four §5 closures shipped by PRs #107, #108,
//! and #109:
//!
//! - `Lint-RedundantLinearityCopyOnBorrowWarn-F1` (PR #107) — opt-in to
//!   `check_mirrors_fix=true` to suppress unfixable warnings.
//! - `Lint-PreferPipeRedundantLinearityPair-F1` (PR #107) — subsumed.
//! - `Lint-ExceptionPathRoot-F1` (PR #108) — workspace-root-anchored
//!   exception matching.
//! - `Linearity-ShapeABroadReturn-F1` (PR #109) — `descend_to_tail_var`
//!   helper covering `let`/`if`/`match` tail-position bodies.
//!
//! Each fixture has a pinned expected outcome. The file is divided into
//! three sections (LP, LE, SR) matching the workstream prefixes.

use assert_cmd::Command;
use std::fs;
use std::path::Path;
use tempfile::tempdir;

fn write_file(path: &Path, contents: &str) {
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent).expect("create parent");
    }
    fs::write(path, contents).expect("write file");
}

fn run_chelis(cwd: &Path, args: &[&str]) -> (i32, String, String) {
    let mut cmd = Command::cargo_bin("chelis").expect("chelis binary");
    cmd.current_dir(cwd);
    for a in args {
        cmd.arg(a);
    }
    let output = cmd.output().expect("run chelis");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&output.stderr).into_owned();
    let code = output.status.code().unwrap_or(-1);
    (code, stdout, stderr)
}

fn fmt_inplace(path: &Path) {
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["fmt", "--inplace", path.to_str().unwrap()])
        .assert()
        .success();
}

// ============================================================
// §LP redundant-linearity-call precision (PR #107)
// ============================================================

/// LP-1: Multi-level nested `copy(copy(y))` on a borrow argument.
/// After the first lint-fix iteration the outer `copy()` can be stripped
/// (the inner `copy(y)` yields an owned tensor, so the outer is
/// redundant). After convergence the inner `copy(y)` remains and its
/// warning is suppressed because the strip would fail type-checking.
/// Pins that the suppression mechanism converges correctly across
/// multiple fix iterations.
#[test]
fn lp_nested_copy_on_borrow_converges_to_zero_redundant_warnings() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("nested_copy.ch");
    write_file(
        &path,
        "def consume_owned[n](x: tensor[n, f32]) -> tensor[n, f32] = realize(x)\n\
         def caller[n](y: &tensor[n, f32]) -> tensor[n, f32] = consume_owned(copy(copy(y)))\n\
         input = to_tensor([1.0, 2.0])\n\
         result = caller(&input)\n",
    );
    fmt_inplace(&path);

    // First lint-fix pass strips the outer copy()...
    Command::cargo_bin("chelis")
        .expect("chelis binary")
        .args(["lint", "--fix", path.to_str().unwrap()])
        .assert()
        .success();
    // ...and a second pass must converge with no further redundant-linearity-call
    // warnings (the inner copy(y) suppression triggers because the
    // strip would fail type-checking).
    let (code, out, _err) = run_chelis(dir.path(), &["lint", "--check", path.to_str().unwrap()]);
    assert_eq!(code, 0, "second pass must exit 0; out={out}");
    assert!(
        !out.contains("redundant-linearity-call"),
        "second pass must not flag redundant-linearity-call (suppressed by typed-pipeline gate); out={out}"
    );
}

/// LP-2: `copy(borrow)` in a let-binding RHS. The CLI driver probes
/// each strip candidate independently and rejects the one whose
/// post-strip program fails type-check. Pins that the rule's
/// suppression covers this shape.
#[test]
fn lp_copy_borrow_in_let_rhs_warning_suppressed() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("let_rhs.ch");
    write_file(
        &path,
        "def consume_owned[n](x: tensor[n, f32]) -> tensor[n, f32] = realize(x)\n\
         def caller[n](y: &tensor[n, f32]) -> tensor[n, f32] = {\n  \
           z = copy(y)\n  \
           consume_owned(z)\n\
         }\n\
         input = to_tensor([1.0, 2.0])\n\
         result = caller(&input)\n",
    );
    fmt_inplace(&path);

    let (code, out, _err) = run_chelis(dir.path(), &["lint", "--check", path.to_str().unwrap()]);
    assert_eq!(code, 0, "lint exit must be 0; out={out}");
    assert!(
        !out.contains("redundant-linearity-call"),
        "let-RHS copy(borrow) must be suppressed (typed-pipeline rejects the strip); out={out}",
    );
}

/// LP-3: Positive control: genuine-redundant copy() on an OWNED tensor.
/// The strip is safe under implicit linearity, so the warning must fire
/// with a `[fix]` marker. Pins that the opt-in does not over-suppress.
#[test]
fn lp_genuine_redundant_copy_still_flagged_with_fix_marker() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("genuine.ch");
    write_file(
        &path,
        "def f[n](x: tensor[n, f32]) -> tensor[n, f32] = realize(copy(x))\n\
         result = f(to_tensor([1.0, 2.0]))\n",
    );
    fmt_inplace(&path);

    let (code, out, _err) = run_chelis(dir.path(), &["lint", "--check", path.to_str().unwrap()]);
    assert_eq!(code, 0, "lint exit must be 0; out={out}");
    assert!(
        out.contains("redundant-linearity-call"),
        "genuine redundant copy() must still fire; out={out}",
    );
    assert!(
        out.contains("[fix]"),
        "[fix] marker required for safe strip; out={out}",
    );
}

/// LP-LEAK-A: REGRESSION SURFACE — `chelis check` ADVISORY EMIT PATH
/// DOES NOT SUPPRESS UNFIXABLE WARNINGS.
///
/// PR #107 opted `redundant-linearity-call` into `check_mirrors_fix=true`,
/// which makes `cmd_lint` call `should_suppress_unfixable_violation` and
/// suppress warnings whose autofix would be rejected by the typed-pipeline
/// gate. But `emit_advisory_lint_warnings_for_file` (the function
/// `chelis check` invokes via `cmd_check_one` at line 1198 of
/// `crates/chelis-cli/src/main.rs`) ONLY applies the path-pattern exception
/// filter, NOT the `should_suppress_unfixable_violation` filter. The
/// 0.7.9 closure therefore only suppresses the false positives in the
/// `chelis lint --check` workflow.
///
/// Customer impact: Nautilus's reported 319 false-positive warnings in
/// `src/linalg.ch` against `chelis check` remain unaddressed for that
/// workflow. The `[Unreleased]` CHANGELOG entry claims closure of
/// `Lint-RedundantLinearityCopyOnBorrowWarn-F1` but the customer's
/// canonical `chelis check` workflow still floods.
///
/// This test pins the leak.
#[test]
#[ignore = "documents the chelis-check advisory-emit leak; flip when emit_advisory_lint_warnings_for_file applies should_suppress_unfixable_violation"]
fn lp_leak_a_chelis_check_advisory_emit_does_not_suppress_unfixable_copy_borrow() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("copy_borrow.ch");
    write_file(
        &path,
        "def consume_owned[n](x: tensor[n, f32]) -> tensor[n, f32] = realize(x)\n\
         def caller[n](y: &tensor[n, f32]) -> tensor[n, f32] = consume_owned(copy(y))\n\
         input = to_tensor([1.0, 2.0])\n\
         result = caller(&input)\n",
    );
    fmt_inplace(&path);

    let (_lint_code, lint_out, _lint_err) =
        run_chelis(dir.path(), &["lint", "--check", path.to_str().unwrap()]);
    assert!(
        !lint_out.contains("redundant-linearity-call"),
        "baseline: chelis lint --check correctly suppresses; out={lint_out}",
    );

    let (_check_code, _check_out, check_err) =
        run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    // POST-FIX EXPECTATION (currently failing because the leak exists):
    // chelis check must NOT emit the warning either.
    assert!(
        !check_err.contains("redundant-linearity-call"),
        "leak: chelis check must apply the same suppression as chelis lint --check; stderr={check_err}",
    );
}

/// LP-LEAK-B: the SAME leak affects `prefer-pipe-operator`. Both rules
/// opt in to `check_mirrors_fix`; both are advisory; both leak through
/// `emit_advisory_lint_warnings_for_file`. Pins that the fix to
/// LP-LEAK-A must apply uniformly across rules with that opt-in.
#[test]
#[ignore = "documents the same advisory-emit leak for prefer-pipe-operator; flip when emit_advisory_lint_warnings_for_file applies should_suppress_unfixable_violation"]
fn lp_leak_b_chelis_check_advisory_emit_does_not_suppress_unfixable_prefer_pipe() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("pipe_drop.ch");
    write_file(
        &path,
        "def f(xs: List[int64], n: int64) -> List[int64] = drop(xs, n)\n\
         result = f(list_of([1, 2, 3, 4]), cast(2, int64))\n",
    );
    fmt_inplace(&path);

    let (_lint_code, lint_out, _lint_err) =
        run_chelis(dir.path(), &["lint", "--check", path.to_str().unwrap()]);
    let lint_pipe = lint_out.matches("prefer-pipe-operator").count();

    let (_check_code, _check_out, check_err) =
        run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    let check_pipe = check_err.matches("prefer-pipe-operator").count();

    assert_eq!(
        check_pipe, lint_pipe,
        "leak: chelis check emits {check_pipe} prefer-pipe-operator warning(s); chelis lint --check emits {lint_pipe}; counts must agree once the advisory-emit path applies should_suppress_unfixable_violation"
    );
}

// ============================================================
// §LE workspace-root exception path matching (PR #108)
// ============================================================

/// LE-1: Deep single-segment subtree walk. `chelis lint --check
/// crates/chelis-surf` from the workspace root must apply the
/// workspace-rooted exception identically to `chelis lint --check .`.
#[test]
fn le_deep_subtree_walk_applies_workspace_rooted_exception() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let fixture_dir = root.join("crates/chelis-surf/tests/fixtures");
    fs::create_dir_all(&fixture_dir).expect("mkdir");
    fs::write(
        fixture_dir.join("block_binding_expr.ch"),
        "def f(x: f32): f32 = {\n  y = mul(x, x)\n  add(y, x)\n}\n",
    )
    .expect("write fixture");

    let (code_deep, out_deep, _) = run_chelis(root, &["lint", "--check", "crates/chelis-surf"]);
    assert!(
        !out_deep.contains("surf-def-arrow-form"),
        "deep subtree walk must apply workspace-rooted exception; out={out_deep}"
    );
    assert_eq!(code_deep, 0, "exit 0 required; out={out_deep}");
}

/// LE-2: Mixed walk targets (file + directory). `chelis lint --check
/// crates/chelis-cli/tests/red_team_0_7_9.rs crates/chelis-types`
/// must continue to apply workspace-rooted exceptions on the fixture.
#[test]
fn le_mixed_file_and_dir_targets_apply_workspace_rooted_exception() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let fixture_dir = root.join("crates/chelis-surf/tests/fixtures");
    fs::create_dir_all(&fixture_dir).expect("mkdir");
    fs::write(
        fixture_dir.join("operators.ch"),
        "def add_mul(x: f32, y: f32): f32 = mul(add(x, y), x)\n",
    )
    .expect("write fixture");
    // Add an unrelated docs file so the mixed-target invocation has
    // something to walk besides the excepted fixture.
    fs::create_dir_all(root.join("docs")).expect("mkdir docs");
    fs::write(root.join("docs/note.md"), "# stub\n").expect("write doc");

    let (code, out, _) = run_chelis(root, &["lint", "--check", "crates", "docs"]);
    assert!(
        !out.contains("surf-def-arrow-form"),
        "mixed walk targets must apply workspace-rooted exception; out={out}"
    );
    assert_eq!(code, 0, "exit 0 required; out={out}");
}

/// LE-LEAK-A: REGRESSION SURFACE — `detect_lint_workspace_root` USES
/// CWD AS WORKSPACE ROOT WITH NO ACTUAL WORKSPACE DETECTION.
///
/// PR #108 anchored exception matching against a "workspace root" but
/// implemented `detect_lint_workspace_root` as
/// `canonicalize(current_working_directory)` — there is no `Cargo.toml`
/// or `.git` probe to verify the CWD actually IS the workspace root.
/// When a developer runs `chelis lint --check .` from `crates/` (a
/// subdirectory of the workspace), the "workspace root" is taken to be
/// `<repo>/crates`, and workspace-rooted exception patterns like
/// `crates/chelis-surf/tests/fixtures/*.ch` fail to match because the
/// relative path is `chelis-surf/tests/fixtures/*.ch` (no leading
/// `crates/` segment).
///
/// The §5 closure entry claims that `chelis lint --check .` and
/// `chelis lint --check crates docs examples packages` produce
/// "identical output" — that holds only when invoked from the workspace
/// root. The customer-visible scope is narrower than the entry implies.
///
/// This test pins the leak with a synthesized workspace tree that
/// mirrors the real repo's exception structure.
#[test]
#[ignore = "documents the cwd-as-workspace-root assumption; flip when detect_lint_workspace_root distinguishes workspace root from CWD"]
fn le_leak_a_cwd_not_workspace_root_breaks_workspace_rooted_exception() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let fixture_dir = root.join("crates/chelis-surf/tests/fixtures");
    fs::create_dir_all(&fixture_dir).expect("mkdir");
    fs::write(
        fixture_dir.join("block_binding_expr.ch"),
        "def f(x: f32): f32 = {\n  y = mul(x, x)\n  add(y, x)\n}\n",
    )
    .expect("write fixture");

    // Invoke from `<root>/crates`, not from `<root>`. The exception
    // pattern `crates/chelis-surf/tests/fixtures/*.ch` should still
    // apply because the violation IS in `crates/chelis-surf/...`.
    let crates_dir = root.join("crates");
    let (code, out, _err) = run_chelis(&crates_dir, &["lint", "--check", "."]);
    assert!(
        !out.contains("surf-def-arrow-form"),
        "leak: workspace-rooted exception must apply regardless of CWD; ran from {}; out={out}",
        crates_dir.display()
    );
    assert_eq!(code, 0, "exit 0 required; out={out}");
}

// ============================================================
// §SR Shape A broader return tail-position coercion (PR #109)
// ============================================================

/// SR-1: Chained let aliases. `let a = x; let b = a; b` resolves
/// every tail through name-aliased var refs. The descent walks `let`
/// recursively, so this must pass.
#[test]
fn sr_chained_let_aliases_lower_cleanly() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("chained.ch");
    write_file(
        &path,
        "module Repro.ChainedLet\n\
         def f[n](x: &tensor[n, f32]) -> tensor[n, f32] = {\n  \
           a = x\n  \
           b = a\n  \
           b\n\
         }\n",
    );
    fmt_inplace(&path);

    let (code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert_eq!(code, 0, "check exit 0 expected; out={out}");
    assert!(out.contains("\"score\": 1"), "score must be 1; out={out}");
}

/// SR-2: NEGATIVE CONTROL. If-branches return different variables.
/// The descent's "same name across siblings" rule rejects; the existing
/// TypeMismatch must surface. Pins that PR #109 does not over-relax.
#[test]
fn sr_if_branches_with_different_vars_still_fail() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("diff_vars.ch");
    write_file(
        &path,
        "module Repro.IfDifferentVars\n\
         def f[n](c: bool, x: &tensor[n, f32], y: &tensor[n, f32]) -> tensor[n, f32] = if c then x else y\n",
    );
    fmt_inplace(&path);

    let (_code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert!(
        out.contains("TypeMismatch"),
        "different-var if-branches must surface TypeMismatch; out={out}"
    );
}

/// SR-3: NEGATIVE CONTROL. Tail expression is a call, not a bare var.
/// The descent rejects `(app realize_it y)`; the existing TypeMismatch
/// must surface.
#[test]
fn sr_tail_call_not_var_still_fails() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("tail_call.ch");
    write_file(
        &path,
        "module Repro.TailIsCall\n\
         def realize_it[n](v: &tensor[n, f32]) -> &tensor[n, f32] = v\n\
         def f[n](x: &tensor[n, f32]) -> tensor[n, f32] = {\n  \
           y = x\n  \
           realize_it(y)\n\
         }\n",
    );
    fmt_inplace(&path);

    let (_code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert!(
        out.contains("TypeMismatch"),
        "tail-call body must surface TypeMismatch; out={out}"
    );
}

/// SR-4: Match scrutinee is a complex expression (a fn call), but
/// every arm body is a bare-var. The descent walks ARM BODIES only,
/// not the scrutinee, so this must pass.
#[test]
fn sr_match_complex_scrutinee_with_bare_var_arms_lowers_cleanly() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("complex_scrutinee.ch");
    write_file(
        &path,
        "module Repro.MatchComplex\n\
         type Choice = | Left | Right\n\
         def helper() -> Choice = Left\n\
         def f[n](x: &tensor[n, f32]) -> tensor[n, f32] = match helper() with {\n  \
           | Left => x\n  \
           | Right => x\n\
         }\n",
    );
    fmt_inplace(&path);

    let (code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert_eq!(code, 0, "check exit 0 expected; out={out}");
    assert!(out.contains("\"score\": 1"), "score must be 1; out={out}");
}

/// SR-5: NEGATIVE CONTROL. Match with one arm being a non-bare-var
/// (a fn call). The descent rejects; existing TypeMismatch surfaces.
#[test]
fn sr_match_one_arm_call_other_var_still_fails() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("one_arm_call.ch");
    write_file(
        &path,
        "module Repro.MatchOneArmCall\n\
         type Choice = | Left | Right\n\
         def realize_it[n](v: &tensor[n, f32]) -> &tensor[n, f32] = v\n\
         def f[n](c: Choice, x: &tensor[n, f32]) -> tensor[n, f32] = match c with {\n  \
           | Left => x\n  \
           | Right => realize_it(x)\n\
         }\n",
    );
    fmt_inplace(&path);

    let (_code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert!(
        out.contains("TypeMismatch"),
        "match arm with non-var body must surface TypeMismatch; out={out}"
    );
}

/// SR-6: Mixed let-of-if pattern. `let y = (if c then x else x) in y`
/// — the let RHS is an if, but the let TAIL is a bare-var `y`. The
/// descent walks let's body (third child), so the if-RHS doesn't matter.
#[test]
fn sr_let_with_if_rhs_lowers_cleanly() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("let_if_rhs.ch");
    write_file(
        &path,
        "module Repro.LetIfRhs\n\
         def f[n](c: bool, x: &tensor[n, f32]) -> tensor[n, f32] = {\n  \
           y = if c then x else x\n  \
           y\n\
         }\n",
    );
    fmt_inplace(&path);

    let (code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert_eq!(code, 0, "check exit 0 expected; out={out}");
    assert!(out.contains("\"score\": 1"), "score must be 1; out={out}");
}

/// SR-LEAK-A: REGRESSION SURFACE — `types_structurally_equal` (PR #91)
/// COMPARES TENSOR RANK ONLY, NOT DIM IDENTITY.
///
/// PR #109's relaxed-retry calls `types_structurally_equal` to guard
/// the relaxation, which only checks dim *count* (`d1.len() == d2.len()`),
/// not dim *identity*. As a result, a function declaring
/// `tensor[n, f32]` but returning `tensor[m, f32]` (different dim vars
/// of the same rank) passes the structural check, the relaxed unify
/// instantiates the body's free dim var `m` with `n`, and the program
/// type-checks and evaluates with a runtime shape mismatch.
///
/// This is a pre-existing soundness gap in PR #91 (Shape A bare-var)
/// that PR #109 widens to `let`/`if`/`match` tail-position bodies.
/// Caveat: the same dim-var-unification leak fires without Shape A
/// (e.g. a plain owned-tensor body returning the wrong-named dim
/// also passes), so the root cause is in HM inference, not just
/// `types_structurally_equal`. Filed here as a SR test because
/// PR #109 extends its reach to additional body shapes.
#[test]
#[ignore = "documents the dim-var-unification soundness gap inherited from PR #91 and widened by PR #109; flip when types_structurally_equal (or the inference unify step) discriminates distinct dim-var bindings"]
fn sr_leak_a_dim_var_mismatch_silently_passes() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("dim_mismatch.ch");
    write_file(
        &path,
        "module Repro.DimMismatch\n\
         def f[n, m](x: &tensor[n, f32], y: &tensor[m, f32]) -> tensor[n, f32] = {\n  \
           z = y\n  \
           z\n\
         }\n\
         x_in = to_tensor([1.0, 2.0])\n\
         y_in = to_tensor([3.0, 4.0, 5.0])\n\
         result = f(&x_in, &y_in)\n",
    );
    fmt_inplace(&path);

    let (_code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    // POST-FIX EXPECTATION: returning &tensor[m, f32] where tensor[n, f32]
    // is declared must surface TypeMismatch because m and n are
    // user-named distinct dim parameters.
    assert!(
        out.contains("TypeMismatch"),
        "dim-var mismatch must surface TypeMismatch; out={out}"
    );
}
