//! Wave 3 terminal red team for the 0.7.9 lint precision + Shape A broader
//! return cleanup workstream.
//!
//! Adversarial fixtures for two §5 closures shipped by PRs #108 and #109:
//!
//! - `Lint-ExceptionPathRoot-F1` (PR #108) — workspace-root-anchored
//!   exception matching.
//! - `Linearity-ShapeABroadReturn-F1` (PR #109) — `descend_to_tail_var`
//!   helper covering `let`/`if`/`match` tail-position bodies.
//!
//! Each fixture has a pinned expected outcome. The file is divided into
//! two sections (LE, SR) matching the workstream prefixes.

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

/// Write a minimal `[workspace]` `Cargo.toml` marker at `root`.
///
/// `detect_lint_workspace_root` probes for the Cargo workspace root via
/// `cargo locate-project --workspace` (LE-LEAK-A fix). A synthesized
/// test tree that exercises workspace-root detection must therefore
/// carry a real workspace marker — without it the probe finds nothing
/// (the tempdir lives under the system temp dir, not inside any Cargo
/// workspace) and the CLI fails clean by design. A test that omitted
/// the marker would be exercising the old cwd-assumption shortcut, not
/// the real detection mechanism.
fn write_workspace_marker(root: &Path) {
    fs::create_dir_all(root).expect("create workspace root");
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = []\nresolver = \"2\"\n",
    )
    .expect("write workspace marker");
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
// §LE workspace-root exception path matching (PR #108)
// ============================================================

/// LE-1: Deep single-segment subtree walk. `chelis lint --check
/// crates/chelis-surf` from the workspace root must apply the
/// workspace-rooted exception identically to `chelis lint --check .`.
#[test]
fn le_deep_subtree_walk_applies_workspace_rooted_exception() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_workspace_marker(root);
    let fixture_dir = root.join("crates/chelisup/bootstrap");
    fs::create_dir_all(&fixture_dir).expect("mkdir");
    fs::write(
        fixture_dir.join("chelisup.sh"),
        "#!/bin/sh\necho bootstrap\n",
    )
    .expect("write fixture");

    let (code_deep, out_deep, _) = run_chelis(root, &["lint", "--check", "crates/chelisup"]);
    assert!(
        !out_deep.contains("no-shell-scripts"),
        "deep subtree walk must apply workspace-rooted exception; out={out_deep}"
    );
    assert_eq!(code_deep, 0, "exit 0 required; out={out_deep}");
}

/// LE-2: Mixed walk targets (file + directory). `chelis lint --check
/// crates/chelis-cli/tests/lint_precision_broad_return_adversarial.rs crates/chelis-types`
/// must continue to apply workspace-rooted exceptions on the fixture.
#[test]
fn le_mixed_file_and_dir_targets_apply_workspace_rooted_exception() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_workspace_marker(root);
    let fixture_dir = root.join("crates/chelisup/bootstrap");
    fs::create_dir_all(&fixture_dir).expect("mkdir");
    fs::write(
        fixture_dir.join("chelisup.sh"),
        "#!/bin/sh\necho bootstrap\n",
    )
    .expect("write fixture");
    // Add an unrelated docs file so the mixed-target invocation has
    // something to walk besides the excepted fixture.
    fs::create_dir_all(root.join("docs")).expect("mkdir docs");
    fs::write(root.join("docs/note.md"), "# stub\n").expect("write doc");

    let (code, out, _) = run_chelis(root, &["lint", "--check", "crates", "docs"]);
    assert!(
        !out.contains("no-shell-scripts"),
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
/// `crates/chelisup/bootstrap/chelisup.sh` fail to match because the
/// relative path is `chelisup/bootstrap/chelisup.sh` (no leading
/// `crates/` segment).
///
/// The §5 closure entry claims that `chelis lint --check .` and
/// `chelis lint --check crates docs examples packages` produce
/// "identical output" — that holds only when invoked from the workspace
/// root. The customer-visible scope is narrower than the entry implies.
///
/// This test pins the leak with a synthesized workspace tree that
/// mirrors the real repo's exception structure. The tree carries a
/// `[workspace]` `Cargo.toml` marker so `detect_lint_workspace_root`'s
/// `cargo locate-project --workspace` probe can find the real
/// workspace root; without the fix the probe result is ignored and the
/// CWD is used directly, breaking the workspace-rooted exception.
#[test]
fn le_leak_a_cwd_not_workspace_root_breaks_workspace_rooted_exception() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_workspace_marker(root);
    let fixture_dir = root.join("crates/chelisup/bootstrap");
    fs::create_dir_all(&fixture_dir).expect("mkdir");
    fs::write(
        fixture_dir.join("chelisup.sh"),
        "#!/bin/sh\necho bootstrap\n",
    )
    .expect("write fixture");

    // Invoke from `<root>/crates`, not from `<root>`. The exception
    // pattern `crates/chelisup/bootstrap/chelisup.sh` should still
    // apply because the violation IS in `crates/chelis-surf/...`.
    let crates_dir = root.join("crates");
    let (code, out, _err) = run_chelis(&crates_dir, &["lint", "--check", "."]);
    assert!(
        !out.contains("no-shell-scripts"),
        "leak: workspace-rooted exception must apply regardless of CWD; ran from {}; out={out}",
        crates_dir.display()
    );
    assert_eq!(code, 0, "exit 0 required; out={out}");
}

/// LE-LEAK-FIX-1: workspace-root detection must produce identical
/// output for `chelis lint --check <abs-workspace-path>` issued from a
/// sibling directory OUTSIDE the workspace and for `chelis lint --check
/// .` issued from the workspace root. Pre-fix, `detect_lint_workspace_root`
/// returned `canonicalize(cwd)` — so a run from a sibling path anchored
/// exception matching against the sibling, not the workspace, and the
/// `crates/chelisup/bootstrap/chelisup.sh` exception failed to match.
#[test]
fn le_leak_fix_sibling_path_invocation_matches_workspace_root_invocation() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    let workspace = root.join("workspace");
    let sibling = root.join("sibling");
    fs::create_dir_all(&sibling).expect("mkdir sibling");
    write_workspace_marker(&workspace);
    let fixture_dir = workspace.join("crates/chelisup/bootstrap");
    fs::create_dir_all(&fixture_dir).expect("mkdir");
    fs::write(
        fixture_dir.join("chelisup.sh"),
        "#!/bin/sh\necho bootstrap\n",
    )
    .expect("write fixture");

    // Baseline: from the workspace root, `chelis lint --check .`.
    let (code_root, out_root, _) = run_chelis(&workspace, &["lint", "--check", "."]);
    assert!(
        !out_root.contains("no-shell-scripts"),
        "baseline: workspace-root invocation must apply the exception; out={out_root}"
    );
    assert_eq!(code_root, 0, "baseline exit 0 required; out={out_root}");

    // From a sibling dir outside the workspace, lint the workspace by
    // absolute path. Output must match the workspace-root invocation.
    let (code_sib, out_sib, _) = run_chelis(
        &sibling,
        &["lint", "--check", workspace.to_str().expect("utf8 path")],
    );
    assert!(
        !out_sib.contains("no-shell-scripts"),
        "sibling-path invocation must apply the workspace-rooted exception; out={out_sib}"
    );
    assert_eq!(code_sib, 0, "sibling-path exit 0 required; out={out_sib}");
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
/// SR-LEAK-A (TypeCheck-FreeDimVarUnification-F1), Path A: the
/// `let`-tail-var Shape A body. The body's tail resolves through a
/// `let` to a bare-var `z` whose inferred type is `&tensor[m, f32]`,
/// while the declared return is `tensor[n, f32]`. The relaxed-retry's
/// `types_structurally_equal` guard now compares dim identity, not
/// just rank, so the divergent dim params `n` and `m` are rejected.
#[test]
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

/// SR-LEAK-A Path A, bare-var body (PR #91's original Shape A shape).
/// `def f[n, m](...) -> tensor[n, f32] = y` where `y: &tensor[m, f32]`.
/// The relaxed-retry's structural guard must reject the divergent dim.
#[test]
fn sr_leak_a_path_a_bare_var_divergent_dim_fails() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bare_var_divergent.ch");
    write_file(
        &path,
        "module Repro.BareVarDivergent\n\
         def f[n, m](x: &tensor[n, f32], y: &tensor[m, f32]) -> tensor[n, f32] = y\n",
    );
    fmt_inplace(&path);

    let (_code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert!(
        out.contains("TypeMismatch"),
        "bare-var body returning the wrong dim param must surface TypeMismatch; out={out}"
    );
}

/// SR-LEAK-A Path A POSITIVE CONTROL, bare-var body, same dim var.
/// `def f[n](x, y: &tensor[n, f32]) -> tensor[n, f32] = y` is genuinely
/// valid: the body returns a borrow of a param whose dim matches the
/// declared return. The relaxed-retry must still recover this.
#[test]
fn sr_leak_a_path_a_bare_var_same_dim_passes() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("bare_var_same.ch");
    write_file(
        &path,
        "module Repro.BareVarSame\n\
         def f[n](x: &tensor[n, f32], y: &tensor[n, f32]) -> tensor[n, f32] = y\n",
    );
    fmt_inplace(&path);

    let (code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert_eq!(code, 0, "check exit 0 expected; out={out}");
    assert!(out.contains("\"score\": 1"), "score must be 1; out={out}");
}

/// SR-LEAK-A Path A, `if`-tail-var body, divergent dim. Both branches
/// resolve to `y: &tensor[m, f32]`, declared return is `tensor[n, f32]`.
#[test]
fn sr_leak_a_path_a_if_tail_divergent_dim_fails() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("if_tail_divergent.ch");
    write_file(
        &path,
        "module Repro.IfTailDivergent\n\
         def f[n, m](c: bool, x: &tensor[n, f32], y: &tensor[m, f32]) -> tensor[n, f32] = \
         if c then y else y\n",
    );
    fmt_inplace(&path);

    let (_code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert!(
        out.contains("TypeMismatch"),
        "if-tail-var body returning the wrong dim param must surface TypeMismatch; out={out}"
    );
}

/// SR-LEAK-A Path A POSITIVE CONTROL, `match`-tail-var body, same dim.
/// Every arm returns `x: &tensor[n, f32]`, declared return tensor[n, f32].
#[test]
fn sr_leak_a_path_a_match_tail_same_dim_passes() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("match_tail_same.ch");
    write_file(
        &path,
        "module Repro.MatchTailSame\n\
         type Choice = | Left | Right\n\
         def f[n](c: Choice, x: &tensor[n, f32]) -> tensor[n, f32] = match c with {\n  \
           | Left => x\n  \
           | Right => x\n\
         }\n",
    );
    fmt_inplace(&path);

    let (code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert_eq!(code, 0, "check exit 0 expected; out={out}");
    assert!(out.contains("\"score\": 1"), "score must be 1; out={out}");
}

/// SR-LEAK-A (TypeCheck-FreeDimVarUnification-F1), Path B: a plain
/// owned-tensor body that never routes through `types_structurally_equal`
/// at all. `def g[n, m](x: tensor[n, f32], y: tensor[m, f32]) ->
/// tensor[n, f32] = y` collapses two distinct declared dim params via
/// free `unify_dim`. The post-body `declared_dvars` rigidity check must
/// flag the `Var->Var` collapse with a DimensionMismatch.
#[test]
fn sr_leak_a_path_b_plain_owned_divergent_dim_fails() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("plain_owned_divergent.ch");
    write_file(
        &path,
        "module Repro.PlainOwnedDivergent\n\
         def g[n, m](x: tensor[n, f32], y: tensor[m, f32]) -> tensor[n, f32] = y\n",
    );
    fmt_inplace(&path);

    let (_code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert!(
        out.contains("DimensionMismatch") || out.contains("TypeMismatch"),
        "plain owned-tensor body collapsing two declared dim params must fail; out={out}"
    );
}

/// SR-LEAK-A Path B POSITIVE CONTROL: a plain owned-tensor body with a
/// single declared dim param, genuinely valid. `def h[n](x: tensor[n,
/// f32], y: tensor[n, f32]) -> tensor[n, f32] = y` returns a value
/// whose dim matches the declared return. No collapse, must pass.
#[test]
fn sr_leak_a_path_b_plain_owned_same_dim_passes() {
    let dir = tempdir().expect("tempdir");
    let path = dir.path().join("plain_owned_same.ch");
    write_file(
        &path,
        "module Repro.PlainOwnedSame\n\
         def h[n](x: tensor[n, f32], y: tensor[n, f32]) -> tensor[n, f32] = y\n",
    );
    fmt_inplace(&path);

    let (code, out, _err) = run_chelis(dir.path(), &["check", path.to_str().unwrap()]);
    assert_eq!(code, 0, "check exit 0 expected; out={out}");
    assert!(out.contains("\"score\": 1"), "score must be 1; out={out}");
}
