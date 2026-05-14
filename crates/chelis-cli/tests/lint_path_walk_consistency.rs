//! Regression tests pinning lint CLI path-walk consistency between explicit
//! path arguments and the CWD walk.
//!
//! Bug shape: `chelis lint --check src/ docs/` and `chelis lint --check .`
//! (with the same CWD) should produce identical violation sets for the same
//! file tree. Today they diverge on rules whose classification depends on a
//! substring match against the absolute or rooted path (the canonical case
//! is `doc-filename-convention` §8.3, which checks for the literal
//! `"/docs/"` substring in `path.to_string_lossy()`).
//!
//! When invoked as `chelis lint --check .`, the walker produces paths like
//! `./docs/0_leading_digit.md` which contain `/docs/`. When invoked as
//! `chelis lint --check docs/`, the walker produces paths like
//! `docs/0_leading_digit.md` which do NOT contain `/docs/`, so the rule's
//! classification falls through to `Slot::Other` and no violation fires.
//!
//! The canonical behavior is the CWD walk (per CI:
//! `chelis lint --check .` is the gate that runs). The explicit-path walk
//! must behave identically.

use assert_cmd::Command;
use std::fs;
use tempfile::tempdir;

/// Build a minimal fixture: `src/example.ch` plus
/// `docs/0_leading_digit.md` (the latter triggers `doc-filename-convention`
/// §8.3 since narrative docs require snake_case starting with a lowercase
/// letter, not a leading digit).
fn write_fixture(root: &std::path::Path) {
    fs::create_dir_all(root.join("src")).expect("mkdir src");
    fs::create_dir_all(root.join("docs")).expect("mkdir docs");
    fs::write(root.join("src/example.ch"), "def main -> i32 = 0\n").expect("write surf");
    fs::write(root.join("docs/0_leading_digit.md"), "# placeholder doc\n").expect("write doc");
}

/// Run `chelis lint --check <args...>` from `cwd` and return
/// (exit_code, stdout) for assertions.
fn run_lint(cwd: &std::path::Path, extra_args: &[&str]) -> (i32, String) {
    let mut cmd = Command::cargo_bin("chelis").expect("binary");
    cmd.current_dir(cwd);
    cmd.arg("lint").arg("--check");
    for a in extra_args {
        cmd.arg(a);
    }
    let output = cmd.output().expect("run chelis lint");
    let stdout = String::from_utf8_lossy(&output.stdout).into_owned();
    let code = output.status.code().unwrap_or(-1);
    (code, stdout)
}

/// Pin the consistency invariant: both invocations must produce identical
/// stdout (modulo file order, which is path-sorted anyway by the lint
/// driver). Concretely: the `doc-filename-convention` violation on
/// `docs/0_leading_digit.md` must fire in BOTH invocations.
#[test]
fn lint_cli_explicit_path_and_cwd_walk_produce_identical_violations() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_fixture(root);

    let (code_cwd, out_cwd) = run_lint(root, &["."]);
    let (code_explicit, out_explicit) = run_lint(root, &["src/", "docs/"]);

    // Both invocations must report the doc-filename-convention violation
    // (the bug today is that the explicit-path walk silently passes).
    let needle = "doc-filename-convention";
    assert!(
        out_cwd.contains(needle),
        "CWD walk should report doc-filename-convention, got stdout:\n{out_cwd}"
    );
    assert!(
        out_explicit.contains(needle),
        "explicit-path walk should report doc-filename-convention, got stdout:\n{out_explicit}"
    );

    // Both invocations should produce the same blocking-error exit code
    // (1 when violations exist with --check, 0 when none).
    assert_eq!(
        code_cwd, code_explicit,
        "exit codes must match between invocations; CWD={code_cwd} explicit={code_explicit}"
    );

    // The exit code must be 1 (a blocking error fired). If it's 0, both
    // walks suppressed the error, which is the wrong canonical behavior.
    assert_eq!(
        code_cwd, 1,
        "expected blocking exit code 1 from CWD walk; stdout was:\n{out_cwd}"
    );
}

/// Sibling pin: same fixture, but invoke with `./src/` and `./docs/`
/// (leading-dot prefix). This invocation today does fire the rule (the
/// path has a `/docs/` substring), so it serves as a positive control
/// distinguishing the bug class from a generic explicit-path failure.
#[test]
fn lint_cli_dot_prefix_explicit_path_already_fires_rule() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_fixture(root);

    let (code, out) = run_lint(root, &["./src/", "./docs/"]);
    assert!(
        out.contains("doc-filename-convention"),
        "`./docs/` prefix should fire the rule today; stdout was:\n{out}"
    );
    assert_eq!(code, 1, "expected blocking exit code 1");
}

/// Build a fixture that exercises exception-list matching. The shared
/// `style_gate::exceptions()` registry contains an entry for
/// `crates/chelis-surf/tests/fixtures/*.ch` with rule
/// `surf-def-arrow-form` (the Surf parser test corpus deliberately
/// exercises legacy colon-form return syntax). The fixture mirrors that
/// path layout so the exception applies, and writes a single file using
/// the colon form to trigger the rule.
fn write_exception_fixture(root: &std::path::Path) {
    // `detect_lint_workspace_root` probes the Cargo workspace root via
    // `cargo locate-project --workspace`. A synthesized tree that
    // exercises workspace-rooted exception matching must carry a real
    // `[workspace]` marker, otherwise the probe finds nothing (the
    // tempdir lives under the system temp dir, outside any Cargo
    // workspace) and exception filtering is skipped because no
    // workspace-rooted glob can apply.
    fs::write(
        root.join("Cargo.toml"),
        "[workspace]\nmembers = []\nresolver = \"2\"\n",
    )
    .expect("write workspace marker");
    let dir = root.join("crates/chelis-surf/tests/fixtures");
    fs::create_dir_all(&dir).expect("mkdir fixture dir");
    fs::write(
        dir.join("block_binding_expr.ch"),
        "def f(x: f32): f32 = {\n  y = mul(x, x)\n  add(y, x)\n}\n",
    )
    .expect("write fixture");
}

/// Pin the exception-matching invariant: workspace-rooted exception
/// patterns must match identically whether the CLI is invoked as
/// `chelis lint --check .` or as `chelis lint --check crates`.
///
/// Bug shape: `apply_exceptions` strip-prefixes the violation path
/// against the walk-target root, not the workspace root. When the
/// walk-target is a sub-directory (e.g., `crates`), the relative path
/// loses its `crates/` segment, so a workspace-rooted exception pattern
/// like `crates/chelis-surf/tests/fixtures/*.ch` never matches and the
/// violation fires as a false positive. CI invokes `chelis lint
/// --check .` so the gate isn't broken, but developers linting
/// sub-trees see spurious errors.
///
/// Both invocations must produce identical stdout and exit code, with
/// zero `surf-def-arrow-form` violations because the exception applies.
#[test]
fn lint_cli_exception_pattern_matches_under_subtree_and_cwd_walks() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_exception_fixture(root);

    let (code_cwd, out_cwd) = run_lint(root, &["."]);
    let (code_explicit, out_explicit) = run_lint(root, &["crates"]);

    // Neither invocation should report `surf-def-arrow-form` on the
    // fixture: the workspace-rooted exception is in effect for both.
    let needle = "surf-def-arrow-form";
    assert!(
        !out_cwd.contains(needle),
        "CWD walk must apply workspace-rooted exception; stdout was:\n{out_cwd}"
    );
    assert!(
        !out_explicit.contains(needle),
        "subtree walk must apply workspace-rooted exception (same shape as CWD walk); stdout was:\n{out_explicit}"
    );

    // Both invocations must exit with the same code (no blocking
    // violation, since the only rule that would fire is excepted).
    assert_eq!(
        code_cwd, code_explicit,
        "exit codes must match between invocations; CWD={code_cwd} explicit={code_explicit}"
    );
    assert_eq!(
        code_cwd, 0,
        "expected exit 0 from CWD walk (exception suppresses the only would-be violation); stdout was:\n{out_cwd}"
    );
}

/// Sibling pin: even when the user passes multiple sub-tree arguments
/// covering the entire workspace (the concrete reproducer from the
/// gap-synthesis entry), the exception must still apply. This is the
/// real-world failing invocation: `chelis lint --check crates docs
/// examples packages` against a workspace that contains the excepted
/// fixture under `crates/`.
#[test]
fn lint_cli_exception_pattern_matches_under_multi_subtree_walk() {
    let dir = tempdir().expect("tempdir");
    let root = dir.path();
    write_exception_fixture(root);
    // Add the sibling directories the real reproducer cites so the
    // invocation shape mirrors the gap-synthesis entry verbatim.
    fs::create_dir_all(root.join("docs")).expect("mkdir docs");
    fs::create_dir_all(root.join("examples")).expect("mkdir examples");
    fs::create_dir_all(root.join("packages")).expect("mkdir packages");

    let (code, out) = run_lint(root, &["crates", "docs", "examples", "packages"]);
    assert!(
        !out.contains("surf-def-arrow-form"),
        "multi-subtree walk must apply workspace-rooted exception; stdout was:\n{out}"
    );
    assert_eq!(
        code, 0,
        "expected exit 0 (no blocking violations); stdout was:\n{out}"
    );
}
