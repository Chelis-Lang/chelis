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
#[ignore = "lint CLI path-walk inconsistency, see fix/lint-cli-path-walk-consistency"]
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
