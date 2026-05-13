//! Wave 3 terminal red team for the 0.7.8 compiler-cleanup workstream.
//!
//! Adversarial fixtures for the CLI-level §5 closures:
//! - HostEval-ScalarFn-F1 (nested zero-arg fn calls, zero-arg in subexpr)
//! - Lint CLI path-walk (Item 7) plus the Lint-ExceptionPathRoot-F1
//!   sibling-sweep §5 entry surfaced by PR #93.
//!
//! Each fixture has a pinned expected outcome; either an exact stdout
//! match or an exact exit-code / pattern.

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

fn eval_file(path: &Path) -> assert_cmd::assert::Assert {
    Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["eval", "--file", path.to_str().unwrap()])
        .assert()
}

// ============================================================
// §3.1 HostEval-ScalarFn-F1 nested zero-arg fn-call adversarial
// ============================================================

/// Two-level nested zero-arg: `outer()` calls `inner()`. The W3 fix
/// at `lower_app`'s arity guard must let both calls through. The
/// existing fixture in `host_eval_scalar_fn_call.rs` only exercises
/// one-level zero-arg.
#[test]
fn host_eval_two_level_nested_zero_arg_i32() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("two_level_nested.ch");
    write_file(
        &fixture,
        "def inner -> i32 = 42\ndef outer -> i32 = inner()\nresult = outer()\n",
    );

    eval_file(&fixture)
        .success()
        .stdout(predicate::str::contains("tensor(shape=[], data=[42.0])"));
}

/// Three-level zero-arg chain. Each level returns f32 via the bare-def
/// form (no parens).
#[test]
fn host_eval_three_level_nested_zero_arg_f32() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("three_level_nested.ch");
    write_file(
        &fixture,
        "def deepest -> f32 = 3.14\n\
         def middle -> f32 = deepest()\n\
         def outer -> f32 = middle()\n\
         result = outer()\n",
    );

    eval_file(&fixture)
        .success()
        .stdout(predicate::str::contains("tensor(shape=[], data=[3.14"));
}

/// Zero-arg fn-call used as a subexpression inside a non-trivial app.
/// `result = add(go(), 1)` exercises the `lower_app` arity guard on
/// a nested `(app ... (var go))` form embedded in another app's
/// arglist.
#[test]
fn host_eval_zero_arg_in_subexpression() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("zero_arg_in_subexpr.ch");
    write_file(&fixture, "def go -> i32 = 7\nresult = add(go(), 1)\n");

    eval_file(&fixture).success().stdout("8\n");
}

/// Zero-arg fn-call inside another zero-arg fn body. `def go = add(helper(), 3)`
/// nests the call through `lower_app` twice in a single decl.
#[test]
fn host_eval_zero_arg_inside_zero_arg_body() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("zero_arg_in_body.ch");
    write_file(
        &fixture,
        "def helper -> i32 = 5\n\
         def go -> i32 = add(helper(), 3)\n\
         result = go()\n",
    );

    eval_file(&fixture).success().stdout("8\n");
}

/// Zero-arg fn returning bool.
#[test]
fn host_eval_zero_arg_bool() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("zero_arg_bool.ch");
    write_file(&fixture, "def go -> bool = true\nresult = go()\n");

    eval_file(&fixture)
        .success()
        .stdout(predicate::str::contains("tensor(shape=[], data=[1.0])"));
}

/// Zero-arg fn returning large i64 (above f32 representable-int range).
/// If the bug were latent on i64, the round-trip through f32 storage
/// would corrupt large integers; the existing fixture only tests `7`.
#[test]
fn host_eval_zero_arg_i64_large_value() {
    let dir = tempdir().expect("tempdir");
    let fixture = dir.path().join("zero_arg_i64_large.ch");
    write_file(&fixture, "def go -> i64 = 9999999999i64\nresult = go()\n");

    eval_file(&fixture)
        .success()
        .stdout(predicate::str::contains("data=[9999999999"));
}

// ============================================================
// §3.6 Lint CLI consistency (Item 7) — full corpus sweep
// ============================================================

/// `chelis lint --check .` and `chelis lint --check <each subtree>`
/// should produce identical results for paths covered by both
/// invocations. PR #93 canonicalized the path resolution.
///
/// Re-verify after PR #93: walking the canonical sub-tree set should
/// not produce false-positive `doc-filename-convention` errors that
/// `chelis lint --check .` excepts.
#[test]
fn lint_subtree_invocation_matches_dot_for_doc_filename_convention() {
    // Find the repo root via the `chelis` binary's location.
    let exe = Command::cargo_bin("chelis").expect("binary");
    let repo_root = std::env::var("CARGO_MANIFEST_DIR")
        .map(|d| {
            // We're under crates/chelis-cli/Cargo.toml; go up two levels.
            std::path::PathBuf::from(d)
                .parent()
                .unwrap()
                .parent()
                .unwrap()
                .to_path_buf()
        })
        .expect("CARGO_MANIFEST_DIR");

    drop(exe);

    let dot_out = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&repo_root)
        .args(["lint", "--check", "."])
        .output()
        .expect("dot lint");

    let subtree_out = Command::cargo_bin("chelis")
        .expect("binary")
        .current_dir(&repo_root)
        .args([
            "lint", "--check", "crates", "docs", "examples", "packages", "scripts", "spec",
        ])
        .output()
        .expect("subtree lint");

    let dot_err_text = String::from_utf8_lossy(&dot_out.stderr).to_string();
    let subtree_err_text = String::from_utf8_lossy(&subtree_out.stderr).to_string();

    // Count `doc-filename-convention` *errors* (not warnings) on each.
    let count_doc_filename_errors = |s: &str| -> usize {
        s.lines()
            .filter(|line| line.contains("error:") && line.contains("doc-filename-convention"))
            .count()
    };

    let dot_err_count = count_doc_filename_errors(&dot_err_text);
    let subtree_err_count = count_doc_filename_errors(&subtree_err_text);

    assert_eq!(
        dot_err_count, subtree_err_count,
        "doc-filename-convention error counts must match between `lint .` ({dot_err_count}) and \
         explicit subtree walk ({subtree_err_count}). PR #93 closure says they should be \
         identical."
    );
}
