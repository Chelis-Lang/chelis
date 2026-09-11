//! chelis#1678: `chelis check <dir>` follows symlinks.
//!
//! The directory walk used `walkdir` without following links, and then
//! dropped every entry whose type was not a regular file. A symlinked
//! source file, or a symlinked directory of sources, was therefore never
//! checked -- with exit 0 and no warning. Worst case: a directory holding a
//! clean file beside a symlink to a file with a genuine type error reported
//! success, while checking that same file directly exited 2. A type error
//! disappeared by being behind a link.
//!
//! Following links introduces failure modes the unfollowed walk never saw,
//! and `walkdir` surfaces each as a walk ERROR, which aborts the whole run.
//! The tests below pin the three this change handles: a loop, a dangling
//! link to a source, and a dangling link to anything else.
//!
//! Deliberately NOT pinned: every other link that fails to resolve -- a link
//! to a directory that cannot be read, a self-referencing link (`ELOOP`), a
//! link through a file (`ENOTDIR`). Those still abort the run, whatever the
//! link is named, because `walkdir` resolves a followed link before its entry
//! filter sees the name. Directory-level failures are the next chelis#1678
//! change's surface, and asserting today's abort would pin it.

#![cfg(unix)]

use std::fs;
use std::os::unix::fs::symlink;
use std::path::Path;

use assert_cmd::Command;
use tempfile::tempdir;

const CLEAN: &str = "def f(x: f32) -> f32 = add(x, x)\n";
const BROKEN: &str = "def g(x: f32) -> f32 = add(x, nope)\n";

/// Run `chelis check` on a directory: (exit code, entry paths, envelope).
fn check_dir(dir: &Path) -> (Option<i32>, Vec<String>, serde_json::Value) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .args(["check"])
        .arg(dir)
        .output()
        .expect("run chelis check");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).to_string();
    let envelope: serde_json::Value = serde_json::from_str(&stdout).unwrap_or_else(|e| {
        panic!("directory mode must emit an envelope ({e}); stdout={stdout} stderr={stderr}")
    });
    let files = envelope["files"]
        .as_array()
        .expect("files array")
        .iter()
        .map(|entry| entry["file"].as_str().unwrap_or("").to_string())
        .collect();
    (output.status.code(), files, envelope)
}

fn entry<'a>(envelope: &'a serde_json::Value, name: &str) -> &'a serde_json::Value {
    envelope["files"]
        .as_array()
        .expect("files array")
        .iter()
        .find(|e| e["file"] == name)
        .unwrap_or_else(|| panic!("no entry for {name}: {envelope}"))
}

#[test]
fn a_type_error_behind_a_symlink_is_not_hidden() {
    // The defect this change exists for, in its worst form.
    let root = tempdir().expect("tempdir");
    let real = root.path().join("real");
    let checked = root.path().join("checked");
    fs::create_dir_all(&real).expect("mkdir");
    fs::create_dir_all(&checked).expect("mkdir");
    fs::write(real.join("broken.ch"), BROKEN).expect("write");
    fs::write(checked.join("a.ch"), CLEAN).expect("write");
    symlink(real.join("broken.ch"), checked.join("broken_link.ch")).expect("symlink");

    let (code, files, envelope) = check_dir(&checked);
    assert_eq!(
        files,
        vec!["a.ch", "broken_link.ch"],
        "both sources are checked"
    );
    let errors = entry(&envelope, "broken_link.ch")["report"]["errors"]
        .as_array()
        .expect("errors")
        .len();
    assert!(
        errors > 0,
        "the linked file's type error is reported: {envelope}"
    );
    assert_eq!(code, Some(2), "a type error anywhere in the walk exits 2");
}

#[test]
fn a_symlinked_source_file_is_checked() {
    let root = tempdir().expect("tempdir");
    let real = root.path().join("real");
    let checked = root.path().join("checked");
    fs::create_dir_all(&real).expect("mkdir");
    fs::create_dir_all(&checked).expect("mkdir");
    fs::write(real.join("r.ch"), CLEAN).expect("write");
    symlink(real.join("r.ch"), checked.join("linked.ch")).expect("symlink");

    let (code, files, _) = check_dir(&checked);
    // The entry is named by the path the walk took -- the link -- not the
    // resolved target, so a user sees the name that is in their directory.
    assert_eq!(files, vec!["linked.ch"]);
    assert_eq!(code, Some(0));
}

#[test]
fn a_symlinked_source_directory_is_checked() {
    let root = tempdir().expect("tempdir");
    let real = root.path().join("real");
    let checked = root.path().join("checked");
    fs::create_dir_all(&real).expect("mkdir");
    fs::create_dir_all(&checked).expect("mkdir");
    fs::write(real.join("r.ch"), CLEAN).expect("write");
    fs::write(
        real.join("s.dp"),
        "(def {} k (lit {type: (t-prim {} f32)} 1.0))\n",
    )
    .expect("write");
    symlink(&real, checked.join("vendored")).expect("symlink");

    let (_, files, _) = check_dir(&checked);
    assert_eq!(
        files,
        vec!["vendored/r.ch", "vendored/s.dp"],
        "the linked tree is walked"
    );
}

#[test]
fn a_symlink_loop_does_not_abort_the_walk() {
    // Regression guard. An unfollowed walk never saw the loop; a following
    // walk reports it as an error, which would abort the run. The link only
    // points back at an ancestor already being walked, so skipping it loses
    // nothing.
    let root = tempdir().expect("tempdir");
    let nested = root.path().join("nested");
    fs::create_dir_all(&nested).expect("mkdir");
    fs::write(root.path().join("x.ch"), CLEAN).expect("write");
    symlink("..", nested.join("up")).expect("symlink");

    let (code, files, _) = check_dir(root.path());
    assert_eq!(
        files,
        vec!["x.ch"],
        "the real file is checked once, the loop skipped"
    );
    assert_eq!(code, Some(0));
}

#[test]
fn a_dangling_link_to_a_source_is_reported_not_skipped() {
    // A link named `gone.ch` is a source the user expects checked. It cannot
    // be read, so it gets exactly the report that naming it directly would
    // produce -- not silence (the old skip) and not an aborted run (a naive
    // follow).
    let root = tempdir().expect("tempdir");
    fs::write(root.path().join("a.ch"), CLEAN).expect("write");
    symlink(
        root.path().join("does_not_exist.ch"),
        root.path().join("gone.ch"),
    )
    .expect("symlink");

    let (code, files, envelope) = check_dir(root.path());
    assert_eq!(files, vec!["a.ch", "gone.ch"]);
    let messages: Vec<String> = entry(&envelope, "gone.ch")["report"]["errors"]
        .as_array()
        .expect("errors")
        .iter()
        .map(|d| d["message"].as_str().unwrap_or("").to_string())
        .collect();
    assert!(
        messages.iter().any(|m| m.contains("failed to read")),
        "the dangling link reports why it could not be checked: {messages:?}"
    );
    assert_eq!(code, Some(2));
}

#[test]
fn a_dangling_link_to_a_non_source_does_not_abort_the_walk() {
    // Regression guard. Not a source, so there is nothing to check -- the
    // same treatment as a regular file without a `.ch` or `.dp` suffix.
    let root = tempdir().expect("tempdir");
    fs::write(root.path().join("a.ch"), CLEAN).expect("write");
    symlink(
        root.path().join("no_such_dir"),
        root.path().join("stale_link"),
    )
    .expect("symlink");

    let (code, files, _) = check_dir(root.path());
    assert_eq!(files, vec!["a.ch"]);
    assert_eq!(code, Some(0));
}

#[test]
fn followed_links_still_respect_the_hidden_and_target_filters() {
    // Following links must not become a way around the existing skip
    // rules: a link named like a dot-directory or `target` is skipped by its
    // own name, exactly as a real directory of that name is.
    let root = tempdir().expect("tempdir");
    let real = root.path().join("real");
    let checked = root.path().join("checked");
    fs::create_dir_all(&real).expect("mkdir");
    fs::create_dir_all(&checked).expect("mkdir");
    fs::write(real.join("r.ch"), CLEAN).expect("write");
    fs::write(checked.join("a.ch"), CLEAN).expect("write");
    symlink(&real, checked.join(".cache")).expect("symlink");
    symlink(&real, checked.join("target")).expect("symlink");
    symlink(real.join("r.ch"), checked.join(".hidden.ch")).expect("symlink");

    let (_, files, _) = check_dir(&checked);
    assert_eq!(
        files,
        vec!["a.ch"],
        "no filtered name is collected through a link"
    );
}

#[test]
fn a_dangling_hidden_link_is_still_filtered() {
    // The dangling-link path runs outside the walker's own filter, because
    // an entry that fails to resolve is reported as an error before the
    // filter sees it. It has to apply the dot-prefix rule itself.
    let root = tempdir().expect("tempdir");
    fs::write(root.path().join("a.ch"), CLEAN).expect("write");
    symlink(root.path().join("nothing.ch"), root.path().join(".swap.ch")).expect("symlink");

    let (code, files, _) = check_dir(root.path());
    assert_eq!(
        files,
        vec!["a.ch"],
        "a hidden dangling link is not collected"
    );
    assert_eq!(code, Some(0));
}

// ---------------------------------------------------------------------------
// `chelis test` walks for test files with the same helper, and had the same
// defect: a symlinked test file was never run, so a failing test behind a
// link let `chelis test` report every test passing.
// ---------------------------------------------------------------------------

/// A bare reef package with a `tests/` dir, mirroring `expect_fail_modes.rs`.
/// Tests call `test_assert` as a direct runtime builtin, so no std dep.
fn make_test_package() -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempdir().expect("tempdir");
    let pkg = dir.path().join("pkg");
    fs::create_dir_all(pkg.join("src")).expect("mkdir");
    fs::create_dir_all(pkg.join("tests")).expect("mkdir");
    fs::write(
        pkg.join("reef.toml"),
        format!(
            "[package]\nname = \"pkg\"\nversion = \"0.1.0\"\ncompiler = \"={ver}\"\nmodule_prefix = \"Probe\"\n",
            ver = chelis_compiler_api::COMPILER_VERSION,
        ),
    )
    .expect("write reef.toml");
    fs::write(
        pkg.join("src/main.ch"),
        "module Probe.Main\n\ndef noop() -> unit = test_assert(true, \"noop\")\n",
    )
    .expect("write main");
    (dir, pkg)
}

/// Run `chelis test tests/ --json` in `pkg`: (exit code, summary line).
fn run_tests(pkg: &Path) -> (Option<i32>, serde_json::Value) {
    let output = Command::cargo_bin("chelis")
        .expect("binary")
        .env("CHELIS_STYLE_GATE_DISABLE", "1")
        .current_dir(pkg)
        .args(["test", "tests/", "--json"])
        .output()
        .expect("run chelis test");
    let stdout = String::from_utf8_lossy(&output.stdout).to_string();
    let summary = stdout
        .lines()
        .filter_map(|line| serde_json::from_str::<serde_json::Value>(line).ok())
        .find(|v| v.get("summary").is_some())
        .unwrap_or_else(|| panic!("no summary line; stdout={stdout}"));
    (output.status.code(), summary["summary"].clone())
}

#[test]
fn chelis_test_runs_a_symlinked_test_file() {
    let (_dir, pkg) = make_test_package();
    fs::create_dir_all(pkg.join("elsewhere")).expect("mkdir");
    fs::write(
        pkg.join("tests/real.ch"),
        "module Probe.Tests.Real\n\ndef test_real_passes() -> unit = test_assert(true, \"real\")\n",
    )
    .expect("write");
    fs::write(
        pkg.join("elsewhere/linked_target.ch"),
        "module Probe.Tests.Linked\n\ndef test_linked_fails() -> unit = test_assert(false, \"the linked test ran\")\n",
    )
    .expect("write");
    symlink("../elsewhere/linked_target.ch", pkg.join("tests/linked.ch")).expect("symlink");

    let (code, summary) = run_tests(&pkg);
    assert_eq!(
        summary,
        serde_json::json!({"passed": 1, "failed": 1}),
        "the symlinked test runs and its failure counts"
    );
    assert_ne!(
        code,
        Some(0),
        "a failing test behind a link must fail the run, as it would unlinked"
    );
}

#[test]
fn chelis_test_is_not_aborted_by_a_symlink_loop() {
    // Regression guard: a loop inside `tests/` must not turn into a walk
    // error that aborts the whole run.
    let (_dir, pkg) = make_test_package();
    fs::write(
        pkg.join("tests/real.ch"),
        "module Probe.Tests.Real\n\ndef test_real_passes() -> unit = test_assert(true, \"real\")\n",
    )
    .expect("write");
    fs::create_dir_all(pkg.join("tests/nested")).expect("mkdir");
    symlink("..", pkg.join("tests/nested/up")).expect("symlink");

    let (code, summary) = run_tests(&pkg);
    assert_eq!(summary, serde_json::json!({"passed": 1, "failed": 0}));
    assert_eq!(code, Some(0));
}
