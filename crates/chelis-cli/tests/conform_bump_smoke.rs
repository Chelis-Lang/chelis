//! Binary smoke for `chelis reef conform bump`'s categorized post-run report
//! (chelis#655): a retrofit whose own output (pins/stamps/skills) is clean but
//! that leaves *author follow-up* rows must succeed (exit 0) and list the
//! remaining steps, rather than reading as a bump failure. The classification
//! itself is unit-tested in `chelis-conformance::bump`.

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

fn init_shell(root: &std::path::Path) {
    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "conform",
            "init",
            "myshell",
            "--module-prefix",
            "Myshell",
            "--output",
        ])
        .arg(root)
        .assert()
        .success();
}

#[test]
fn bump_with_only_author_follow_up_exits_zero_and_lists_steps() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    init_shell(&root);

    // Drop the scaffolded negative probe so the bump's executable suites are
    // skipped — this test isolates the audit-report categorization.
    std::fs::remove_dir_all(root.join("tests_neg/example")).unwrap();

    // Inject an author-follow-up failure the bump cannot fix: a narrowing cite
    // with no tests_blocked/ probe and no UPSTREAM_BUGS entry (rows 9 + 12).
    std::fs::create_dir_all(root.join("src")).unwrap();
    std::fs::write(
        root.join("src/x.ch"),
        "def f -> I32 = fail(\"blocked on chelis#999\")\n",
    )
    .unwrap();

    // Bump to the current version: the bump's own output stays clean, so only
    // the injected follow-up rows remain — which must NOT read as a bump failure.
    let ver = chelis_compiler_api::COMPILER_VERSION;
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "bump", ver, "--path"])
        .arg(&root)
        .assert()
        .success()
        .stdout(predicate::str::contains("mechanically complete"))
        .stdout(predicate::str::contains("conformance step(s) remain"));
}
