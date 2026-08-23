//! Binary smoke for `chelis reef conform bump`'s categorized post-run report
//! (chelis#655): a retrofit whose own output (pins/stamps/skills) is clean but
//! that leaves *author follow-up* rows must succeed (exit 0) and list the
//! remaining steps, rather than reading as a bump failure. The classification
//! itself is unit-tested in `chelis-conformance::bump`.
//!
//! Also the write-path preflight (chelis#1263): on a repo that has not been
//! conformed, `bump` and `sync` must refuse BEFORE their first write. The
//! precondition is unit-tested in `chelis-conformance`; what only the binary can
//! prove is the pair of properties a caller actually depends on, namely a
//! nonzero exit AND a tree that is byte-identical to the one the command found.

use std::collections::BTreeMap;
use std::path::Path;

use assert_cmd::Command;
use predicates::prelude::*;
use tempfile::tempdir;

/// A chelis version below the toolchain's own, so a bump to the current version
/// has real pins to rewrite. A no-op bump would make the "nothing was written"
/// assertions vacuous.
const OLD_PIN: &str = "0.17.0";

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

/// Stamp a shell and roll its pins back to [`OLD_PIN`], then delete `drop` (the
/// pre-conformance gap under test). Returns the tree snapshot to compare against
/// after the refusal.
fn pre_conformance_shell(root: &Path, drop: &[&str]) -> BTreeMap<String, String> {
    init_shell(root);
    let ver = chelis_compiler_api::COMPILER_VERSION;
    // Roll the managed-block stamps back too, not just the pins: a shell sitting
    // at OLD_PIN has stale stamps, and leaving them at the current version would
    // make "the refusal did not restamp" unprovable. The body hash is unaffected
    // (it covers the body, not the stamped version).
    for rel in [
        "reef.toml",
        ".github/workflows/ci.yml",
        "AGENTS.md",
        "docs/CHELIS_SURFACE.md",
    ] {
        let p = root.join(rel);
        let text = std::fs::read_to_string(&p).unwrap().replace(ver, OLD_PIN);
        std::fs::write(&p, text).unwrap();
    }
    for rel in drop {
        let p = root.join(rel);
        if p.is_dir() {
            std::fs::remove_dir_all(&p).unwrap();
        } else {
            std::fs::remove_file(&p).unwrap();
        }
    }
    snapshot(root)
}

/// Every path under `root` mapped to its content (or, for a symlink, its
/// target). Symlinks are recorded by target rather than followed: `CLAUDE.md`
/// points at `AGENTS.md`, which some fixtures deliberately delete.
fn snapshot(root: &Path) -> BTreeMap<String, String> {
    let mut out = BTreeMap::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        for entry in std::fs::read_dir(&dir).unwrap().flatten() {
            let path = entry.path();
            let rel = path
                .strip_prefix(root)
                .unwrap()
                .to_string_lossy()
                .into_owned();
            let ty = entry.file_type().unwrap();
            if ty.is_symlink() {
                let target = std::fs::read_link(&path).unwrap();
                out.insert(rel, format!("symlink -> {}", target.display()));
            } else if ty.is_dir() {
                out.insert(rel, "<dir>".to_string());
                stack.push(path);
            } else {
                out.insert(rel, std::fs::read_to_string(&path).unwrap_or_default());
            }
        }
    }
    out
}

fn assert_unchanged(root: &Path, before: &BTreeMap<String, String>) {
    let after = snapshot(root);
    let added: Vec<&String> = after.keys().filter(|k| !before.contains_key(*k)).collect();
    let removed: Vec<&String> = before.keys().filter(|k| !after.contains_key(*k)).collect();
    let edited: Vec<&String> = before
        .iter()
        .filter(|(k, v)| after.get(*k).is_some_and(|now| now != *v))
        .map(|(k, _)| k)
        .collect();
    assert!(
        added.is_empty() && removed.is_empty() && edited.is_empty(),
        "a refused conform verb must leave the tree untouched.\n  created: {added:?}\n  \
         deleted: {removed:?}\n  modified: {edited:?}"
    );
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
        "def f() -> I32 = fail(\"blocked on chelis#999\")\n",
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

#[test]
fn bump_with_a_bump_owned_failure_exits_nonzero() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    init_shell(&root);

    // Inject a defect in a bump-owned artifact: a malformed shell-local block
    // (no end marker) on a shared skill. `materialize_skills` preserves it, so
    // §8 (row `vendored-skills`, bump-owned) fails after the bump — which MUST
    // read as a failure (exit 1), not an author-follow-up exit 0. This is the
    // negative parity for `bump_with_only_author_follow_up_exits_zero...`.
    let skill = root.join("agent-skills/spec-sync/SKILL.md");
    let body = std::fs::read_to_string(&skill).expect("scaffolded skill");
    std::fs::write(
        &skill,
        format!("{body}\n<!-- shell-local:begin -->\nno end marker\n"),
    )
    .unwrap();

    let ver = chelis_compiler_api::COMPILER_VERSION;
    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "bump", ver, "--path"])
        .arg(&root)
        .assert()
        .failure()
        .stderr(predicate::str::contains("bump-owned artifact"))
        .stderr(predicate::str::contains("vendored-skills"));
}

// ------------------------------------------------- chelis#1263 write preflight

/// The hello-chelis shape: never conformed, no `AGENTS.md`. Measured on 0.18.5,
/// `conform bump` repinned `reef.toml` and `ci.yml`, materialized
/// `agent-skills/`, and only then died on the restamp.
#[test]
fn bump_on_a_repo_without_agents_md_refuses_and_writes_nothing() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    let before = pre_conformance_shell(&root, &["AGENTS.md", "CLAUDE.md"]);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "conform",
            "bump",
            chelis_compiler_api::COMPILER_VERSION,
            "--path",
        ])
        .arg(&root)
        .assert()
        .failure()
        .stderr(predicate::str::contains("AGENTS.md"))
        .stderr(predicate::str::contains("Nothing was written"))
        .stderr(predicate::str::contains("chelis reef conform init"));

    assert_unchanged(&root, &before);
    assert!(
        std::fs::read_to_string(root.join("reef.toml"))
            .unwrap()
            .contains(OLD_PIN),
        "the pin must not have moved"
    );
}

/// The c-earchin / calcify shape: `docs/CHELIS_SURFACE.md` absent. This one is
/// the sharper regression lock, because the pre-fix run got FURTHER: it repinned,
/// materialized skills, AND restamped `AGENTS.md` before failing, so the failure
/// left a managed block stamped for a version the shell had not adopted.
#[test]
fn bump_on_a_repo_without_chelis_surface_refuses_and_writes_nothing() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    let before = pre_conformance_shell(&root, &["docs/CHELIS_SURFACE.md"]);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "conform",
            "bump",
            chelis_compiler_api::COMPILER_VERSION,
            "--path",
        ])
        .arg(&root)
        .assert()
        .failure()
        .stderr(predicate::str::contains("docs/CHELIS_SURFACE.md"))
        .stderr(predicate::str::contains("Nothing was written"));

    assert_unchanged(&root, &before);
    assert!(
        !std::fs::read_to_string(root.join("AGENTS.md"))
            .unwrap()
            .contains(chelis_compiler_api::COMPILER_VERSION),
        "the AGENTS.md managed block must not have been restamped ahead of the refusal"
    );
}

/// `sync` shares the restamp step and had the same partial-write shape.
#[test]
fn sync_on_a_pre_conformance_repo_refuses_and_writes_nothing() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    let before = pre_conformance_shell(&root, &["docs/CHELIS_SURFACE.md"]);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "sync", "--path"])
        .arg(&root)
        .assert()
        .failure()
        .stderr(predicate::str::contains("docs/CHELIS_SURFACE.md"))
        .stderr(predicate::str::contains("Nothing was written"));

    assert_unchanged(&root, &before);
}

/// A repo with none of the managed documents reports the whole gap in one run,
/// rather than making a retrofit agent rediscover the next missing file after
/// fixing the previous one.
#[test]
fn bump_names_every_missing_artifact_in_one_refusal() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    let before =
        pre_conformance_shell(&root, &["AGENTS.md", "CLAUDE.md", "docs/CHELIS_SURFACE.md"]);

    Command::cargo_bin("chelis")
        .expect("binary")
        .args([
            "reef",
            "conform",
            "bump",
            chelis_compiler_api::COMPILER_VERSION,
            "--path",
        ])
        .arg(&root)
        .assert()
        .failure()
        .stderr(predicate::str::contains("AGENTS.md"))
        .stderr(predicate::str::contains("docs/CHELIS_SURFACE.md"));

    assert_unchanged(&root, &before);
}

/// Negative parity for every refusal above: a fully conformed shell still bumps
/// cleanly, exit 0, with the pins actually rewritten. A preflight that blocked a
/// normal bump would be a worse defect than the partial write it replaces.
#[test]
fn bump_on_a_conformed_shell_still_succeeds_and_moves_the_pins() {
    let dir = tempdir().unwrap();
    let root = dir.path().join("shell");
    pre_conformance_shell(&root, &[]);
    // Drop the scaffolded negative probe so the executable suites are skipped;
    // this test isolates the write path.
    std::fs::remove_dir_all(root.join("tests_neg/example")).unwrap();
    let ver = chelis_compiler_api::COMPILER_VERSION;

    Command::cargo_bin("chelis")
        .expect("binary")
        .args(["reef", "conform", "bump", ver, "--path"])
        .arg(&root)
        .assert()
        .success()
        .stdout(predicate::str::contains("mechanically complete"));

    let reef = std::fs::read_to_string(root.join("reef.toml")).unwrap();
    assert!(reef.contains(&format!("\"={ver}\"")), "reef pin: {reef}");
    let ci = std::fs::read_to_string(root.join(".github/workflows/ci.yml")).unwrap();
    assert!(ci.contains(&format!("CHELIS_VERSION: {ver}")), "ci: {ci}");
    let agents = std::fs::read_to_string(root.join("AGENTS.md")).unwrap();
    assert!(
        agents.contains(&format!("chelis@{ver}")),
        "the managed block must be restamped"
    );
}
