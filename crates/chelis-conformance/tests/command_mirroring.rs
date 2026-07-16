//! Shared-skill command projection regression suite.
//!
//! Every effective `agent-skills/<name>/SKILL.md` must be projected byte-for-byte
//! to both command trees. The prior upstream manifest is the only ownership
//! record used to prune retired generated wrappers.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use chelis_conformance::{audit, scaffold, skills};
use sha2::{Digest, Sha256};

const VER: &str = env!("CARGO_PKG_VERSION");
const BLOCK: &str = "<!-- shell-local:begin -->\n## Shell override\nKeep this verbatim.\n<!-- shell-local:end -->\n";

fn shell() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("shell");
    scaffold::scaffold(&root, "shell", "Shell", VER).expect("scaffold");
    (tmp, root)
}

fn row<'a>(report: &'a audit::AuditReport, key: &str) -> &'a audit::RowResult {
    report
        .rows
        .iter()
        .find(|row| row.key == key)
        .unwrap_or_else(|| panic!("missing audit row {key}"))
}

fn skill_path(root: &Path, name: &str) -> PathBuf {
    root.join("agent-skills").join(name).join("SKILL.md")
}

fn wrapper_path(root: &Path, tool: &str, name: &str) -> PathBuf {
    root.join(tool).join("commands").join(format!("{name}.md"))
}

fn bytes(path: impl AsRef<Path>) -> Vec<u8> {
    std::fs::read(path).unwrap()
}

fn write(path: impl AsRef<Path>, body: impl AsRef<[u8]>) {
    let path = path.as_ref();
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).unwrap();
    }
    std::fs::write(path, body).unwrap();
}

fn append(path: &Path, suffix: &str) {
    let mut body = std::fs::read_to_string(path).unwrap();
    body.push_str(suffix);
    std::fs::write(path, body).unwrap();
}

fn sha256_or_missing(path: &Path) -> String {
    match std::fs::read(path) {
        Ok(body) => format!("{:x}", Sha256::digest(body)),
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => "MISSING".to_string(),
        Err(error) => panic!("hash {}: {error}", path.display()),
    }
}

fn hashes(root: &Path, paths: &[&str]) -> BTreeMap<String, String> {
    paths
        .iter()
        .map(|path| ((*path).to_string(), sha256_or_missing(&root.join(path))))
        .collect()
}

fn append_prior_skill(root: &Path, name: &str) {
    let manifest_path = root.join("agent-skills/UPSTREAM.toml");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    let updated = manifest.replace("]\n", &format!("  \"{name}\",\n]\n"));
    assert_ne!(manifest, updated, "fixture must extend the prior manifest");
    std::fs::write(manifest_path, updated).unwrap();
}

fn assert_wrapper_failure(root: &Path, expected_paths: &[&str]) {
    let report = audit::audit(root);
    let result = row(&report, "vendored-skills");
    assert_eq!(result.verdict, audit::Verdict::Fail);
    for path in expected_paths {
        assert!(
            result.diagnostic.contains(path),
            "diagnostic must identify {path}: {}",
            result.diagnostic
        );
    }
    assert!(
        result.fix.contains("chelis reef conform sync"),
        "repair must name conform sync: {}",
        result.fix
    );
}

#[test]
fn changed_effective_skill_with_both_old_wrappers_fails_vendored_skills() {
    let (_tmp, root) = shell();
    let skill = skill_path(&root, "cli-surface");
    let claude = wrapper_path(&root, ".claude", "cli-surface");
    let codex = wrapper_path(&root, ".codex", "cli-surface");
    let previous = bytes(&skill);
    assert_eq!(bytes(&claude), previous);
    assert_eq!(bytes(&codex), previous);

    append(&skill, &format!("\n{BLOCK}"));

    assert_wrapper_failure(
        &root,
        &[
            ".claude/commands/cli-surface.md",
            ".codex/commands/cli-surface.md",
        ],
    );
}

#[test]
fn missing_claude_wrapper_has_path_specific_repair() {
    let (_tmp, root) = shell();
    std::fs::remove_file(wrapper_path(&root, ".claude", "spec-sync")).unwrap();
    assert_wrapper_failure(&root, &[".claude/commands/spec-sync.md"]);
}

#[test]
fn missing_codex_wrapper_has_path_specific_repair() {
    let (_tmp, root) = shell();
    std::fs::remove_file(wrapper_path(&root, ".codex", "spec-sync")).unwrap();
    assert_wrapper_failure(&root, &[".codex/commands/spec-sync.md"]);
}

#[test]
fn one_stale_wrapper_has_path_specific_repair() {
    let (_tmp, root) = shell();
    write(
        wrapper_path(&root, ".claude", "spec-sync"),
        "stale command body\n",
    );
    assert_wrapper_failure(&root, &[".claude/commands/spec-sync.md"]);
}

#[test]
fn claude_and_codex_disagreement_identifies_both_surfaces() {
    let (_tmp, root) = shell();
    write(
        wrapper_path(&root, ".claude", "spec-sync"),
        "stale Claude body\n",
    );
    write(
        wrapper_path(&root, ".codex", "spec-sync"),
        "different stale Codex body\n",
    );
    assert_wrapper_failure(
        &root,
        &[
            ".claude/commands/spec-sync.md",
            ".codex/commands/spec-sync.md",
        ],
    );
}

#[test]
fn fresh_scaffold_projects_every_shared_skill_to_both_command_trees() {
    let (_tmp, root) = shell();
    for name in skills::SHARED_SKILLS {
        let effective = bytes(skill_path(&root, name));
        assert_eq!(
            bytes(wrapper_path(&root, ".claude", name)),
            effective,
            "Claude projection for {name}"
        );
        assert_eq!(
            bytes(wrapper_path(&root, ".codex", name)),
            effective,
            "Codex projection for {name}"
        );
    }
    assert!(audit::audit(&root).ok(), "fresh scaffold must audit green");
}

#[test]
fn sync_adds_both_wrappers_for_a_name_absent_from_the_prior_set() {
    let (_tmp, root) = shell();
    let name = "issue-resolution";
    let manifest_path = root.join("agent-skills/UPSTREAM.toml");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    std::fs::write(
        &manifest_path,
        manifest.replace(&format!("  \"{name}\",\n"), ""),
    )
    .unwrap();
    std::fs::remove_file(skill_path(&root, name)).unwrap();
    std::fs::remove_file(wrapper_path(&root, ".claude", name)).unwrap();
    std::fs::remove_file(wrapper_path(&root, ".codex", name)).unwrap();

    scaffold::materialize_skills(&root).unwrap();

    let effective = bytes(skill_path(&root, name));
    assert_eq!(bytes(wrapper_path(&root, ".claude", name)), effective);
    assert_eq!(bytes(wrapper_path(&root, ".codex", name)), effective);
}

#[test]
fn sync_prunes_only_retired_manifest_owned_wrappers() {
    let (_tmp, root) = shell();
    append_prior_skill(&root, "retired-shared");
    // Even a legacy/mistaken ownership record cannot convert the supported
    // red-team alias into a retired generated projection.
    append_prior_skill(&root, "red-team");
    for tool in [".claude", ".codex"] {
        write(
            wrapper_path(&root, tool, "retired-shared"),
            "old generated body\n",
        );
        write(
            wrapper_path(&root, tool, "red-team"),
            format!("{tool} red-team alias\n"),
        );
        write(
            wrapper_path(&root, tool, "local-command"),
            format!("{tool} local command\n"),
        );
    }
    let mut preserved = Vec::new();
    for tool in [".claude", ".codex"] {
        for name in ["red-team", "local-command"] {
            let path = wrapper_path(&root, tool, name);
            preserved.push((path.clone(), bytes(path)));
        }
    }

    scaffold::materialize_skills(&root).unwrap();

    for tool in [".claude", ".codex"] {
        assert!(
            !wrapper_path(&root, tool, "retired-shared").exists(),
            "retired generated {tool} wrapper must be removed"
        );
    }
    for (path, before) in preserved {
        assert_eq!(bytes(&path), before, "{} must be preserved", path.display());
    }
}

#[test]
fn shell_local_block_is_projected_verbatim_and_second_sync_is_a_no_op() {
    let (_tmp, root) = shell();
    let name = "example-corpus";
    append(&skill_path(&root, name), &format!("\n{BLOCK}"));

    scaffold::materialize_skills(&root).unwrap();
    let paths = [
        skill_path(&root, name),
        wrapper_path(&root, ".claude", name),
        wrapper_path(&root, ".codex", name),
    ];
    let once: Vec<Vec<u8>> = paths.iter().map(bytes).collect();
    assert_eq!(once[0], once[1]);
    assert_eq!(once[0], once[2]);
    assert!(once[0].ends_with(BLOCK.as_bytes()));

    scaffold::materialize_skills(&root).unwrap();
    let twice: Vec<Vec<u8>> = paths.iter().map(bytes).collect();
    assert_eq!(twice, once, "a second sync must be byte-for-byte stable");
}

#[test]
fn sync_repairs_every_repairable_wrapper_failure() {
    for case in 0..4 {
        let (_tmp, root) = shell();
        match case {
            0 => {
                std::fs::remove_file(wrapper_path(&root, ".claude", "spec-sync")).unwrap();
            }
            1 => {
                std::fs::remove_file(wrapper_path(&root, ".codex", "spec-sync")).unwrap();
            }
            2 => write(
                wrapper_path(&root, ".claude", "spec-sync"),
                "stale wrapper\n",
            ),
            3 => {
                write(
                    wrapper_path(&root, ".claude", "spec-sync"),
                    "stale Claude wrapper\n",
                );
                write(
                    wrapper_path(&root, ".codex", "spec-sync"),
                    "different stale Codex wrapper\n",
                );
            }
            _ => unreachable!(),
        }
        assert_eq!(
            row(&audit::audit(&root), "vendored-skills").verdict,
            audit::Verdict::Fail,
            "case {case} must establish drift"
        );

        scaffold::materialize_skills(&root).unwrap();

        assert!(
            audit::audit(&root).ok(),
            "sync must repair wrapper failure case {case}"
        );
    }
}

#[test]
fn a_wrapper_defect_is_owned_by_the_existing_bump_row() {
    let (_tmp, root) = shell();
    write(
        wrapper_path(&root, ".claude", "spec-sync"),
        "stale wrapper\n",
    );
    let report = audit::audit(&root);
    let result = row(&report, "vendored-skills");
    assert_eq!(result.verdict, audit::Verdict::Fail);
    assert!(
        chelis_conformance::bump::is_bump_owned(result.key),
        "same-name command drift remains owned by vendored-skills during a bump"
    );
}

#[test]
fn current_shared_command_names_are_overwritten_not_exempted() {
    let (_tmp, root) = shell();
    write(
        wrapper_path(&root, ".codex", "spec-sync"),
        "attempted shell-local same-name command\n",
    );

    scaffold::materialize_skills(&root).unwrap();

    assert_eq!(
        bytes(wrapper_path(&root, ".codex", "spec-sync")),
        bytes(skill_path(&root, "spec-sync")),
        "current shared names are reserved generated outputs"
    );
}

#[test]
fn pin_bump_lifecycle_fixture_records_exact_before_after_hashes() {
    let (_tmp, root) = shell();
    let manifest_path = root.join("agent-skills/UPSTREAM.toml");
    let manifest = std::fs::read_to_string(&manifest_path).unwrap();
    std::fs::write(
        &manifest_path,
        manifest.replace("  \"issue-resolution\",\n", ""),
    )
    .unwrap();
    append_prior_skill(&root, "retired-shared");

    for tool in [".claude", ".codex"] {
        write(
            wrapper_path(&root, tool, "cli-surface"),
            "previous toolchain CLI body\n",
        );
        std::fs::remove_file(wrapper_path(&root, tool, "issue-resolution")).unwrap();
        write(
            wrapper_path(&root, tool, "retired-shared"),
            "retired generated body\n",
        );
        write(
            wrapper_path(&root, tool, "red-team"),
            format!("{tool} red-team alias\n"),
        );
        write(
            wrapper_path(&root, tool, "local-command"),
            format!("{tool} local command\n"),
        );
    }
    std::fs::remove_file(skill_path(&root, "issue-resolution")).unwrap();
    append(&skill_path(&root, "example-corpus"), &format!("\n{BLOCK}"));

    let tracked = [
        "agent-skills/cli-surface/SKILL.md",
        ".claude/commands/cli-surface.md",
        ".codex/commands/cli-surface.md",
        "agent-skills/issue-resolution/SKILL.md",
        ".claude/commands/issue-resolution.md",
        ".codex/commands/issue-resolution.md",
        "agent-skills/example-corpus/SKILL.md",
        ".claude/commands/example-corpus.md",
        ".codex/commands/example-corpus.md",
        ".claude/commands/retired-shared.md",
        ".codex/commands/retired-shared.md",
        ".claude/commands/red-team.md",
        ".codex/commands/red-team.md",
        ".claude/commands/local-command.md",
        ".codex/commands/local-command.md",
    ];
    let before = hashes(&root, &tracked);

    scaffold::materialize_skills(&root).unwrap();

    let after = hashes(&root, &tracked);
    for path in tracked {
        eprintln!("{path}: {} -> {}", before[path], after[path]);
    }
    for tool in [".claude", ".codex"] {
        assert_eq!(
            bytes(wrapper_path(&root, tool, "cli-surface")),
            bytes(skill_path(&root, "cli-surface"))
        );
        assert_eq!(
            bytes(wrapper_path(&root, tool, "issue-resolution")),
            bytes(skill_path(&root, "issue-resolution"))
        );
        assert_eq!(
            bytes(wrapper_path(&root, tool, "example-corpus")),
            bytes(skill_path(&root, "example-corpus"))
        );
        assert_eq!(
            after[&format!("{tool}/commands/retired-shared.md")],
            "MISSING"
        );
        for preserved in ["red-team", "local-command"] {
            let path = format!("{tool}/commands/{preserved}.md");
            assert_eq!(after[&path], before[&path], "{path} changed");
        }
    }
    assert!(
        audit::audit(&root).ok(),
        "combined bump fixture must converge"
    );
}

#[test]
fn missing_prior_manifest_preserves_unproven_commands_and_repairs_current_wrappers() {
    let (_tmp, root) = shell();
    let unknown = b"unproven generated-looking body\n";
    for tool in [".claude", ".codex"] {
        write(wrapper_path(&root, tool, "legacy-shared"), unknown);
    }
    write(
        wrapper_path(&root, ".claude", "spec-sync"),
        "stale current wrapper\n",
    );
    std::fs::remove_file(root.join("agent-skills/UPSTREAM.toml")).unwrap();

    let notices = scaffold::materialize_skills(&root).unwrap();

    for tool in [".claude", ".codex"] {
        assert_eq!(bytes(wrapper_path(&root, tool, "legacy-shared")), unknown);
    }
    assert_eq!(
        bytes(wrapper_path(&root, ".claude", "spec-sync")),
        bytes(skill_path(&root, "spec-sync"))
    );
    assert!(
        notices
            .iter()
            .any(|notice| notice.contains("UPSTREAM.toml") && notice.contains("preserv")),
        "missing ownership data needs an actionable non-destructive notice: {notices:?}"
    );
}

#[test]
fn malformed_prior_manifest_preserves_unproven_commands_and_repairs_current_wrappers() {
    let (_tmp, root) = shell();
    let unknown = b"local command with an old shared-looking name\n";
    for tool in [".claude", ".codex"] {
        write(wrapper_path(&root, tool, "legacy-shared"), unknown);
    }
    write(
        root.join("agent-skills/UPSTREAM.toml"),
        "skills = [\n  \"legacy-shared\",\n]\nthis is not valid TOML\n",
    );
    write(
        wrapper_path(&root, ".codex", "spec-sync"),
        "stale current wrapper\n",
    );

    let notices = scaffold::materialize_skills(&root).unwrap();

    for tool in [".claude", ".codex"] {
        assert_eq!(bytes(wrapper_path(&root, tool, "legacy-shared")), unknown);
    }
    assert_eq!(
        bytes(wrapper_path(&root, ".codex", "spec-sync")),
        bytes(skill_path(&root, "spec-sync"))
    );
    assert!(
        notices.iter().any(|notice| {
            notice.contains("UPSTREAM.toml")
                && notice.contains("malformed")
                && notice.contains("preserv")
        }),
        "malformed ownership data needs an actionable non-destructive notice: {notices:?}"
    );
}
