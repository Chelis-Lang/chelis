//! Inherited text keeps working links in a shell (contract §1, §3, §8).
//!
//! The inherited documents are authored in the chelis repository, so their
//! repo-relative links name chelis files. Sync keeps a link relative when its
//! target is materialized into the shell too, and pins every other one to the
//! chelis release the shell pins. These tests read what a scaffolded and synced
//! shell actually contains.

use std::path::{Path, PathBuf};

use chelis_conformance::{audit, managed_block, scaffold};
use pulldown_cmark::{Event, Parser, Tag};

const VER: &str = env!("CARGO_PKG_VERSION");

fn shell() -> (tempfile::TempDir, PathBuf) {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("shell");
    scaffold::scaffold(&root, "shell", "Shell", VER).expect("scaffold");
    (tmp, root)
}

fn pinned(path: &str) -> String {
    format!("https://github.com/Chelis-Lang/chelis/blob/v{VER}/{path}")
}

fn block(root: &Path, file: &str, id: &str) -> String {
    let text = std::fs::read_to_string(root.join(file)).unwrap();
    managed_block::find(&text, id).unwrap().body
}

/// Every Markdown link destination in `text`.
fn destinations(text: &str) -> Vec<String> {
    Parser::new(text)
        .filter_map(|event| match event {
            Event::Start(Tag::Link { dest_url, .. })
            | Event::Start(Tag::Image { dest_url, .. }) => Some(dest_url.to_string()),
            _ => None,
        })
        .collect()
}

#[test]
fn a_link_without_a_local_copy_is_pinned_to_the_shell_release() {
    let (_tmp, root) = shell();
    let agents = block(&root, "AGENTS.md", "agents-inheritance");
    let links = destinations(&agents);
    assert!(links.contains(&pinned("docs/local_gate.md")), "{links:?}");
    assert!(
        !links.iter().any(|l| l == "docs/local_gate.md"),
        "{links:?}"
    );
}

#[test]
fn a_link_to_a_materialized_target_stays_relative() {
    let (_tmp, root) = shell();
    let agents = block(&root, "AGENTS.md", "agents-inheritance");
    let links = destinations(&agents);
    assert!(
        links
            .iter()
            .any(|l| l == "agent-skills/redteam-exec/SKILL.md"),
        "{links:?}"
    );
}

#[test]
fn a_parent_link_resolves_against_the_source_directory_and_keeps_its_anchor() {
    let (_tmp, root) = shell();
    let surface = block(&root, "docs/CHELIS_SURFACE.md", "chelis-surface");
    let links = destinations(&surface);
    // `../spec/...` inside docs/CHELIS_SURFACE.md names the repo's spec/.
    assert!(
        links.contains(&pinned("spec/04-type-system.md")),
        "{links:?}"
    );
    // `manual_gates.md#...` resolves into docs/ and keeps its anchor.
    assert!(
        links
            .iter()
            .any(|l| l.starts_with(&pinned("docs/manual_gates.md#"))),
        "{links:?}"
    );
    // No `../spec/` link is left relative: the shell has no spec/ tree.
    assert!(
        !links.iter().any(|l| l.starts_with("../spec/")),
        "{links:?}"
    );
}

#[test]
fn absolute_urls_and_in_page_anchors_are_untouched() {
    let (_tmp, root) = shell();
    let agents = block(&root, "AGENTS.md", "agents-inheritance");
    let links = destinations(&agents);
    assert!(links.iter().any(|l| l == "#issue-tracking"), "{links:?}");
    assert!(
        links
            .iter()
            .any(|l| l == "https://github.com/Chelis-Lang/openspec"),
        "{links:?}"
    );
}

/// The invariant behind the four cases above, over every inherited file sync
/// writes: each link is absolute, an in-page anchor, or resolves to a file that
/// exists in the shell. A relative link to a chelis-only file would be dead.
#[test]
fn every_inherited_link_resolves_in_the_shell() {
    let (_tmp, root) = shell();
    let mut files = vec![
        (
            "AGENTS.md".to_string(),
            block(&root, "AGENTS.md", "agents-inheritance"),
        ),
        (
            "docs/CHELIS_SURFACE.md".to_string(),
            block(&root, "docs/CHELIS_SURFACE.md", "chelis-surface"),
        ),
    ];
    for name in chelis_conformance::skills::SHARED_SKILLS {
        let rel = format!("agent-skills/{name}/SKILL.md");
        files.push((
            rel.clone(),
            std::fs::read_to_string(root.join(&rel)).unwrap(),
        ));
    }
    let mut dead = Vec::new();
    for (file, text) in &files {
        let dir = root.join(file).parent().unwrap().to_path_buf();
        for link in destinations(text) {
            if link.starts_with('#') || link.contains("://") || link.starts_with("mailto:") {
                continue;
            }
            let path = link.split('#').next().unwrap();
            if !dir.join(path).exists() {
                dead.push(format!("{file}: {link}"));
            }
        }
    }
    assert!(dead.is_empty(), "dead inherited links: {dead:#?}");
}

#[test]
fn audit_stays_clean_after_sync_restamps_pinned_links() {
    let (_tmp, root) = shell();
    assert!(audit::audit(&root).ok(), "scaffold must audit green");
    scaffold::sync_managed_blocks(&root, VER).expect("sync");
    let report = audit::audit(&root);
    let fails: Vec<_> = report
        .rows
        .iter()
        .filter(|r| r.verdict == audit::Verdict::Fail)
        .map(|r| format!("{}: {}", r.key, r.diagnostic))
        .collect();
    assert!(fails.is_empty(), "{fails:?}");

    // The raw, unpinned text in the block is now drift.
    let path = root.join("AGENTS.md");
    let text = std::fs::read_to_string(&path).unwrap();
    let found = managed_block::find(&text, "agents-inheritance").unwrap();
    let raw = managed_block::render(
        "agents-inheritance",
        &found.version,
        chelis_conformance::canonical::body("agents-inheritance").unwrap(),
    );
    std::fs::write(
        &path,
        format!("{}{raw}{}", &text[..found.span.0], &text[found.span.1..]),
    )
    .unwrap();
    let report = audit::audit(&root);
    let row = report.rows.iter().find(|r| r.key == "agents-md").unwrap();
    assert_eq!(row.verdict, audit::Verdict::Fail, "{}", row.diagnostic);
}

/// Shared skills carry the same link rewrite, applied by one function on both
/// the sync and the audit side. The chelis-std skill's absolute documentation
/// URLs survive unchanged. A fixture exercises relative links from that same
/// package location, independently of which links its prose currently carries.
/// Audit accepts the materialized skill and rejects a changed destination.
#[test]
fn shared_skill_links_are_pinned_by_sync_and_expected_by_audit() {
    let (_tmp, root) = shell();
    let path = root.join("agent-skills/chelis-std/SKILL.md");
    let skill = std::fs::read_to_string(&path).unwrap();
    let links = destinations(&skill);
    let raw = chelis_conformance::skills::skill_body("chelis-std").unwrap();
    assert_eq!(skill, raw, "absolute URLs must survive materialization");
    assert!(
        links.contains(&"https://chelis.ch/docs/chelis/reef/".to_string()),
        "{links:?}"
    );
    let fixture = format!(
        "{raw}\n[contract](../../spec/design/shell_repo_contract.md)\n[surface](../../docs/CHELIS_SURFACE.md)\n"
    );
    let rewritten = chelis_conformance::links::pin_links(
        &fixture,
        &chelis_conformance::skills::source_path("chelis-std"),
        "agent-skills/chelis-std/SKILL.md",
        VER,
        &chelis_conformance::links::LocalTargets::for_shell(&[]),
    );
    let rewritten_links = destinations(&rewritten);
    assert!(
        rewritten_links.contains(&pinned("spec/design/shell_repo_contract.md")),
        "{rewritten_links:?}"
    );
    assert!(
        rewritten_links
            .iter()
            .any(|l| l == "../../docs/CHELIS_SURFACE.md"),
        "{rewritten_links:?}"
    );

    scaffold::materialize_skills(&root, VER).expect("sync skills");
    assert_eq!(std::fs::read_to_string(&path).unwrap(), skill);
    let vendored = |root: &Path| {
        let report = audit::audit(root);
        report
            .rows
            .iter()
            .find(|r| r.key == "vendored-skills")
            .unwrap()
            .clone()
    };
    let row = vendored(&root);
    assert_eq!(row.verdict, audit::Verdict::Pass, "{}", row.diagnostic);

    let changed = skill.replace(
        "https://chelis.ch/docs/chelis/reef/",
        "https://example.invalid/reef/",
    );
    assert_ne!(
        changed, skill,
        "the negative control must change a destination"
    );
    std::fs::write(&path, changed).unwrap();
    let row = vendored(&root);
    assert_eq!(row.verdict, audit::Verdict::Fail);
    assert!(
        row.diagnostic.contains("chelis-std: forked/stale"),
        "{}",
        row.diagnostic
    );
}
