//! Phase 3 oracle: `conform sync` round-trip.
//!
//! Scaffold a shell (green), break every managed block (stale stamp + hand-edited
//! body), fork every skill, and drift both command trees, then `sync` and confirm
//! the audit is green again — AND that content outside the managed surfaces is
//! preserved byte-for-byte.

use std::path::Path;

use chelis_conformance::audit;
use chelis_conformance::scaffold;

// Must track the crate version: sync stamps at VER and the audit's
// canonical-body comparison only applies at AUDITOR_VERSION, so a hardcoded
// version quietly downgrades this round-trip to the stale-stamp path only.
const VER: &str = env!("CARGO_PKG_VERSION");

fn sentinel_line() -> &'static str {
    "SHELL-OWNED SENTINEL: do not let sync clobber me\n"
}

fn inject_sentinel(root: &Path) {
    // A line the shell owns, outside any managed fence (right after the H1).
    let agents = root.join("AGENTS.md");
    let text = std::fs::read_to_string(&agents).unwrap();
    let injected = text.replacen(
        "## Repo Identity\n",
        &format!("## Repo Identity\n\n{}", sentinel_line()),
        1,
    );
    std::fs::write(&agents, injected).unwrap();
}

#[test]
fn sync_restores_green_and_preserves_shell_content() {
    let tmp = tempfile::tempdir().unwrap();
    let root = tmp.path().join("shell");
    scaffold::scaffold(&root, "shell", "Shell", VER).expect("scaffold");
    inject_sentinel(&root);

    assert!(audit::audit(&root).ok(), "baseline must be green");

    // Break everything sync is responsible for:
    // 1. stale + hand-edited managed blocks in both docs
    for rel in ["AGENTS.md", "docs/CHELIS_SURFACE.md"] {
        let p = root.join(rel);
        let t = std::fs::read_to_string(&p).unwrap();
        let t = t.replace(&format!("chelis@{VER}"), "chelis@0.9.0"); // stale stamp
        let t = t.replace("upstream", "UPSTREAM-TAMPERED"); // body hand-edit
        std::fs::write(&p, t).unwrap();
    }
    // 2. fork every skill and independently drift both generated command trees.
    for name in chelis_conformance::skills::SHARED_SKILLS {
        let p = root.join(format!("agent-skills/{name}/SKILL.md"));
        let mut t = std::fs::read_to_string(&p).unwrap();
        t.push_str("\nforked\n");
        std::fs::write(&p, t).unwrap();

        let claude = root.join(format!(".claude/commands/{name}.md"));
        let mut command = std::fs::read_to_string(&claude).unwrap();
        command.push_str("\nstale Claude command\n");
        std::fs::write(claude, command).unwrap();
        std::fs::remove_file(root.join(format!(".codex/commands/{name}.md"))).unwrap();
    }

    let broken = audit::audit(&root);
    assert!(!broken.ok(), "tampered shell must fail the audit");

    // Sync fixes it.
    scaffold::materialize_skills(&root).expect("materialize skills");
    scaffold::sync_managed_blocks(&root, VER).expect("sync blocks");

    let fixed = audit::audit(&root);
    assert!(
        fixed.ok(),
        "sync must restore green; residual failures: {:?}",
        fixed
            .rows
            .iter()
            .filter(|r| r.verdict == audit::Verdict::Fail)
            .map(|r| (r.key, r.diagnostic.clone()))
            .collect::<Vec<_>>()
    );

    // Shell-owned content outside the fences survived.
    let agents = std::fs::read_to_string(root.join("AGENTS.md")).unwrap();
    assert!(
        agents.contains(sentinel_line().trim()),
        "sync must not touch content outside managed fences"
    );
    // Exactly one managed block per doc (no duplicate insertion).
    assert_eq!(agents.matches("BEGIN CHELIS MANAGED BLOCK").count(), 1);
}
