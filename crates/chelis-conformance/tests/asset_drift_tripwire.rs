//! Tripwire locking the embedded assets to the live repo files they copy.
//!
//! `scripts/regenerate_conformance_assets.py` copies `agent-skills/` into
//! `assets/skills/`, which `src/skills.rs` embeds via `include_str!`, and projects
//! the same bytes to both monorepo command trees. This test asserts the embedded
//! bytes, `SHARED_SKILLS`, live skill files, and same-name command files agree. A
//! forgotten re-run (or a hand edit) fails with a pointer back at the generator.

use std::collections::BTreeSet;
use std::path::PathBuf;

use chelis_conformance::skills::{EMBEDDED_SKILLS, SHARED_SKILLS};

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .expect("repo root must exist")
}

#[test]
fn embedded_skills_match_repo() {
    let root = repo_root();
    let regen = "python3 scripts/regenerate_conformance_assets.py";

    // 1. The embedded list and the SHARED_SKILLS list name the same skills.
    let embedded_names: BTreeSet<&str> = EMBEDDED_SKILLS.iter().map(|(n, _)| *n).collect();
    let shared_names: BTreeSet<&str> = SHARED_SKILLS.iter().copied().collect();
    assert_eq!(
        embedded_names, shared_names,
        "EMBEDDED_SKILLS and SHARED_SKILLS disagree in crates/chelis-conformance/src/skills.rs"
    );

    // 2. The live agent-skills/ directory contains exactly SHARED_SKILLS.
    let skills_dir = root.join("agent-skills");
    let on_disk: BTreeSet<String> = std::fs::read_dir(&skills_dir)
        .unwrap_or_else(|e| panic!("read {skills_dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    let shared_owned: BTreeSet<String> = SHARED_SKILLS.iter().map(|s| s.to_string()).collect();
    assert_eq!(
        on_disk, shared_owned,
        "agent-skills/ directory does not match SHARED_SKILLS. If the shared skill \
         set changed, update SHARED_SKILLS (src/skills.rs + build.rs + the regenerate \
         script) and run `{regen}`."
    );

    // 3. Every embedded body byte-equals the live source file.
    let mut stale = Vec::new();
    for (name, body) in EMBEDDED_SKILLS {
        let src = root.join("agent-skills").join(name).join("SKILL.md");
        let live = std::fs::read_to_string(&src).unwrap_or_else(|e| panic!("read {src:?}: {e}"));
        if live != *body {
            stale.push(name.to_string());
        }
    }
    assert!(
        stale.is_empty(),
        "embedded skill copy is stale for {stale:?}. Run `{regen}` and commit \
         crates/chelis-conformance/assets/."
    );

    // 4. The monorepo's same-name command projections equal the live source.
    let mut command_drift = Vec::new();
    for name in SHARED_SKILLS {
        let source = std::fs::read(root.join("agent-skills").join(name).join("SKILL.md"))
            .unwrap_or_else(|error| panic!("read live skill {name}: {error}"));
        for tool in [".claude", ".codex"] {
            let command = root.join(tool).join("commands").join(format!("{name}.md"));
            match std::fs::read(&command) {
                Ok(bytes) if bytes == source => {}
                Ok(_) => command_drift.push(format!("{} differs", command.display())),
                Err(error) => command_drift.push(format!("{}: {error}", command.display())),
            }
        }
    }
    assert!(
        command_drift.is_empty(),
        "same-name command projections are stale: {command_drift:?}. Run `{regen}`"
    );
}
