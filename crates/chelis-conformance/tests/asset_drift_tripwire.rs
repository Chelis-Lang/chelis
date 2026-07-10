//! Tripwire locking the embedded assets to the live repo files they copy.
//!
//! `scripts/regenerate_conformance_assets.py` copies `agent-skills/` into
//! `assets/skills/`, which `src/skills.rs` embeds via `include_str!`. This test
//! asserts three things agree: the embedded bytes, the `SHARED_SKILLS` list, and
//! the live `agent-skills/` directory. A forgotten re-run (or a hand-edited
//! embedded copy, or a skill added/removed upstream) fails here with a pointer
//! back at the regenerate script — the same guarantee `chelis-std-bundle`'s
//! `archive_self_consistency` gives for the runtime bytes.

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
}
