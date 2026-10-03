//! Tripwire locking the embedded assets to the live repo files they copy.
//!
//! `scripts/regenerate_conformance_assets.py` copies `agent-skills/` into
//! `assets/skills/`, which `src/skills.rs` embeds via `include_str!`. This test
//! asserts the embedded bytes, both agent-surface symlinks, the `SHARED_SKILLS`
//! list, and the live `agent-skills/` directory agree. A forgotten re-run (or a hand-edited
//! embedded copy, or a skill added/removed upstream) fails here with a pointer
//! back at the regenerate script.

use std::collections::BTreeSet;
use std::path::PathBuf;

use chelis_conformance::skills::{EMBEDDED_SKILLS, PACKAGE_SKILLS, SHARED_SKILLS, source_path};

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

    // 2. The live agent-skills/ directory contains exactly the shared skills
    //    that are not authored beside a package.
    let skills_dir = root.join("agent-skills");
    let on_disk: BTreeSet<String> = std::fs::read_dir(&skills_dir)
        .unwrap_or_else(|e| panic!("read {skills_dir:?}: {e}"))
        .filter_map(|e| e.ok())
        .filter(|e| e.path().is_dir())
        .map(|e| e.file_name().to_string_lossy().to_string())
        .collect();
    let shared_owned: BTreeSet<String> = SHARED_SKILLS
        .iter()
        .filter(|s| !PACKAGE_SKILLS.iter().any(|(p, _)| p == *s))
        .map(|s| s.to_string())
        .collect();
    assert_eq!(
        on_disk, shared_owned,
        "agent-skills/ directory does not match SHARED_SKILLS. If the shared skill \
         set changed, update SHARED_SKILLS (src/skills.rs + build.rs + the regenerate \
         script) and run `{regen}`."
    );

    // 3. Every embedded body byte-equals its one authored source file.
    let mut stale = Vec::new();
    for (name, body) in EMBEDDED_SKILLS {
        let src = root.join(source_path(name));
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

    for surface in [".claude/skills", ".codex/skills"] {
        let path = root.join(surface);
        let metadata =
            std::fs::symlink_metadata(&path).unwrap_or_else(|e| panic!("inspect {path:?}: {e}"));
        assert!(
            metadata.file_type().is_symlink(),
            "{surface} is not a symlink"
        );
        assert_eq!(
            std::fs::read_link(&path).unwrap(),
            std::path::Path::new("../agent-skills"),
            "{surface} must point to the one authored skill tree"
        );
    }
}

/// chelis#2831: shells inherit the complete `docs/CHELIS_SURFACE.md`, so the
/// embedded `chelis-surface` body is a generated copy of that one authored
/// file, never a second hand-maintained description of the surface.
#[test]
fn embedded_surface_guide_matches_repo() {
    let root = repo_root();
    let path = root.join("docs/CHELIS_SURFACE.md");
    let live = std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("read {path:?}: {e}"));
    let embedded = chelis_conformance::canonical::body("chelis-surface")
        .expect("chelis-surface canonical body");
    assert_eq!(
        embedded, live,
        "the embedded chelis-surface body must be regenerated from docs/CHELIS_SURFACE.md; \
         run `python3 scripts/regenerate_conformance_assets.py`"
    );
}

#[test]
fn embedded_agents_contract_and_documented_skill_list_match_repo() {
    let root = repo_root();
    let agents_path = root.join("AGENTS.md");
    let agents = std::fs::read_to_string(&agents_path)
        .unwrap_or_else(|e| panic!("read {agents_path:?}: {e}"));
    let embedded = chelis_conformance::canonical::body("agents-inheritance")
        .expect("agents-inheritance canonical body");
    assert_eq!(
        embedded, agents,
        "the embedded agents-inheritance body must be regenerated from root AGENTS.md; \
         run `python3 scripts/regenerate_conformance_assets.py`"
    );

    let pointers = agents
        .split_once("## Pointers")
        .map(|(_, section)| section)
        .expect("AGENTS.md must carry its Pointers section");
    for skill in SHARED_SKILLS {
        assert!(
            pointers.contains(&format!("`{skill}`")),
            "AGENTS.md's shared-skill list omits repository skill {skill:?}"
        );
    }
}
