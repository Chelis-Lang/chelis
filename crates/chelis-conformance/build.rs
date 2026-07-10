//! Compile-time guard for the embedded conformance assets.
//!
//! `src/skills.rs` uses `include_str!()` against fixed paths under
//! `assets/skills/` — that is what bakes the bytes into the chelis binary. This
//! build script does not produce or copy assets; it only:
//!
//! 1. asserts the expected asset files exist before `include_str!` runs (a
//!    missing file there surfaces as a less-readable macro-site error), and
//! 2. emits `cargo:rerun-if-changed=assets/...` so cargo invalidates this crate
//!    (and every crate depending on it) when the embedded copy changes on disk.
//!
//! Asset regeneration is `scripts/regenerate_conformance_assets.py` (repo root),
//! which copies the live `agent-skills/` into `assets/`. The build.rs
//! intentionally does not shell out — the assets live in the repo as committed
//! files, exactly like `chelis-std-bundle`'s `dist/`.

use std::path::PathBuf;

/// Keep in lockstep with `chelis_conformance::skills::SHARED_SKILLS` and
/// `scripts/regenerate_conformance_assets.py::SHARED_SKILLS`. A separate
/// compilation unit cannot import the crate's const, so the list is mirrored;
/// the `embedded_skills_match_repo` tripwire is what actually enforces the
/// three-way agreement.
const SHARED_SKILLS: &[&str] = &[
    "backend-numerics",
    "cli-surface",
    "example-corpus",
    "issue-resolution",
    "packaging-install",
    "phase-gate",
    "redteam-exec",
    "spec-sync",
];

fn main() {
    let manifest_dir = PathBuf::from(
        std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR is always set by cargo"),
    );
    let skills = manifest_dir.join("assets").join("skills");

    for skill in SHARED_SKILLS {
        let path = skills.join(skill).join("SKILL.md");
        if !path.exists() {
            panic!(
                "conformance asset missing at {}.\n\
                 Run `python3 scripts/regenerate_conformance_assets.py` from the\n\
                 repository root to rebuild crates/chelis-conformance/assets/ from\n\
                 the live agent-skills/, then commit them.",
                path.display()
            );
        }
        println!("cargo:rerun-if-changed={}", path.display());
    }

    // Managed-block canonical bodies are authored directly under
    // assets/canonical/ (not mirrored from another repo file). Guard existence
    // so a missing body surfaces here rather than at the include_str! site.
    let canonical = manifest_dir.join("assets").join("canonical");
    for id in ["agents-inheritance", "chelis-surface-header"] {
        let path = canonical.join(format!("{id}.md"));
        if !path.exists() {
            panic!("conformance canonical body missing at {}", path.display());
        }
        println!("cargo:rerun-if-changed={}", path.display());
    }

    println!("cargo:rerun-if-changed=build.rs");
}
