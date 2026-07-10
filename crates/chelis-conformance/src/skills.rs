//! The shared agent-skill set, embedded into the toolchain.
//!
//! Contract §8 requires every shell to carry the monorepo's shared skills. In
//! the old model each shell committed a *forked copy* that drifted. Here the
//! canonical bytes are embedded in the binary (below); `conform sync`
//! materializes them into a shell's `agent-skills/` and `conform audit` checks
//! any present copy byte-for-byte against them (fork detection). The shell owns
//! zero skill content — the embedded set for the shell's pinned toolchain is the
//! source of truth.
//!
//! The embedded copies live under `assets/skills/`, regenerated from the live
//! `agent-skills/` by `scripts/regenerate_conformance_assets.py`. Three lists
//! must agree — [`SHARED_SKILLS`], the `assets/skills/` tree, and the live
//! `agent-skills/` dir — and `tests/asset_drift_tripwire.rs` locks all three.

/// The shared skill names, in the order the contract §8 lists them. Keep in
/// lockstep with `scripts/regenerate_conformance_assets.py::SHARED_SKILLS` and
/// the `agent-skills/` directory.
pub const SHARED_SKILLS: &[&str] = &[
    "backend-numerics",
    "cli-surface",
    "example-corpus",
    "issue-resolution",
    "packaging-install",
    "phase-gate",
    "redteam-exec",
    "spec-sync",
];

/// `(skill-name, SKILL.md contents)` embedded at compile time. One entry per
/// [`SHARED_SKILLS`] name; a new skill needs an entry here plus a re-embed.
pub const EMBEDDED_SKILLS: &[(&str, &str)] = &[
    (
        "backend-numerics",
        include_str!("../assets/skills/backend-numerics/SKILL.md"),
    ),
    (
        "cli-surface",
        include_str!("../assets/skills/cli-surface/SKILL.md"),
    ),
    (
        "example-corpus",
        include_str!("../assets/skills/example-corpus/SKILL.md"),
    ),
    (
        "issue-resolution",
        include_str!("../assets/skills/issue-resolution/SKILL.md"),
    ),
    (
        "packaging-install",
        include_str!("../assets/skills/packaging-install/SKILL.md"),
    ),
    (
        "phase-gate",
        include_str!("../assets/skills/phase-gate/SKILL.md"),
    ),
    (
        "redteam-exec",
        include_str!("../assets/skills/redteam-exec/SKILL.md"),
    ),
    (
        "spec-sync",
        include_str!("../assets/skills/spec-sync/SKILL.md"),
    ),
];

/// The embedded `SKILL.md` for `name`, or `None` if `name` is not a shared skill.
pub fn skill_body(name: &str) -> Option<&'static str> {
    EMBEDDED_SKILLS
        .iter()
        .find(|(n, _)| *n == name)
        .map(|(_, body)| *body)
}
