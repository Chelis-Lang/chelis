//! The single machine-readable registry of downstream shell repos.
//!
//! Before this table there were two hand-maintained shell lists that had
//! already drifted apart: the `chelis_canonical_reference.md` §Shell-Ecosystem
//! prose table and the `.github/workflows/ecosystem-drift.yml` canary matrix.
//! [`REGISTRY`] is the reconciled source of truth. Per the user decision, the
//! **canary matrix is ground truth** for the active set: `registry_tripwire`'s
//! `registry_matches_drift_matrix` asserts [`active_shells`] equals the canary's
//! `- repo:` legs, so adding a shell to the canary without adding it here (or
//! vice versa) fails the build.
//!
//! Future stubs (darwin, hydrostatic, beacon) are added with
//! [`ShellStatus::Future`] during the canonical-reference reconciliation; the
//! canary tripwire only ranges over [`ShellStatus::Active`] entries, so a stub
//! that has no canary leg does not break it.

/// How a shell consumes chelis, which determines the canary's build recipe.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellKind {
    /// Pure-Chelis: consumes the toolchain binary + chelis-std via reef.
    Reef,
    /// Rust workspace; may link chelis crates as Cargo path deps.
    Cargo,
    /// Ships/tests inside a Docker image built from a release tarball.
    Docker,
}

/// Lifecycle state. Only [`Active`](ShellStatus::Active) shells appear in the
/// canary matrix and are audited by the push canary.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShellStatus {
    /// A live repo with CI; in the ecosystem-drift matrix.
    Active,
    /// Scaffolded but not yet a full consumer.
    Stub,
    /// Named in the canonical reference but not yet a repo.
    Future,
}

/// One downstream shell.
#[derive(Debug, Clone, Copy)]
pub struct Shell {
    /// Repo name under `Chelis-Lang/` (also the canary matrix `repo:` value).
    pub name: &'static str,
    pub kind: ShellKind,
    pub status: ShellStatus,
    /// True iff the shell links chelis compiler crates as Cargo path deps,
    /// which triggers the contract's row-18 source-crate MUST.
    pub links_chelis_crates: bool,
    /// Other shells this one depends on via reef (by name). Mirrors the canary
    /// matrix `reef_deps`.
    pub deps: &'static [&'static str],
}

/// Every shell. Active entries mirror the `ecosystem-drift.yml` matrix exactly.
pub const REGISTRY: &[Shell] = &[
    // ---- Pure-Chelis / reef shells (toolchain-binary consumers).
    Shell {
        name: "nautilus",
        kind: ShellKind::Reef,
        status: ShellStatus::Active,
        links_chelis_crates: false,
        deps: &[],
    },
    Shell {
        name: "coral",
        kind: ShellKind::Reef,
        status: ShellStatus::Active,
        links_chelis_crates: false,
        deps: &["nautilus"],
    },
    Shell {
        name: "shoals",
        kind: ShellKind::Reef,
        status: ShellStatus::Active,
        links_chelis_crates: false,
        deps: &["nautilus", "coral"],
    },
    Shell {
        name: "school",
        kind: ShellKind::Reef,
        status: ShellStatus::Active,
        links_chelis_crates: false,
        deps: &[],
    },
    Shell {
        name: "hull",
        kind: ShellKind::Reef,
        status: ShellStatus::Active,
        links_chelis_crates: false,
        deps: &[],
    },
    Shell {
        name: "whale",
        kind: ShellKind::Reef,
        status: ShellStatus::Active,
        links_chelis_crates: false,
        deps: &["nautilus", "coral", "shoals"],
    },
    // ---- Cargo shells (Rust workspaces; some link chelis crates as path deps).
    Shell {
        name: "octant",
        kind: ShellKind::Cargo,
        status: ShellStatus::Active,
        links_chelis_crates: true,
        deps: &["nautilus"],
    },
    Shell {
        name: "calcify",
        kind: ShellKind::Cargo,
        status: ShellStatus::Active,
        links_chelis_crates: true,
        deps: &["nautilus", "coral"],
    },
    Shell {
        name: "c-earchin",
        kind: ShellKind::Cargo,
        status: ShellStatus::Active,
        links_chelis_crates: false,
        deps: &[],
    },
    Shell {
        name: "hydronnx",
        kind: ShellKind::Cargo,
        status: ShellStatus::Active,
        links_chelis_crates: true,
        deps: &[],
    },
    // ---- Docker shell.
    Shell {
        name: "hello-chelis",
        kind: ShellKind::Docker,
        status: ShellStatus::Active,
        links_chelis_crates: false,
        deps: &[],
    },
];

/// The active shells, in registry order. This is the set the canary matrix
/// ranges over.
pub fn active_shells() -> impl Iterator<Item = &'static Shell> {
    REGISTRY.iter().filter(|s| s.status == ShellStatus::Active)
}

/// Look up a shell by name.
pub fn shell(name: &str) -> Option<&'static Shell> {
    REGISTRY.iter().find(|s| s.name == name)
}
