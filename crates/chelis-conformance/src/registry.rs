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
    /// Sibling shells this one needs installed, by name: the **transitive
    /// closure** of its `reef.toml` `[dependencies]` over sibling shells, not
    /// just its direct edges. `chelis-std` is never listed (it ships with the
    /// toolchain, not as a shell). Transitive, because that is what a consumer
    /// has to act on: `reef install --from-github` resolves one release artifact
    /// at a time, so whale needs nautilus and coral on disk even though only
    /// shoals appears in its own manifest. The same closure is what a cascade
    /// wave walks to order the bumps, which is why an empty entry for a leaf
    /// shell is a real defect and not a cosmetic one (chelis#1259: hello-chelis
    /// recorded `&[]` while sitting at the bottom of the graph).
    ///
    /// Mirrors the canary matrix `reef_deps`, and
    /// `manifest_tripwire::registry_deps_match_drift_matrix` asserts the two
    /// agree name-for-name, so a one-sided edit fails the build.
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
    // hello-chelis is the LEAF of the ecosystem dependency graph: its reef.toml
    // depends on coral, nautilus, and school, so a cascade wave cannot bump it
    // until all three have published at the new pin. It recorded `&[]` until
    // chelis#1259. (Its Dockerfile additionally `reef install`s octant and
    // c-earchin as tool packages; those are not reef.toml `[dependencies]` of the
    // hello-chelis package, so they are not `deps` edges. The canary leg's
    // `token_repos` covers them.)
    Shell {
        name: "hello-chelis",
        kind: ShellKind::Docker,
        status: ShellStatus::Active,
        links_chelis_crates: false,
        deps: &["coral", "nautilus", "school"],
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

/// The upstream monorepo's repo name. Not a shell, so it is not in
/// [`REGISTRY`], but it is the primary target of a §4 citation.
pub const UPSTREAM_REPO: &str = "chelis";

/// Whether `name` is a repo a contract §4 citation may name: the upstream
/// monorepo, or any shell in [`REGISTRY`] (chelis#1270).
///
/// Registry membership is the whole point. §4 exists so that every workaround
/// cites something live and checkable, and a sibling shell's issue is exactly as
/// live and checkable as an upstream one. What the rule keeps out is a reference
/// nobody can resolve without guessing: a bare `#NNN` (which tracker?) or an
/// arbitrary repo name. Before this, a cascade wave forced shells to invent a
/// local file whose only content was a pointer at a sibling PR, which satisfied
/// the grammar while defeating it.
pub fn is_citable_repo(name: &str) -> bool {
    name == UPSTREAM_REPO || REGISTRY.iter().any(|s| s.name == name)
}

/// Every citable repo name, upstream first then the registry in order. Used by
/// diagnostics that need to name the accepted set.
pub fn citable_repos() -> impl Iterator<Item = &'static str> {
    std::iter::once(UPSTREAM_REPO).chain(REGISTRY.iter().map(|s| s.name))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn upstream_and_every_shell_are_citable() {
        assert!(is_citable_repo("chelis"));
        for s in REGISTRY {
            assert!(is_citable_repo(s.name), "{} must be citable", s.name);
        }
    }

    #[test]
    fn a_repo_outside_the_registry_is_not_citable() {
        // Negative parity for chelis#1270: the grammar widened to registry
        // siblings, NOT to any repo-shaped token.
        for name in ["torch", "numpy", "Chelis-Lang", "shell", "", "chelis-std"] {
            assert!(!is_citable_repo(name), "{name:?} must not be citable");
        }
    }

    #[test]
    fn hello_chelis_is_the_graph_leaf() {
        // chelis#1259: the entry that was recorded empty. Locked so a future
        // edit cannot silently re-empty it.
        let hc = shell("hello-chelis").expect("hello-chelis is registered");
        assert_eq!(hc.deps, &["coral", "nautilus", "school"]);
    }

    #[test]
    fn every_dep_names_a_registered_shell() {
        // A dep edge that names nothing in the registry is a typo the canary
        // would carry into a token scope.
        for s in REGISTRY {
            for d in s.deps {
                assert!(
                    shell(d).is_some(),
                    "{}: dep {d:?} is not registered",
                    s.name
                );
                assert_ne!(*d, s.name, "{}: a shell cannot depend on itself", s.name);
            }
        }
    }

    #[test]
    fn deps_are_transitively_closed() {
        // `deps` is the install closure, not the direct edges (see the field
        // doc): whale lists nautilus + coral because shoals pulls them in.
        for s in REGISTRY {
            for d in s.deps {
                let Some(dep) = shell(d) else { continue };
                for t in dep.deps {
                    assert!(
                        s.deps.contains(t),
                        "{}: deps must be transitively closed; {d} needs {t}",
                        s.name
                    );
                }
            }
        }
    }
}
