//! Canonical bodies for the stamped managed blocks.
//!
//! Every body is generated from an authored document in the chelis repository:
//! `agents-inheritance` from the root `AGENTS.md`, and `chelis-surface` from
//! `docs/CHELIS_SURFACE.md`. `scripts/regenerate_conformance_assets.py` copies
//! each source to its asset under `assets/canonical/`, and
//! `tests/asset_drift_tripwire.rs` fails when a copy differs from its source.
//! `conform sync` renders each body into the corresponding shell document via
//! [`crate::managed_block`], and `conform audit` checks the shell's block
//! against the body for the shell's pinned version. Adding a block id means
//! adding its source to the regeneration script and an entry here.

/// `(block-id, canonical body)` embedded at compile time.
pub const CANONICAL: &[(&str, &str)] = &[
    (
        "agents-inheritance",
        include_str!("../assets/canonical/agents-inheritance.md"),
    ),
    (
        "chelis-surface",
        include_str!("../assets/canonical/chelis-surface.md"),
    ),
];

/// The block id an older toolchain stamped into `docs/CHELIS_SURFACE.md` for a
/// short pointer header, before the block carried the complete surface guide.
/// `conform sync` replaces a block with this id in place by the `chelis-surface`
/// block, so the superseded header does not linger with a stale stamp.
pub const LEGACY_SURFACE_HEADER: &str = "chelis-surface-header";

/// The repository path of the document a block id inherits, which is also its
/// path in the shell. Inherited links resolve against it (see [`crate::links`]).
pub fn document_path(id: &str) -> Option<&'static str> {
    match id {
        "agents-inheritance" => Some("AGENTS.md"),
        "chelis-surface" => Some("docs/CHELIS_SURFACE.md"),
        _ => None,
    }
}

/// The canonical body for a managed-block id, if known.
pub fn body(id: &str) -> Option<&'static str> {
    CANONICAL.iter().find(|(k, _)| *k == id).map(|(_, v)| *v)
}
