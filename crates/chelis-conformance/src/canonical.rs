//! Canonical bodies for the stamped managed blocks.
//!
//! The `agents-inheritance` body is generated from the root `AGENTS.md`; the
//! other bodies are authored here. `conform sync` renders each
//! into the corresponding shell document via [`crate::managed_block`], and
//! `conform audit` checks the shell's block against the body for the shell's
//! pinned version. Adding a block id means adding its `.md` under
//! `assets/canonical/` and an entry here.

/// `(block-id, canonical body)` embedded at compile time.
pub const CANONICAL: &[(&str, &str)] = &[
    (
        "agents-inheritance",
        include_str!("../assets/canonical/agents-inheritance.md"),
    ),
    (
        "chelis-surface-header",
        include_str!("../assets/canonical/chelis-surface-header.md"),
    ),
];

/// The canonical body for a managed-block id, if known.
pub fn body(id: &str) -> Option<&'static str> {
    CANONICAL.iter().find(|(k, _)| *k == id).map(|(_, v)| *v)
}
