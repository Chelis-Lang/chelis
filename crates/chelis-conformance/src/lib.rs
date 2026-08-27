//! Downstream shell-repo conformance: the machine-readable form of
//! [`spec/design/shell_repo_contract.md`] and the tooling that enforces and
//! propagates it.
//!
//! This crate is the single source of truth that ships *inside the toolchain*
//! so it cannot copy-drift the way `school`'s hand-copied Python scripts did:
//!
//! - [`manifest`] — the §11 conformance table as data ([`manifest::MANIFEST`]).
//! - [`registry`] — the reconciled shell registry ([`registry::REGISTRY`]),
//!   ground-truthed against the `ecosystem-drift.yml` canary matrix.
//! - [`skills`] — the shared agent-skill set, embedded for `conform sync` to
//!   materialize and `conform audit` to fork-check.
//!
//! Later phases add the audit engine, the pointer managed-block mechanism, the
//! `conform sync`/`init` scaffolding, and the `conform bump`/`bump-check`
//! version-propagation surface. The `chelis reef conform` CLI verbs dispatch
//! into this crate.
//!
//! [`spec/design/shell_repo_contract.md`]: https://github.com/Chelis-Lang/chelis/blob/main/spec/design/shell_repo_contract.md

pub mod audit;
pub mod bump;
pub mod canonical;
pub mod conform;
pub mod expect;
pub mod managed_block;
pub mod manifest;
pub mod registry;
pub mod scaffold;
pub mod skills;
