//! Lint rule implementations, one module per rule.
//!
//! Each rule corresponds to a section of `chelis/spec/01-nomenclature.md`.
//! The mapping is recorded in [`crate::registry::all_rules`].

pub mod no_shell_scripts;
pub mod phase_identifier_case;
