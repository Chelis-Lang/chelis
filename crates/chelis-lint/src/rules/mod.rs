//! Lint rule implementations, one module per rule.
//!
//! Each rule corresponds to a section of `chelis/spec/01-nomenclature.md`.
//! The mapping is recorded in [`crate::registry::all_rules`].

pub mod doc_filename_convention;
pub mod module_compound_titlecase;
pub mod module_decl;
pub mod module_pascal_components;
pub mod no_shell_scripts;
pub mod phase_identifier_case;
pub mod snapshot_filename_pattern;
