//! Lint rule implementations, one module per rule.
//!
//! Each rule corresponds to a section of `chelis/spec/01-nomenclature.md`.
//! The mapping is recorded in [`crate::registry::all_rules`].

pub mod deep_user_symbol_charset;
pub mod doc_filename_convention;
pub mod module_compound_titlecase;
pub mod module_decl;
pub mod no_em_dash_in_public_strings;
pub mod no_shell_scripts;
pub mod opaque_domain_construction;
pub mod phase_identifier_case;
pub mod prefer_pipe_operator;
pub mod prefix_namespace;
pub mod redundant_linearity_call;
pub mod snapshot_filename_pattern;
pub mod surf_def_arrow_form;
pub mod surf_test_name_prefix;
pub mod surf_type_pascal_case;
pub mod surf_value_snake_case;
pub mod type_suffix_policy;
