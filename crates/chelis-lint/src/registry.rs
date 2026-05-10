//! Rule registry — the central list of all lint rules, in dispatch order.

use crate::Rule;
use crate::rules;

/// Return every rule the lint enforces.
pub fn all_rules() -> Vec<Box<dyn Rule>> {
    vec![
        Box::new(rules::no_shell_scripts::NoShellScripts),
        Box::new(rules::phase_identifier_case::PhaseIdentifierCase),
        Box::new(rules::module_compound_titlecase::ModuleCompoundTitlecase),
        Box::new(rules::module_pascal_components::ModulePascalComponents),
        Box::new(rules::doc_filename_convention::DocFilenameConvention),
        Box::new(rules::snapshot_filename_pattern::SnapshotFilenamePattern),
        Box::new(rules::type_suffix_policy::TypeSuffixPolicy),
        Box::new(rules::prefix_namespace::PrefixNamespace),
        Box::new(rules::deep_user_symbol_charset::DeepUserSymbolCharset),
        Box::new(rules::surf_type_pascal_case::SurfTypePascalCase),
        Box::new(rules::surf_value_snake_case::SurfValueSnakeCase),
        Box::new(rules::surf_test_name_prefix::SurfTestNamePrefix),
        Box::new(rules::surf_def_arrow_form::SurfDefArrowForm),
    ]
}

/// Return warning-only rules. These are user-facing lint advisories, but they
/// are intentionally excluded from the blocking style-gate registry.
pub fn advisory_rules() -> Vec<Box<dyn Rule>> {
    vec![Box::new(
        rules::redundant_linearity_call::RedundantLinearityCall,
    )]
}
