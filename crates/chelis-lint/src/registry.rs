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
    ]
}
