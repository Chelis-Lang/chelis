//! Rule registry — the central list of all lint rules, in dispatch order.

use crate::Rule;
use crate::rules;

/// Return every rule the lint enforces.
pub fn all_rules() -> Vec<Box<dyn Rule>> {
    vec![
        Box::new(rules::no_shell_scripts::NoShellScripts),
        Box::new(rules::phase_identifier_case::PhaseIdentifierCase),
        Box::new(rules::module_compound_titlecase::ModuleCompoundTitlecase),
        Box::new(rules::doc_filename_convention::DocFilenameConvention),
        Box::new(rules::snapshot_filename_pattern::SnapshotFilenamePattern),
        Box::new(rules::type_suffix_policy::TypeSuffixPolicy),
        Box::new(rules::prefix_namespace::PrefixNamespace),
        Box::new(rules::deep_user_symbol_charset::DeepUserSymbolCharset),
        Box::new(rules::surf_type_pascal_case::SurfTypePascalCase),
        Box::new(rules::surf_value_snake_case::SurfValueSnakeCase),
        Box::new(rules::surf_test_name_prefix::SurfTestNamePrefix),
        Box::new(rules::surf_def_arrow_form::SurfDefArrowForm),
        Box::new(rules::no_em_dash_in_public_strings::NoEmDashInPublicStrings),
        Box::new(rules::opaque_domain_construction::OpaqueDomainConstruction),
    ]
}

/// Return non-blocking user-facing rules. They run in the standalone lint
/// command and selected user-facing warnings, but are excluded from the
/// blocking style gate.
pub fn non_blocking_rules() -> Vec<Box<dyn Rule>> {
    vec![
        Box::new(rules::redundant_linearity_call::RedundantLinearityCall),
        Box::new(rules::prefer_pipe_operator::PreferPipeOperator),
        Box::new(rules::opaque_without_invariant::OpaqueWithoutInvariant),
        Box::new(rules::invariant_float_equality::InvariantFloatEquality),
        Box::new(rules::unreachable_producer::UnreachableProducer),
        Box::new(rules::opaque_escape_site::OpaqueEscapeSite),
    ]
}

/// Backward-compatible name used by older call sites.
pub fn advisory_rules() -> Vec<Box<dyn Rule>> {
    non_blocking_rules()
}

pub fn selectable_rules() -> Vec<Box<dyn Rule>> {
    let mut rules = all_rules();
    rules.extend(non_blocking_rules());
    rules
}
