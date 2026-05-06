//! Rule registry — the central list of all lint rules, in dispatch order.

use crate::Rule;
use crate::rules;

/// Return every rule the lint enforces.
pub fn all_rules() -> Vec<Box<dyn Rule>> {
    vec![
        Box::new(rules::no_shell_scripts::NoShellScripts),
        Box::new(rules::phase_identifier_case::PhaseIdentifierCase),
    ]
}
