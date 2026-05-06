//! Rule registry — the central list of all lint rules, in dispatch order.

use crate::Rule;

/// Return every rule the lint enforces. Rules are added here as their modules
/// land in subsequent commits.
pub fn all_rules() -> Vec<Box<dyn Rule>> {
    Vec::new()
}
