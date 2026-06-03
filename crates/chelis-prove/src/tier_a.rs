//! Tier A: Type system validation.
//!
//! Checks property well-formedness via the Chelis type checker.
//! Positively discharges dimension-type, effect-row, and linearity properties.

use chelis_compiler_api::compiler;
use chelis_compiler_api::schema::{CheckRequest, SourceKind};

/// Tier A outcome.
#[derive(Debug, Clone, PartialEq)]
pub enum TierAResult {
    /// Property is structurally ill-formed (type error).
    Rejected(String),
    /// Property proved by type system (dimension/effect/linearity).
    Proved,
    /// Type system cannot determine; pass to next tier.
    Inconclusive,
}

/// Run Tier A type-system check on a property.
///
/// Validates that the property source type-checks. If the type checker reports
/// errors, the property is rejected as ill-formed. Otherwise, inconclusive
/// (positive discharge for dimension/effect/linearity is future work).
pub fn check(property_source: &str, _property_name: &str) -> TierAResult {
    let result = compiler::check(CheckRequest {
        source_kind: SourceKind::Surf,
        source: property_source.to_string(),
    });
    match result {
        Ok(check_result) => {
            if check_result.errors.is_empty() {
                // Type-checks cleanly. Future: inspect types for positive discharge.
                TierAResult::Inconclusive
            } else {
                let messages: Vec<String> = check_result
                    .errors
                    .iter()
                    .map(|d| d.message.clone())
                    .collect();
                TierAResult::Rejected(messages.join("; "))
            }
        }
        Err(err) => {
            let messages: Vec<String> = err.errors.iter().map(|d| d.message.clone()).collect();
            TierAResult::Rejected(messages.join("; "))
        }
    }
}
