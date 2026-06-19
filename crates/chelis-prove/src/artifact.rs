//! Proof artifact types representing verification outcomes.

use serde::{Deserialize, Serialize};

use crate::composition::{AssumptionRecord, CompositeVerdict};
use crate::obligations::ObligationMeta;

/// Which tier produced the verification result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ProofTier {
    /// Type system (dimension, effect, linearity).
    TypeSystem,
    /// SMT solver (cvc5).
    Smt,
    /// Randomized fuzz testing.
    Fuzz,
}

/// Verification outcome.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ProofStatus {
    /// Property formally proved (Tier A type-cert or Tier B smt-cert).
    Proved,
    /// Property formally disproved with counterexample.
    Disproved { counterexample: serde_json::Value },
    /// Property statistically validated (Tier C, N samples passed).
    StatisticallyValidated { samples: usize },
    /// The requested tier could not establish a verdict. This is distinct
    /// from `Disproved`, which requires a counterexample.
    Unsupported { reason: String },
    /// Property rejected as structurally ill-formed.
    Rejected { reason: String },
    /// Tier B was not amenable for this property.
    NotAmenable { reason: String },
}

/// Complete proof artifact for a single property.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ProofArtifact {
    pub property_name: String,
    pub tier: ProofTier,
    pub status: ProofStatus,
    pub duration_ms: u64,
    /// SMT-specific status (None if tier != Smt).
    pub smt_status: Option<SmtStatus>,
    /// Derived-obligation metadata (RFC D-OBLIG). `None` for ordinary
    /// user properties; `Some` when this artifact is an
    /// invariant-producer obligation. Serde-additive: `skip_serializing_if`
    /// keeps existing user-property artifacts byte-identical, and
    /// `#[serde(default)]` lets old payloads deserialize.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub obligation: Option<ObligationMeta>,
    /// Assumptions this artifact depends on, each with a discharge record.
    #[serde(default)]
    pub assumptions: Vec<AssumptionRecord>,
    /// Weakest-link verdict after composing the artifact with assumptions.
    #[serde(default)]
    pub composite_verdict: CompositeVerdict,
}

impl ProofArtifact {
    pub fn new(
        property_name: impl Into<String>,
        tier: ProofTier,
        status: ProofStatus,
        duration_ms: u64,
        smt_status: Option<SmtStatus>,
    ) -> Self {
        let composite_verdict = match &status {
            ProofStatus::Proved => CompositeVerdict::Proven,
            ProofStatus::StatisticallyValidated { samples } if *samples > 0 => {
                CompositeVerdict::ProvenModuloFuzzValidatedContract
            }
            ProofStatus::StatisticallyValidated { .. } => CompositeVerdict::Unsupported,
            ProofStatus::Disproved { .. } => CompositeVerdict::Failed,
            ProofStatus::Rejected { .. }
            | ProofStatus::NotAmenable { .. }
            | ProofStatus::Unsupported { .. } => CompositeVerdict::Unsupported,
        };
        Self {
            property_name: property_name.into(),
            tier,
            status,
            duration_ms,
            smt_status,
            obligation: None,
            assumptions: Vec::new(),
            composite_verdict,
        }
    }
}

/// SMT solver outcome detail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SmtStatus {
    Proved,
    Disproved,
    Timeout,
    Unknown,
    NotAmenable,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn base() -> ProofArtifact {
        ProofArtifact::new(
            "p",
            ProofTier::Smt,
            ProofStatus::Proved,
            1,
            Some(SmtStatus::Proved),
        )
    }

    #[test]
    fn artifact_without_obligation_omits_the_field() {
        let json = serde_json::to_string(&base()).unwrap();
        assert!(
            !json.contains("obligation"),
            "obligation:None is skipped: {json}"
        );
        // Round-trips.
        let back: ProofArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(back, base());
    }

    #[test]
    fn artifact_with_obligation_round_trips() {
        let mut a = base();
        a.obligation = Some(ObligationMeta {
            obligation_kind: "invariant_producer".to_string(),
            source_type: "Probability".to_string(),
            producer: "probability".to_string(),
        });
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains("invariant_producer"));
        let back: ProofArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn old_payload_without_obligation_field_deserializes() {
        // A pre-feature payload has no `obligation` key; serde default
        // must fill it with None.
        let old = r#"{"property_name":"p","tier":"Smt","status":"Proved","duration_ms":1,"smt_status":"Proved"}"#;
        let back: ProofArtifact = serde_json::from_str(old).unwrap();
        assert_eq!(back.obligation, None);
        assert!(back.assumptions.is_empty());
        assert_eq!(back.composite_verdict, CompositeVerdict::Proven);
    }

    #[test]
    fn unsupported_artifact_is_not_a_green_composite() {
        let artifact = ProofArtifact::new(
            "p",
            ProofTier::Smt,
            ProofStatus::Unsupported {
                reason: "cvc5 returned unknown".to_string(),
            },
            1,
            Some(SmtStatus::Unknown),
        );
        assert_eq!(artifact.composite_verdict, CompositeVerdict::Unsupported);
    }
}
