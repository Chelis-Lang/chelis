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
    /// Structural induction over recursive definitions.
    Induction,
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
    /// Agent-oriented failure summary (chelis#489). Present for non-passing
    /// properties/obligations so agents can parse degradation structurally.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_summary: Option<FailureSummary>,
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
            // chelis#422: a Tier-B proof is over the reals, disclosing the
            // machine-arithmetic gap; a Tier-C statistically-validated pass is
            // a fuzz-only base, empirically validated but NOT proven. Neither
            // may read as a plain `proven` / `proven_modulo_fuzz_*` badge.
            ProofStatus::Proved => CompositeVerdict::ProvenModuloRealArithmetic,
            ProofStatus::StatisticallyValidated { samples } if *samples > 0 => {
                CompositeVerdict::FuzzValidatedEmpirical
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
            failure_summary: None,
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

/// Agent-oriented failure summary for a property/obligation (chelis#489).
///
/// Emitted in `prove --json` for each non-passing property so agents can
/// parse structured degradation without scraping harness text.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FailureSummary {
    pub status: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub requested_tier: Option<String>,
    pub actual_tier: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub degradation: Option<Degradation>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub promotability: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub counterexample: Option<serde_json::Value>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub seed: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub samples: Option<usize>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub timeout_ms: Option<u64>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub producer: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub obligation_origin: Option<String>,
}

/// Structured tier-degradation metadata (chelis#489).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Degradation {
    pub degraded: bool,
    pub reason: String,
    pub from_tier: String,
    pub to_tier: String,
}

/// Property dependency edge: maps a property to the exports it references
/// in its body (chelis#490).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PropertyDependency {
    pub property: String,
    pub references: Vec<String>,
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

    #[test]
    fn failure_summary_round_trips() {
        let summary = FailureSummary {
            status: "failed".to_string(),
            requested_tier: Some("smt".to_string()),
            actual_tier: "fuzz".to_string(),
            degradation: Some(Degradation {
                degraded: true,
                reason: "smt timeout".to_string(),
                from_tier: "smt".to_string(),
                to_tier: "fuzz".to_string(),
            }),
            promotability: Some("amenable_to_smt".to_string()),
            counterexample: Some(serde_json::json!({"x": 42})),
            seed: Some(123),
            samples: Some(100),
            timeout_ms: Some(5000),
            producer: None,
            obligation_origin: None,
        };
        let json = serde_json::to_string(&summary).unwrap();
        let back: FailureSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back, summary);
    }

    #[test]
    fn failure_summary_minimal_round_trips() {
        let summary = FailureSummary {
            status: "unsupported".to_string(),
            requested_tier: None,
            actual_tier: "fuzz".to_string(),
            degradation: None,
            promotability: None,
            counterexample: None,
            seed: None,
            samples: None,
            timeout_ms: None,
            producer: None,
            obligation_origin: None,
        };
        let json = serde_json::to_string(&summary).unwrap();
        assert!(!json.contains("requested_tier"));
        assert!(!json.contains("degradation"));
        let back: FailureSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(back, summary);
    }

    #[test]
    fn degradation_round_trips() {
        let d = Degradation {
            degraded: true,
            reason: "property not amenable to SMT".to_string(),
            from_tier: "smt".to_string(),
            to_tier: "fuzz".to_string(),
        };
        let json = serde_json::to_string(&d).unwrap();
        let back: Degradation = serde_json::from_str(&json).unwrap();
        assert_eq!(back, d);
    }

    #[test]
    fn artifact_with_failure_summary_round_trips() {
        let mut a = base();
        a.failure_summary = Some(FailureSummary {
            status: "failed".to_string(),
            requested_tier: Some("smt".to_string()),
            actual_tier: "fuzz".to_string(),
            degradation: None,
            promotability: None,
            counterexample: None,
            seed: Some(0),
            samples: Some(100),
            timeout_ms: None,
            producer: None,
            obligation_origin: None,
        });
        let json = serde_json::to_string(&a).unwrap();
        assert!(json.contains("failure_summary"));
        let back: ProofArtifact = serde_json::from_str(&json).unwrap();
        assert_eq!(back, a);
    }

    #[test]
    fn property_dependency_round_trips() {
        let dep = PropertyDependency {
            property: "test_add".to_string(),
            references: vec!["add".to_string(), "helper".to_string()],
        };
        let json = serde_json::to_string(&dep).unwrap();
        let back: PropertyDependency = serde_json::from_str(&json).unwrap();
        assert_eq!(back, dep);
    }
}
