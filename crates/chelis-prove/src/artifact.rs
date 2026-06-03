//! Proof artifact types representing verification outcomes.

use serde::{Deserialize, Serialize};

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
