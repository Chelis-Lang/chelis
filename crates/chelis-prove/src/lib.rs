//! Three-tier property verification dispatcher.
//!
//! This crate implements the specification-consumption layer of the Chelis
//! trust stack. It takes `@property` declarations and dispatches them through:
//!
//! - **Tier A:** Type system validation (dimension, effect, linearity discharge)
//! - **Tier B:** SMT solving via cvc5 (nonlinear real arithmetic)
//! - **Tier C:** Randomized fuzz testing (existing `chelis prove` logic)
//!
//! See `docs/trust-stack-verification.md` for architectural framing.

pub mod artifact;
pub mod convert;
pub mod dispatch;
pub mod from_property_spec;
pub mod inlineability;
pub mod solver;
pub mod tier_a;
pub mod tier_b;
pub mod tier_c;

pub use artifact::{ProofArtifact, ProofStatus, ProofTier};
pub use dispatch::{DispatchOptions, dispatch_property};
pub use from_property_spec::{PropertySpecInput, to_smt_property, to_dispatch_amenability};
pub use inlineability::{Fuzzability, Inlineability, classify_fuzzability, classify_inlineability};
pub use tier_b::{SmtProperty, solve_property};

// --- Verification Pipeline API ---

/// Source language for verification input.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SourceKind { Surf, Deep }

/// Tier selection for verification dispatch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum TierSelection { Auto, FuzzOnly, SmtOnly, TypeOnly }

/// Request to verify source containing @property declarations.
#[derive(Debug, Clone)]
pub struct VerificationRequest {
    pub source: String,
    pub source_kind: SourceKind,
    pub tier: TierSelection,
    pub smt_timeout_ms: u64,
    pub fuzz_samples: u32,
    pub fuzz_seed: u64,
    pub inlining_depth_limit: u32,
}

impl Default for VerificationRequest {
    fn default() -> Self {
        Self { source: String::new(), source_kind: SourceKind::Surf, tier: TierSelection::Auto, smt_timeout_ms: 5000, fuzz_samples: 100, fuzz_seed: 0, inlining_depth_limit: 3 }
    }
}

/// Provenance chain back to EARS source.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Provenance {
    pub requirement_id: Option<String>,
    pub file: Option<String>,
    pub line: Option<usize>,
    pub text: Option<String>,
}

/// Counterexample from SMT or fuzz.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct Counterexample {
    pub bindings: Vec<(String, f64)>,
}

/// Result for a single property verification.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct PropertyResult {
    pub name: String,
    pub proof_tier: ProofTier,
    pub status: ProofStatus,
    pub counterexample: Option<Counterexample>,
    pub provenance: Option<Provenance>,
    pub duration_ms: u64,
}

/// Aggregate verification summary.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationSummary {
    pub total: usize,
    pub proved: usize,
    pub statistically_validated: usize,
    pub failed: usize,
    pub rejected: usize,
    pub inconclusive: usize,
}

/// Complete verification result.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct VerificationResult {
    pub properties: Vec<PropertyResult>,
    pub summary: VerificationSummary,
}

/// Error from verification.
#[derive(Debug, Clone, thiserror::Error)]
pub enum VerifyError {
    #[error("parse error: {0}")]
    Parse(String),
    #[error("no properties found in source")]
    NoProperties,
    #[error("internal error: {0}")]
    Internal(String),
}

/// Main entry point: verify all properties in a source string.
/// Both CLI and MCP server call this function.
pub fn verify_source(_req: VerificationRequest) -> Result<VerificationResult, VerifyError> {
    // Stub: will be implemented in Task 1c-5
    Err(VerifyError::Internal("not yet implemented".into()))
}
