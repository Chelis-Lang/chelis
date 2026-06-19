//! Three-tier dispatch orchestration.

use crate::artifact::{ProofArtifact, ProofStatus, ProofTier, SmtStatus};
use std::time::Instant;

/// Dispatch configuration.
#[derive(Debug, Clone)]
pub struct DispatchOptions {
    /// Which tiers to run.
    pub tier_mode: TierMode,
    /// SMT timeout in milliseconds.
    pub smt_timeout_ms: u64,
    /// Fuzz sample count.
    pub fuzz_samples: usize,
    /// Fuzz seed.
    pub fuzz_seed: u64,
}

impl Default for DispatchOptions {
    fn default() -> Self {
        Self {
            tier_mode: TierMode::Auto,
            smt_timeout_ms: 5000,
            fuzz_samples: 100,
            fuzz_seed: 0,
        }
    }
}

/// Tier selection mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TierMode {
    /// Run A → B → C (default).
    Auto,
    /// Tier A only.
    TypeOnly,
    /// Tier B only (error on non-amenable).
    SmtOnly,
    /// Tier C only (existing behavior).
    FuzzOnly,
}

/// SMT amenability classification for a property.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SmtAmenability {
    /// Pure linear arithmetic.
    Linear,
    /// Polynomial (nonlinear) arithmetic.
    Polynomial,
    /// Involves transcendentals (exp, log, etc.).
    Transcendental,
    /// Not expressible in SMT.
    Opaque,
}

impl SmtAmenability {
    /// Whether this property should attempt Tier B.
    pub fn is_smt_amenable(self) -> bool {
        matches!(self, Self::Linear | Self::Polynomial | Self::Transcendental)
    }
}

/// Dispatch a single property through the tier pipeline.
///
/// Returns a proof artifact indicating which tier produced the result.
pub fn dispatch_property(
    _property_source: &str,
    _property_name: &str,
    amenability: SmtAmenability,
    options: &DispatchOptions,
) -> ProofArtifact {
    let start = Instant::now();

    // Tier A: Type system check
    match options.tier_mode {
        TierMode::FuzzOnly => {}
        _ => {
            let tier_a_result = crate::tier_a::check(_property_source, _property_name);
            match tier_a_result {
                TierAResult::Rejected(reason) => {
                    return artifact(
                        _property_name,
                        ProofTier::TypeSystem,
                        ProofStatus::Rejected { reason },
                        start,
                        None,
                    );
                }
                TierAResult::Proved => {
                    return artifact(
                        _property_name,
                        ProofTier::TypeSystem,
                        ProofStatus::Proved,
                        start,
                        None,
                    );
                }
                TierAResult::Inconclusive => {}
            }
        }
    }

    if options.tier_mode == TierMode::TypeOnly {
        return artifact(
            _property_name,
            ProofTier::TypeSystem,
            ProofStatus::Unsupported {
                reason: "type-only tier did not produce a proof artifact".to_string(),
            },
            start,
            None,
        );
    }

    // Tier B: SMT
    if options.tier_mode != TierMode::FuzzOnly {
        if !amenability.is_smt_amenable() {
            if options.tier_mode == TierMode::SmtOnly {
                return artifact(
                    _property_name,
                    ProofTier::Smt,
                    ProofStatus::NotAmenable {
                        reason: "property not amenable to SMT verification; use --tier auto for fuzz fallback".to_string(),
                    },
                    start,
                    Some(SmtStatus::NotAmenable),
                );
            }
        } else {
            let tier_b_result =
                crate::tier_b::solve(_property_source, _property_name, options.smt_timeout_ms);
            match tier_b_result {
                TierBResult::Proved => {
                    return artifact(
                        _property_name,
                        ProofTier::Smt,
                        ProofStatus::Proved,
                        start,
                        Some(SmtStatus::Proved),
                    );
                }
                TierBResult::Disproved(model) => {
                    return artifact(
                        _property_name,
                        ProofTier::Smt,
                        ProofStatus::Disproved {
                            counterexample: model,
                        },
                        start,
                        Some(SmtStatus::Disproved),
                    );
                }
                TierBResult::Timeout => {
                    if options.tier_mode == TierMode::SmtOnly {
                        return artifact(
                            _property_name,
                            ProofTier::Smt,
                            ProofStatus::Unsupported {
                                reason: "cvc5 timed out".to_string(),
                            },
                            start,
                            Some(SmtStatus::Timeout),
                        );
                    }
                    // Fall through to Tier C
                }
                TierBResult::Unknown => {
                    if options.tier_mode == TierMode::SmtOnly {
                        return artifact(
                            _property_name,
                            ProofTier::Smt,
                            ProofStatus::Unsupported {
                                reason: "cvc5 returned unknown".to_string(),
                            },
                            start,
                            Some(SmtStatus::Unknown),
                        );
                    }
                    // Fall through to Tier C
                }
                TierBResult::Error(reason) => {
                    // The property did not lower to a valid SMT term
                    // (RT5-F1). In smt-only this is rejected; otherwise
                    // fall through to Tier C.
                    if options.tier_mode == TierMode::SmtOnly {
                        return artifact(
                            _property_name,
                            ProofTier::Smt,
                            ProofStatus::Unsupported { reason },
                            start,
                            Some(SmtStatus::NotAmenable),
                        );
                    }
                    // Fall through to Tier C
                }
            }
        }
    }

    // Tier C: Fuzz
    let tier_c_result = crate::tier_c::fuzz(
        _property_source,
        _property_name,
        options.fuzz_samples,
        options.fuzz_seed,
    );
    match tier_c_result {
        TierCResult::AllPassed(n) => artifact(
            _property_name,
            ProofTier::Fuzz,
            ProofStatus::StatisticallyValidated { samples: n },
            start,
            None,
        ),
        TierCResult::Failed(counterexample) => artifact(
            _property_name,
            ProofTier::Fuzz,
            ProofStatus::Disproved { counterexample },
            start,
            None,
        ),
        TierCResult::Error(reason) => artifact(
            _property_name,
            ProofTier::Fuzz,
            ProofStatus::Rejected { reason },
            start,
            None,
        ),
    }
}

fn artifact(
    property_name: &str,
    tier: ProofTier,
    status: ProofStatus,
    start: Instant,
    smt_status: Option<SmtStatus>,
) -> ProofArtifact {
    ProofArtifact::new(
        property_name,
        tier,
        status,
        start.elapsed().as_millis() as u64,
        smt_status,
    )
}

// Re-export tier result types used by dispatch.
pub use crate::tier_a::TierAResult;
pub use crate::tier_b::TierBResult;
pub use crate::tier_c::TierCResult;
