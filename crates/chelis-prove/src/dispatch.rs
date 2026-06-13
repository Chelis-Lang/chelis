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
                    return ProofArtifact {
                        property_name: _property_name.to_string(),
                        tier: ProofTier::TypeSystem,
                        status: ProofStatus::Rejected { reason },
                        duration_ms: start.elapsed().as_millis() as u64,
                        smt_status: None,
                        obligation: None,
                    };
                }
                TierAResult::Proved => {
                    return ProofArtifact {
                        property_name: _property_name.to_string(),
                        tier: ProofTier::TypeSystem,
                        status: ProofStatus::Proved,
                        duration_ms: start.elapsed().as_millis() as u64,
                        smt_status: None,
                        obligation: None,
                    };
                }
                TierAResult::Inconclusive => {}
            }
        }
    }

    if options.tier_mode == TierMode::TypeOnly {
        return ProofArtifact {
            property_name: _property_name.to_string(),
            tier: ProofTier::TypeSystem,
            status: ProofStatus::StatisticallyValidated { samples: 0 },
            duration_ms: start.elapsed().as_millis() as u64,
            smt_status: None,
            obligation: None,
        };
    }

    // Tier B: SMT
    if options.tier_mode != TierMode::FuzzOnly {
        if !amenability.is_smt_amenable() {
            if options.tier_mode == TierMode::SmtOnly {
                return ProofArtifact {
                    property_name: _property_name.to_string(),
                    tier: ProofTier::Smt,
                    status: ProofStatus::NotAmenable {
                        reason: "property not amenable to SMT verification; use --tier auto for fuzz fallback".to_string(),
                    },
                    duration_ms: start.elapsed().as_millis() as u64,
                    smt_status: Some(SmtStatus::NotAmenable),
                    obligation: None,
                };
            }
        } else {
            let tier_b_result =
                crate::tier_b::solve(_property_source, _property_name, options.smt_timeout_ms);
            match tier_b_result {
                TierBResult::Proved => {
                    return ProofArtifact {
                        property_name: _property_name.to_string(),
                        tier: ProofTier::Smt,
                        status: ProofStatus::Proved,
                        duration_ms: start.elapsed().as_millis() as u64,
                        smt_status: Some(SmtStatus::Proved),
                        obligation: None,
                    };
                }
                TierBResult::Disproved(model) => {
                    return ProofArtifact {
                        property_name: _property_name.to_string(),
                        tier: ProofTier::Smt,
                        status: ProofStatus::Disproved {
                            counterexample: model,
                        },
                        duration_ms: start.elapsed().as_millis() as u64,
                        smt_status: Some(SmtStatus::Disproved),
                        obligation: None,
                    };
                }
                TierBResult::Timeout => {
                    if options.tier_mode == TierMode::SmtOnly {
                        return ProofArtifact {
                            property_name: _property_name.to_string(),
                            tier: ProofTier::Smt,
                            status: ProofStatus::StatisticallyValidated { samples: 0 },
                            duration_ms: start.elapsed().as_millis() as u64,
                            smt_status: Some(SmtStatus::Timeout),
                            obligation: None,
                        };
                    }
                    // Fall through to Tier C
                }
                TierBResult::Unknown => {
                    if options.tier_mode == TierMode::SmtOnly {
                        return ProofArtifact {
                            property_name: _property_name.to_string(),
                            tier: ProofTier::Smt,
                            status: ProofStatus::StatisticallyValidated { samples: 0 },
                            duration_ms: start.elapsed().as_millis() as u64,
                            smt_status: Some(SmtStatus::Unknown),
                            obligation: None,
                        };
                    }
                    // Fall through to Tier C
                }
                TierBResult::Error(reason) => {
                    // The property did not lower to a valid SMT term
                    // (RT5-F1). In smt-only this is rejected; otherwise
                    // fall through to Tier C.
                    if options.tier_mode == TierMode::SmtOnly {
                        return ProofArtifact {
                            property_name: _property_name.to_string(),
                            tier: ProofTier::Smt,
                            status: ProofStatus::Rejected { reason },
                            duration_ms: start.elapsed().as_millis() as u64,
                            smt_status: Some(SmtStatus::NotAmenable),
                            obligation: None,
                        };
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
        TierCResult::AllPassed(n) => ProofArtifact {
            property_name: _property_name.to_string(),
            tier: ProofTier::Fuzz,
            status: ProofStatus::StatisticallyValidated { samples: n },
            duration_ms: start.elapsed().as_millis() as u64,
            smt_status: None,
            obligation: None,
        },
        TierCResult::Failed(counterexample) => ProofArtifact {
            property_name: _property_name.to_string(),
            tier: ProofTier::Fuzz,
            status: ProofStatus::Disproved { counterexample },
            duration_ms: start.elapsed().as_millis() as u64,
            smt_status: None,
            obligation: None,
        },
        TierCResult::Error(reason) => ProofArtifact {
            property_name: _property_name.to_string(),
            tier: ProofTier::Fuzz,
            status: ProofStatus::Rejected { reason },
            duration_ms: start.elapsed().as_millis() as u64,
            smt_status: None,
            obligation: None,
        },
    }
}

// Re-export tier result types used by dispatch.
pub use crate::tier_a::TierAResult;
pub use crate::tier_b::TierBResult;
pub use crate::tier_c::TierCResult;
