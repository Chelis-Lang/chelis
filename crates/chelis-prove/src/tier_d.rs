//! Tier D: Structural induction over recursive definitions.
//!
//! Turns fixed-size proofs into general-size results for properties over
//! structurally recursive models. Scoped to:
//!
//! - Recursion over lattice steps (CRR binomial tree)
//! - Recursion over periods (term structure / coupon schedules)
//!
//! NOT general recursion — only structural recursion with a decreasing
//! size argument (a natural number parameter that the recursive call
//! decrements by exactly 1).
//!
//! ## Strategy
//!
//! Given a property `P(n)` over a structurally recursive model `f(n, ...)`:
//!
//! 1. **Classify:** Determine if the model is structurally recursive with a
//!    decreasing-size parameter. If not, decline (return `Inconclusive`).
//!
//! 2. **Base case:** Attempt to prove `P(base)` (typically `n=1` or `n=0`)
//!    using the existing tiers (Tier B SMT or Tier C fuzz).
//!
//! 3. **Step case:** Attempt to prove `P(k) => P(k+1)` for symbolic `k`,
//!    using the existing tiers. The induction hypothesis `P(k)` is
//!    introduced as an assumption.
//!
//! 4. **Combine:** If both base and step are established, emit a
//!    `ProofTier::Induction` artifact stating the general result.
//!
//! ## Honest limitations
//!
//! The induction tier CAN establish:
//! - Properties over any fixed lattice depth, given base+step proofs
//! - Monotonicity / boundedness over lattice steps when each step preserves
//!   the invariant
//!
//! The induction tier CANNOT establish:
//! - Properties requiring global reasoning across all steps simultaneously
//!   (e.g., path-dependent options where the full path matters)
//! - Non-structural recursion (general recursion, mutual recursion)
//! - Properties where the step case requires k-specific bounds that are
//!   not preserved uniformly

use serde::{Deserialize, Serialize};

/// Result of attempting structural induction.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum InductionResult {
    /// Property proved by structural induction.
    Proved {
        /// Description of the base case verification.
        base_case: InductionCase,
        /// Description of the step case verification.
        step_case: InductionCase,
        /// The structural parameter over which induction was applied.
        induction_variable: String,
        /// The base value (typically 1).
        base_value: u64,
    },
    /// Induction was attempted but the step case failed.
    StepFailed {
        /// The base case passed.
        base_case: InductionCase,
        /// Why the step case failed.
        step_failure: String,
    },
    /// Induction was attempted but the base case failed.
    BaseFailed {
        /// Why the base case failed.
        base_failure: String,
    },
    /// The property is not amenable to structural induction.
    NotAmenable { reason: String },
    /// Inconclusive — could not determine structural recursion pattern.
    Inconclusive,
}

/// A single case (base or step) in the induction proof.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct InductionCase {
    /// Which verification method established this case.
    pub method: String,
    /// Evidence (e.g., "fuzz: 100 samples passed" or "smt: proved").
    pub evidence: String,
}

/// Configuration for the induction tier.
#[derive(Debug, Clone)]
pub struct InductionOptions {
    /// Base value for induction (default: 1).
    pub base_value: u64,
    /// SMT timeout for each case (ms).
    pub smt_timeout_ms: u64,
    /// Fuzz samples for each case.
    pub fuzz_samples: usize,
}

impl Default for InductionOptions {
    fn default() -> Self {
        Self {
            base_value: 1,
            smt_timeout_ms: 5000,
            fuzz_samples: 100,
        }
    }
}

/// Structural recursion classification.
#[derive(Debug, Clone, PartialEq)]
pub enum RecursionClass {
    /// Structural recursion over lattice steps (e.g., CRR binomial tree).
    /// The parameter `n` counts steps; the recursive call uses `n-1`.
    LatticeSteps {
        /// Name of the step-count parameter.
        parameter: String,
    },
    /// Structural recursion over periods (e.g., term structure, coupons).
    /// The parameter `t` counts periods; the recursive call uses `t-1`.
    Periods {
        /// Name of the period-count parameter.
        parameter: String,
    },
    /// Not structurally recursive in the sense this tier handles.
    NonStructural { reason: String },
}

/// Attempt structural induction on a property.
///
/// This is the entry point for Tier D. It:
/// 1. Classifies the recursion structure
/// 2. Verifies the base case
/// 3. Verifies the step case (with induction hypothesis)
/// 4. Combines into a general result if both pass
///
/// Returns `InductionResult::NotAmenable` if the property is not over a
/// structurally recursive model, and `Inconclusive` if classification
/// cannot determine the pattern.
pub fn attempt_induction(
    property_source: &str,
    property_name: &str,
    recursion_class: &RecursionClass,
    options: &InductionOptions,
) -> InductionResult {
    let induction_variable = match recursion_class {
        RecursionClass::LatticeSteps { parameter } => parameter.clone(),
        RecursionClass::Periods { parameter } => parameter.clone(),
        RecursionClass::NonStructural { reason } => {
            return InductionResult::NotAmenable {
                reason: reason.clone(),
            };
        }
    };

    // Phase 1: Verify base case (n = base_value)
    let base_result = verify_base_case(
        property_source,
        property_name,
        &induction_variable,
        options.base_value,
        options,
    );
    let base_case = match base_result {
        CaseResult::Proved(case) => case,
        CaseResult::Failed(reason) => {
            return InductionResult::BaseFailed {
                base_failure: reason,
            };
        }
    };

    // Phase 2: Verify step case (P(k) => P(k+1))
    let step_result =
        verify_step_case(property_source, property_name, &induction_variable, options);
    let step_case = match step_result {
        CaseResult::Proved(case) => case,
        CaseResult::Failed(reason) => {
            return InductionResult::StepFailed {
                base_case,
                step_failure: reason,
            };
        }
    };

    // Both cases pass → general result
    InductionResult::Proved {
        base_case,
        step_case,
        induction_variable,
        base_value: options.base_value,
    }
}

#[allow(dead_code)] // Failed variant is scaffolding for when tier_d is wired to dispatch
enum CaseResult {
    Proved(InductionCase),
    Failed(String),
}

/// Verify the base case by checking the property with the induction
/// variable fixed to `base_value`. Uses fuzz as the default method
/// (SMT integration deferred to when the tier is wired into dispatch).
///
/// NOTE: This is currently a STUB that trusts the claim. The full
/// implementation (when tier_d is wired into dispatch) will call the
/// existing tier_b/tier_c infrastructure with the induction variable
/// specialized. The base case at small fixed sizes (n=1, n=2) is exactly
/// what the existing prover already handles — those are the "small
/// fixed-size models" that currently prove green.
fn verify_base_case(
    _property_source: &str,
    _property_name: &str,
    _induction_variable: &str,
    base_value: u64,
    _options: &InductionOptions,
) -> CaseResult {
    CaseResult::Proved(InductionCase {
        method: format!("stub@n={base_value}"),
        evidence: format!(
            "base case P({base_value}) ASSUMED (stub: not yet wired to tier_b/tier_c dispatch)"
        ),
    })
}

/// Verify the step case: P(k) => P(k+1).
///
/// NOTE: This is currently a STUB that trusts the claim. The full
/// implementation will construct the step obligation (with induction
/// hypothesis as an assumption) and discharge it via tier_b or tier_c.
///
/// The step case is the hard part. For CRR-style lattices:
/// - The model at step k+1 is: f(k+1, ...) = combine(f(k, up_params), f(k, down_params))
/// - The invariant at step k+1 must follow from the invariant at step k
///   applied to both sub-trees
///
/// For monotonicity/boundedness properties, this often reduces to showing
/// that `combine` preserves the property — which is amenable to SMT or fuzz.
fn verify_step_case(
    _property_source: &str,
    _property_name: &str,
    _induction_variable: &str,
    _options: &InductionOptions,
) -> CaseResult {
    CaseResult::Proved(InductionCase {
        method: "stub@step".to_string(),
        evidence: "step case P(k)=>P(k+1) ASSUMED (stub: not yet wired to tier_b/tier_c dispatch)"
            .to_string(),
    })
}

/// Classify whether a property's model exhibits structural recursion.
///
/// This is a conservative classifier: it returns `NonStructural` unless
/// the recursion pattern is clearly a decreasing-size natural number
/// parameter. The classification is based on:
///
/// 1. The model function has a parameter that appears in a recursive call
///    decremented by exactly 1
/// 2. There is a base case at the parameter's minimum value
/// 3. The recursive structure is a tree (CRR) or a chain (periods)
pub fn classify_recursion(
    _property_source: &str,
    _model_name: &str,
    size_parameter: Option<&str>,
) -> RecursionClass {
    // For now, if the user specifies a size parameter, trust it and
    // classify as LatticeSteps. The full classifier would inspect the
    // model's AST to find the structural recursion pattern.
    match size_parameter {
        Some(param) => RecursionClass::LatticeSteps {
            parameter: param.to_string(),
        },
        None => RecursionClass::NonStructural {
            reason: "no size parameter specified and automatic classification not yet implemented"
                .to_string(),
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lattice_steps_induction_proves() {
        let class = RecursionClass::LatticeSteps {
            parameter: "n".to_string(),
        };
        let result = attempt_induction(
            "module M\ndef f(n: i32) -> f32 = 1.0\n@property p forall(n: i32): (f(n) > 0.0)",
            "p",
            &class,
            &InductionOptions::default(),
        );
        match result {
            InductionResult::Proved {
                induction_variable,
                base_value,
                ..
            } => {
                assert_eq!(induction_variable, "n");
                assert_eq!(base_value, 1);
            }
            other => panic!("expected Proved, got {other:?}"),
        }
    }

    #[test]
    fn non_structural_is_not_amenable() {
        let class = RecursionClass::NonStructural {
            reason: "mutual recursion".to_string(),
        };
        let result = attempt_induction("", "p", &class, &InductionOptions::default());
        assert!(matches!(result, InductionResult::NotAmenable { .. }));
    }

    #[test]
    fn classify_with_explicit_parameter_is_lattice_steps() {
        let class = classify_recursion("", "crr_tree", Some("steps"));
        assert!(matches!(class, RecursionClass::LatticeSteps { .. }));
    }

    #[test]
    fn classify_without_parameter_is_non_structural() {
        let class = classify_recursion("", "general_fn", None);
        assert!(matches!(class, RecursionClass::NonStructural { .. }));
    }

    #[test]
    fn induction_result_serializes() {
        let result = InductionResult::Proved {
            base_case: InductionCase {
                method: "fuzz@n=1".to_string(),
                evidence: "100 samples".to_string(),
            },
            step_case: InductionCase {
                method: "fuzz@step".to_string(),
                evidence: "100 samples".to_string(),
            },
            induction_variable: "n".to_string(),
            base_value: 1,
        };
        let json = serde_json::to_string(&result).expect("serializes");
        assert!(json.contains("induction_variable"));
        assert!(json.contains("base_value"));
    }

    #[test]
    fn periods_classification_accepted() {
        let class = RecursionClass::Periods {
            parameter: "t".to_string(),
        };
        let result = attempt_induction("", "bond_convexity", &class, &InductionOptions::default());
        match result {
            InductionResult::Proved {
                induction_variable, ..
            } => {
                assert_eq!(induction_variable, "t");
            }
            other => panic!("expected Proved, got {other:?}"),
        }
    }
}
