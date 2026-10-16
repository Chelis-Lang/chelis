//! Machine-arithmetic corroboration of an over-the-reals green (chelis#3634).
//!
//! A Tier B decision is made over the REALS (`f32`/`f64` lower to `Real`,
//! integer widths to an unbounded integer sort), so it rules out a real
//! counterexample and nothing else. A property that is true over the reals can
//! still be false in binary64 -- `(x + 1.0f64) - x > 0.5f64` holds over the
//! reals and fails at `x = 1e17f64`, where `x + 1.0` rounds back to `x`.
//!
//! Before chelis#3634 `--tier auto` treated such a green as terminal: Tier C
//! never ran, so `auto` reported `passed` on properties `--tier fuzz-only`
//! refuted on the same file, and the plain-text render printed the `0/0 passed`
//! sample ratio an over-reals proof does not have. `auto` now runs Tier C after
//! an over-reals green as a REFUTATION SEARCH and records the result:
//!
//! - a Tier C counterexample is a concrete machine value, so it is a DEFINITE
//!   failure and replaces the green;
//! - a clean Tier C run corroborates the green without strengthening it
//!   (sampling is not exhaustive, so the `real_arithmetic` qualifier stays);
//! - a Tier C run that cannot sample leaves the green exactly as it was. A
//!   sampler limitation never degrades a proof.
//!
//! The lane is keyed on the record's published `qualifiers` array rather than on
//! a verdict token, so a future badge that discloses the same gap inherits the
//! corroboration without a second list to keep in step.
//!
//! Both user `@property` records and derived producer obligations carry the
//! check. Wiring the obligation lane required fixing chelis#3651 first: its
//! Tier C validation lowered an int-typed module constant in the invariant
//! predicate as a real, so it reported a false counterexample that `auto` would
//! have promoted over a correct proof.

use serde::{Deserialize, Serialize};

/// The outcome of the machine-arithmetic check `auto` runs after an
/// over-the-reals green. Absent from a record when no check was attempted (an
/// explicitly selected tier, or a green whose base is not over the reals).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MachineArithmeticCheck {
    pub status: MachineArithmeticStatus,
    /// Samples the Tier C refutation search accepted and evaluated.
    pub samples: usize,
    /// The over-the-reals verdict this check was run against. On a refutation
    /// it names the green the counterexample displaces.
    pub real_arithmetic_verdict: &'static str,
    /// Why the check could not be completed. Present only on
    /// [`MachineArithmeticStatus::Unchecked`].
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum MachineArithmeticStatus {
    /// Every accepted machine sample satisfied the goal. The green stands; its
    /// `real_arithmetic` qualifier is unchanged, because sampling is not
    /// exhaustive.
    Corroborated,
    /// A concrete machine counterexample refuted the goal. The record is a
    /// definite failure.
    Refuted,
    /// The refutation search ran but established nothing (a binder Tier C
    /// cannot sample, generator starvation, or precondition exhaustion). The
    /// green stands unchanged.
    Unchecked,
}

impl MachineArithmeticCheck {
    pub fn corroborated(samples: usize, real_arithmetic_verdict: &'static str) -> Self {
        Self {
            status: MachineArithmeticStatus::Corroborated,
            samples,
            real_arithmetic_verdict,
            reason: None,
        }
    }

    pub fn refuted(samples: usize, real_arithmetic_verdict: &'static str) -> Self {
        Self {
            status: MachineArithmeticStatus::Refuted,
            samples,
            real_arithmetic_verdict,
            reason: None,
        }
    }

    pub fn unchecked(
        samples: usize,
        real_arithmetic_verdict: &'static str,
        reason: impl Into<String>,
    ) -> Self {
        Self {
            status: MachineArithmeticStatus::Unchecked,
            samples,
            real_arithmetic_verdict,
            reason: Some(reason.into()),
        }
    }

    /// The one-clause plain-text disclosure a CLI render appends to an
    /// over-the-reals green, in place of the `0/0 passed` sample ratio such a
    /// green does not have. "Machine arithmetic" rather than "binary64": the
    /// same gap covers `f32` and the integer widths, which lower to an
    /// unbounded integer sort.
    pub fn disclosure(&self) -> String {
        match self.status {
            MachineArithmeticStatus::Corroborated => format!(
                "proved over the reals; machine arithmetic corroborated on {} samples",
                self.samples
            ),
            // A refuted check never renders beside a green.
            MachineArithmeticStatus::Refuted => {
                "proved over the reals; refuted in machine arithmetic".to_string()
            }
            MachineArithmeticStatus::Unchecked => format!(
                "proved over the reals; machine arithmetic unchecked: {}",
                self.reason.as_deref().unwrap_or("no reason recorded")
            ),
        }
    }
}

/// The disclosure for an over-the-reals green that carries no check at all,
/// because the selected tier did not run one (`--tier smt-only`).
pub const UNCHECKED_DISCLOSURE: &str = "proved over the reals; machine arithmetic unchecked";

/// The `reason` recorded on a record whose over-the-reals green was replaced by
/// a machine counterexample, so the failure says why a proof did not prevent it.
pub const REFUTED_REASON: &str = "proved over the reals, refuted in machine arithmetic: the Tier B decision lowers \
     floating point to the reals, so it does not rule out this counterexample (chelis#3634)";

/// Whether `auto` owes a green a machine-arithmetic refutation search. The
/// question is asked of the record's own PUBLISHED caveats (`disclosed`, the
/// `qualifiers:[...]` array), so the lane and the disclosure can never disagree
/// about which greens carry the gap.
///
/// Three conditions:
///
/// - the tier is the default composite lane. An explicitly selected tier is
///   never widened: `smt-only` asked for the deductive verdict alone, and
///   `induction-only` / `beacon-only` are terminal lanes by their own
///   contracts.
/// - the record is a reported pass. A failure or an unsupported result makes no
///   claim a counterexample could displace.
/// - the green discloses `real_arithmetic` and does NOT rest on an asserted
///   axiom. An axiom is a user-declared assumption the concrete evaluator does
///   not honor -- the Clarabel ideal-optimality contract idealizes the float
///   solver away on purpose -- so Tier C there would refute the axiom rather
///   than the derivation, and the conditional green stands.
pub fn corroboration_required(tier: &str, reported_pass: bool, disclosed: &[&str]) -> bool {
    tier == "auto"
        && reported_pass
        && disclosed.contains(&crate::discharge::Qualifier::RealArith.as_str())
        && !disclosed.contains(&crate::discharge::Qualifier::Axiom.as_str())
}

/// Fold a Tier C refutation search into the over-the-reals green it was run
/// against.
///
/// A counterexample REPLACES the green: it is a concrete machine witness, so
/// the record is a definite `failed`, not a reals-hedged one. Anything else
/// leaves the green's status, tier, badge, and qualifiers exactly as Tier B
/// produced them and only records what the search established -- a sampler that
/// cannot generate a binder must never degrade a proof.
pub(crate) fn resolve_property(
    green: crate::property_runner::PropertyOutcome,
    probe: crate::property_runner::PropertyOutcome,
) -> crate::property_runner::PropertyOutcome {
    let verdict = green.composite_verdict.as_str();
    match probe.display_status() {
        "failed" if probe.counterexample.is_some() => {
            let samples = probe.samples;
            let mut refuted = probe;
            refuted.reason = Some(match refuted.reason {
                Some(existing) => format!("{REFUTED_REASON}; {existing}"),
                None => REFUTED_REASON.to_string(),
            });
            refuted.with_machine_arithmetic(MachineArithmeticCheck::refuted(samples, verdict))
        }
        "passed" => green
            .with_machine_arithmetic(MachineArithmeticCheck::corroborated(probe.samples, verdict)),
        _ => {
            let reason = probe
                .reason
                .unwrap_or_else(|| NOTHING_ESTABLISHED.to_string());
            green.with_machine_arithmetic(MachineArithmeticCheck::unchecked(
                probe.samples,
                verdict,
                reason,
            ))
        }
    }
}

/// Fold a Tier C refutation search into the over-the-reals green it was run
/// against (derived-obligation lane). Same rule as [`resolve_property`].
pub(crate) fn resolve_obligation(
    green: crate::obligation_engine::ObligationOutcome,
    probe: crate::obligation_engine::ObligationOutcome,
) -> crate::obligation_engine::ObligationOutcome {
    let verdict = green.composite_verdict.as_str();
    if probe.composite_verdict == crate::composition::CompositeVerdict::Failed
        && probe.counterexample.is_some()
    {
        let samples = probe.samples;
        let mut refuted = probe;
        refuted.reason = Some(match refuted.reason {
            Some(existing) => format!("{REFUTED_REASON}; {existing}"),
            None => REFUTED_REASON.to_string(),
        });
        return refuted.with_machine_arithmetic(MachineArithmeticCheck::refuted(samples, verdict));
    }
    if probe.composite_verdict.is_green()
        && probe.status == crate::obligation_engine::ObligationStatus::Passed
    {
        return green
            .with_machine_arithmetic(MachineArithmeticCheck::corroborated(probe.samples, verdict));
    }
    let samples = probe.samples;
    let reason = probe
        .reason
        .unwrap_or_else(|| NOTHING_ESTABLISHED.to_string());
    green.with_machine_arithmetic(MachineArithmeticCheck::unchecked(samples, verdict, reason))
}

/// The recorded reason when the refutation search produced neither a
/// counterexample nor a clean pass and left no reason of its own.
const NOTHING_ESTABLISHED: &str = "Tier C established nothing";

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::CompositeVerdict;
    use crate::discharge::{Qualifier, QualifierSet, Soundness};
    use crate::obligation_engine::{ObligationOutcome, ObligationStatus, ObligationTier};
    use crate::obligations::ObligationMeta;
    use crate::property_runner::{PropertyOutcome, PropertyStatus, PropertyTier};

    fn real_arith_base() -> Option<(Soundness, QualifierSet)> {
        Some((
            Soundness::SoundApproximate,
            QualifierSet::from_iter_kinds([Qualifier::RealArith]),
        ))
    }

    fn over_reals_green() -> PropertyOutcome {
        PropertyOutcome::with_base_discharge(
            "p",
            PropertyStatus::Passed,
            PropertyTier::Smt,
            0,
            7,
            None,
            None,
            false,
            Vec::new(),
            real_arith_base(),
        )
    }

    fn fuzz_probe(
        status: PropertyStatus,
        samples: usize,
        counterexample: Option<serde_json::Value>,
        reason: Option<String>,
    ) -> PropertyOutcome {
        PropertyOutcome::new(
            "p",
            status,
            PropertyTier::Fuzz,
            samples,
            7,
            counterexample,
            reason,
            false,
            Vec::new(),
        )
    }

    fn obligation_meta() -> ObligationMeta {
        ObligationMeta {
            obligation_kind: "invariant_producer".to_string(),
            source_type: "Probability".to_string(),
            producer: "probability".to_string(),
        }
    }

    fn obligation_green() -> ObligationOutcome {
        ObligationOutcome::new(
            "invariant:Probability:probability",
            obligation_meta(),
            ObligationStatus::Passed,
            ObligationTier::Smt,
            0,
            7,
            real_arith_base(),
            None,
            None,
            Vec::new(),
        )
    }

    fn obligation_probe(
        status: ObligationStatus,
        samples: usize,
        counterexample: Option<serde_json::Value>,
        reason: Option<String>,
    ) -> ObligationOutcome {
        ObligationOutcome::new(
            "invariant:Probability:probability",
            obligation_meta(),
            status,
            ObligationTier::Fuzz,
            samples,
            7,
            None,
            counterexample,
            reason,
            Vec::new(),
        )
    }

    // chelis#3634: the whole point of the lane. A concrete machine
    // counterexample is not hedged by the real-arithmetic model, so it REPLACES
    // the green with a definite failure that names what it displaced.
    #[test]
    fn a_machine_counterexample_replaces_the_over_reals_green() {
        let resolved = resolve_property(
            over_reals_green(),
            fuzz_probe(
                PropertyStatus::Failed,
                100,
                Some(serde_json::json!({ "x": "1e17" })),
                None,
            ),
        );
        assert_eq!(resolved.display_status(), "failed");
        assert_eq!(resolved.composite_verdict, CompositeVerdict::Failed);
        assert_eq!(resolved.proof_tier, PropertyTier::Fuzz);
        assert!(resolved.counterexample.is_some());
        let check = resolved.machine_arithmetic.expect("check recorded");
        assert_eq!(check.status, MachineArithmeticStatus::Refuted);
        assert_eq!(check.samples, 100);
        assert_eq!(
            check.real_arithmetic_verdict, "proven_modulo_real_arithmetic",
            "the refutation names the green it displaced"
        );
        assert!(
            resolved
                .reason
                .as_deref()
                .unwrap_or_default()
                .contains("refuted in machine arithmetic"),
            "the failure says why a proof did not prevent it: {:?}",
            resolved.reason
        );
    }

    // Negative parity: a clean machine run must not touch the verdict. Sampling
    // is not exhaustive, so it cannot drop `real_arithmetic` or promote the
    // badge -- it only records that nothing was found.
    #[test]
    fn a_clean_machine_run_corroborates_without_strengthening() {
        let resolved = resolve_property(
            over_reals_green(),
            fuzz_probe(PropertyStatus::Passed, 100, None, None),
        );
        assert_eq!(resolved.display_status(), "passed");
        assert_eq!(
            resolved.composite_verdict,
            CompositeVerdict::ProvenModuloRealArithmetic
        );
        assert_eq!(resolved.proof_tier, PropertyTier::Smt);
        assert_eq!(resolved.samples, 0, "an SMT proof still takes no samples");
        assert_eq!(resolved.disclosed_qualifiers(), vec!["real_arithmetic"]);
        let check = resolved.machine_arithmetic.expect("check recorded");
        assert_eq!(check.status, MachineArithmeticStatus::Corroborated);
        assert_eq!(check.samples, 100);
    }

    // A sampler limitation is not evidence about the property. The proof must
    // survive it verbatim, with the reason recorded.
    #[test]
    fn a_sampler_that_establishes_nothing_never_degrades_the_proof() {
        let resolved = resolve_property(
            over_reals_green(),
            fuzz_probe(
                PropertyStatus::Unsupported,
                0,
                None,
                Some("generator starved".to_string()),
            ),
        );
        assert_eq!(resolved.display_status(), "passed");
        assert_eq!(
            resolved.composite_verdict,
            CompositeVerdict::ProvenModuloRealArithmetic
        );
        let check = resolved.machine_arithmetic.expect("check recorded");
        assert_eq!(check.status, MachineArithmeticStatus::Unchecked);
        assert_eq!(check.reason.as_deref(), Some("generator starved"));
    }

    #[test]
    fn the_obligation_lane_resolves_the_same_three_ways() {
        let refuted = resolve_obligation(
            obligation_green(),
            obligation_probe(
                ObligationStatus::Failed,
                16,
                Some(serde_json::json!({ "__arg0": "1e17" })),
                None,
            ),
        );
        assert_eq!(refuted.status, ObligationStatus::Failed);
        assert_eq!(refuted.composite_verdict, CompositeVerdict::Failed);
        assert_eq!(refuted.proof_tier, ObligationTier::Fuzz);
        assert_eq!(
            refuted.machine_arithmetic.map(|c| c.status),
            Some(MachineArithmeticStatus::Refuted)
        );

        let corroborated = resolve_obligation(
            obligation_green(),
            obligation_probe(ObligationStatus::Passed, 16, None, None),
        );
        assert_eq!(corroborated.status, ObligationStatus::Passed);
        assert_eq!(
            corroborated.composite_verdict,
            CompositeVerdict::ProvenModuloRealArithmetic
        );
        assert_eq!(corroborated.proof_tier, ObligationTier::Smt);
        assert_eq!(
            corroborated.machine_arithmetic.map(|c| c.status),
            Some(MachineArithmeticStatus::Corroborated)
        );

        let unchecked = resolve_obligation(
            obligation_green(),
            obligation_probe(
                ObligationStatus::Unsupported,
                0,
                None,
                Some("producer has no callable signature to sample".to_string()),
            ),
        );
        assert_eq!(unchecked.status, ObligationStatus::Passed);
        assert_eq!(
            unchecked.composite_verdict,
            CompositeVerdict::ProvenModuloRealArithmetic
        );
        let check = unchecked.machine_arithmetic.expect("check recorded");
        assert_eq!(check.status, MachineArithmeticStatus::Unchecked);
        assert_eq!(
            check.reason.as_deref(),
            Some("producer has no callable signature to sample")
        );
    }

    // An explicitly selected tier is never widened, a non-pass owes nothing,
    // and a green whose base is not over the reals has no gap to close.
    #[test]
    fn corroboration_is_owed_only_by_an_auto_green_decided_over_the_reals() {
        let over_reals = ["real_arithmetic"];
        assert!(corroboration_required("auto", true, &over_reals));
        for tier in ["smt-only", "fuzz-only", "induction-only", "beacon-only"] {
            assert!(
                !corroboration_required(tier, true, &over_reals),
                "{tier} must not be widened into a second tier"
            );
        }
        assert!(
            !corroboration_required("auto", false, &over_reals),
            "a non-pass is already not a green claim"
        );
        assert!(
            !corroboration_required("auto", true, &["fuzz_base"]),
            "a base that is not over the reals has no machine-arithmetic gap"
        );
        assert!(
            !corroboration_required("auto", true, &["axiom", "real_arithmetic"]),
            "a green conditional on a user-asserted axiom is not refuted by a \
             search that ignores the axiom"
        );
        assert!(
            corroboration_required("auto", true, &["fuzz", "real_arithmetic"]),
            "a fuzz-validated CONTRACT is empirical, not an asserted idealization"
        );
    }

    // The predicate is keyed on the DISCLOSED qualifier, not on a token list,
    // so the two outcome types answer it from their own composed base.
    #[test]
    fn a_fuzz_base_green_does_not_disclose_real_arithmetic() {
        let fuzz_green = fuzz_probe(PropertyStatus::Passed, 100, None, None);
        assert_eq!(
            fuzz_green.composite_verdict,
            CompositeVerdict::FuzzValidatedEmpirical
        );
        assert!(!fuzz_green.discloses_real_arithmetic());
        assert!(over_reals_green().discloses_real_arithmetic());
        assert!(obligation_green().discloses_real_arithmetic());
    }
}
