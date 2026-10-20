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
    /// The over-the-reals verdict this check was run against -- the green a
    /// refutation displaces, or the disproof a confirmation makes definite. It
    /// is also what tells the two sides of the lane apart: the disproof side
    /// records `disproved_modulo_real_arithmetic` here.
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
    /// A concrete machine input CONFIRMED an over-the-reals disproof: the
    /// property's own preconditions admit the solver's witness at its declared
    /// machine precision and the body is false there. The record is a definite
    /// failure at that point (chelis#3971).
    Confirmed,
    /// An over-the-reals disproof's witness is NOT a machine counterexample:
    /// no machine value equals it (an irrational algebraic witness), the
    /// property's own preconditions reject it once rounded, or the body holds
    /// there. The over-the-reals decision stands, but it has exhibited nothing
    /// about machine arithmetic (chelis#3971).
    Unconfirmed,
    /// The check ran but established nothing: on the green side a binder Tier C
    /// cannot sample, generator starvation, or precondition exhaustion; on the
    /// disproof side a model value the projection cannot read or a witness the
    /// concrete evaluator could not run. The Tier B decision stands unchanged.
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

    /// chelis#3971: the disproof side's positive answer. It takes no samples --
    /// one concrete evaluation of the solver's own witness decided it.
    pub fn confirmed(real_arithmetic_verdict: &'static str) -> Self {
        Self {
            status: MachineArithmeticStatus::Confirmed,
            samples: 0,
            real_arithmetic_verdict,
            reason: None,
        }
    }

    /// chelis#3971: the witness is not a machine counterexample, and `reason`
    /// says which of the three ways it failed to be one.
    pub fn unconfirmed(real_arithmetic_verdict: &'static str, reason: impl Into<String>) -> Self {
        Self {
            status: MachineArithmeticStatus::Unconfirmed,
            samples: 0,
            real_arithmetic_verdict,
            reason: Some(reason.into()),
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

    /// The one-clause plain-text disclosure a CLI render appends to a record
    /// carrying this check -- for an over-the-reals green, in place of the
    /// `0/0 passed` sample ratio such a green does not have. "Machine
    /// arithmetic" rather than "binary64": the same gap covers `f32` and the
    /// integer widths, which lower to an unbounded integer sort.
    pub fn disclosure(&self) -> String {
        let decision = self.real_arithmetic_decision();
        match self.status {
            MachineArithmeticStatus::Corroborated => format!(
                "{decision}; machine arithmetic corroborated on {} samples",
                self.samples
            ),
            // A refuted check never renders beside a green.
            MachineArithmeticStatus::Refuted => {
                format!("{decision}; refuted in machine arithmetic")
            }
            // A confirmed check never renders beside a pass.
            MachineArithmeticStatus::Confirmed => {
                format!("{decision}; confirmed at a machine input")
            }
            MachineArithmeticStatus::Unconfirmed => format!(
                "{decision}; not confirmed in machine arithmetic: {}",
                self.reason.as_deref().unwrap_or("no reason recorded")
            ),
            MachineArithmeticStatus::Unchecked => format!(
                "{decision}; machine arithmetic unchecked: {}",
                self.reason.as_deref().unwrap_or("no reason recorded")
            ),
        }
    }

    /// Whether this check DISPLACED the Tier B decision it was run against, so
    /// the record's reported tier is the lane's choice rather than a fall-back
    /// from a decision Tier B could not reach.
    ///
    /// A consumer that explains why a record's tier differs from the requested
    /// one has to ask this: under `auto` a refutation and an unconfirmed
    /// disproof both report the Tier C tier AFTER Tier B decided, so calling
    /// that a tier degradation states a cause that did not happen.
    ///
    /// [`MachineArithmeticStatus::Unchecked`] is deliberately NOT a
    /// displacement: there the lane established nothing and left the Tier B
    /// record exactly as it was, so whatever explains the reported tier is
    /// unchanged by this check. Neither is
    /// [`MachineArithmeticStatus::Corroborated`] or
    /// [`MachineArithmeticStatus::Confirmed`], which keep the deductive tier.
    pub fn displaced_the_tier_b_decision(&self) -> bool {
        matches!(
            self.status,
            MachineArithmeticStatus::Refuted | MachineArithmeticStatus::Unconfirmed
        )
    }

    /// The clause naming the over-the-reals decision this check was run
    /// against. Read off the recorded verdict TOKEN rather than inferred from
    /// the status, so the two sides of the lane cannot be described by each
    /// other's sentence.
    fn real_arithmetic_decision(&self) -> &'static str {
        use crate::composition::CompositeVerdict;
        if self.real_arithmetic_verdict == CompositeVerdict::DisprovedModuloRealArithmetic.as_str()
        {
            return "disproved over the reals";
        }
        // Only two verdicts can be checked against, and reaching the fallback
        // with anything else would describe a decision as its own opposite.
        // Assert it rather than let a future third producer render a lie.
        debug_assert_eq!(
            self.real_arithmetic_verdict,
            CompositeVerdict::ProvenModuloRealArithmetic.as_str(),
            "a machine-arithmetic check was recorded against a verdict this \
             disclosure has no sentence for"
        );
        "proved over the reals"
    }
}

/// The disclosure for an over-the-reals green that carries no check at all,
/// because the selected tier did not run one (`--tier smt-only`).
pub const UNCHECKED_DISCLOSURE: &str = "proved over the reals; machine arithmetic unchecked";

/// The `reason` recorded on a record whose over-the-reals green was replaced by
/// a machine counterexample, so the failure says why a proof did not prevent it.
pub const REFUTED_REASON: &str = "proved over the reals, refuted in machine arithmetic: the Tier B decision lowers \
     floating point to the reals, so it does not rule out this counterexample (chelis#3634)";

/// The `reason` recorded on a record whose over-the-reals disproof was
/// CONFIRMED at a concrete machine input (chelis#3971). It states only what
/// was checked: the solver's model is the provenance of the input -- it is why
/// this point was tried -- while the two verified facts are that the
/// property's own preconditions admit the rounded value and that the body is
/// false there.
pub const CONFIRMED_REASON: &str = "an over-the-reals disproof located this input: the property's own preconditions \
     admit it and the property body is false there in machine arithmetic \
     (chelis#3971)";

/// The `reason` recorded when the solver's witness is an irrational algebraic
/// number. This is the chelis#3971 headline shape: `1/((x*x) - 2)` is disproved
/// finite at `x = sqrt(2)`, and no binary64 value squares to exactly 2.
pub const IRRATIONAL_WITNESS_REASON: &str = "the solver's witness is an irrational algebraic number, which no machine \
     value equals, so it is not a machine counterexample (chelis#3971)";

/// The `reason` recorded when the rounded witness is not admitted by the
/// property's own preconditions. Rounding can move a value out of the guard
/// region that the exact real satisfied, and a point the guards reject is not
/// a counterexample to the guarded property.
pub const WITNESS_REJECTED_BY_GUARDS_REASON: &str = "the solver's witness does not satisfy the property's own preconditions at its \
     declared machine precision, so it is not a machine counterexample \
     (chelis#3971)";

/// The `reason` recorded when the rounded witness is admitted and the property
/// body HOLDS there.
pub const WITNESS_BODY_HOLDS_REASON: &str = "the property body holds at the solver's witness in machine arithmetic, so the \
     witness is not a machine counterexample (chelis#3971)";

/// The `reason` recorded on the record that reports in place of an
/// over-the-reals disproof whose witness machine arithmetic did not confirm
/// (chelis#3971).
///
/// Such a record may be neither a failure nor a pass, and the reason is what
/// says why. The solver decided the property FALSE over the reals, so there is
/// no basis for a pass; and the witness check establishes only that the ONE
/// point the solver exhibited is not a machine counterexample, which is a fact
/// about that point and not about the domain, so there is no basis for a
/// failure either. The machine-arithmetic question is undecided, which is what
/// `unsupported` states.
///
/// The step this wording exists to prevent is the one chelis#3971's first
/// implementation took: reading "no machine value equals this witness" as "no
/// machine value falsifies this property". `(x * x) != 11.0f64` on `[3, 4]` is
/// the counterexample to that reading -- cvc5's witness is the irrational
/// `sqrt(11)`, and the nearest double squares to exactly `11.0`, so the
/// property is false at a binary64 input its own guards admit.
pub fn unconfirmed_disproof_reason(witness_reason: &str) -> String {
    format!(
        "disproved over the reals, and no machine counterexample established: {witness_reason}. \
         The over-the-reals decision rules out no machine input, and the witness check \
         establishes nothing about inputs other than the witness, so machine arithmetic is \
         undecided here (chelis#3971)"
    )
}

/// Whether an over-the-reals DISPROOF owes its witness a machine-arithmetic
/// confirmation before it is reported as a failure (chelis#3971).
///
/// A Tier B disproof is decided over the reals, so its witness is an exact real
/// point, not a machine input. Reporting it as a property failure claims a
/// defect at some machine input, which the solver has not exhibited: an exact
/// rational, and an irrational algebraic number even more so, may have no
/// binary64 value at which the property is false.
///
/// Unlike the green side's refutation search, the confirmation is a SINGLE
/// concrete evaluation of the solver's own witness, not a second tier. It
/// therefore runs under `--tier smt-only` as well -- the chelis#3635
/// domain-witness confirmation already does, for the same reason -- and does
/// not widen an explicitly selected tier. What the two tiers differ on is only
/// how much is attempted when the witness is NOT confirmed: `auto` has a Tier C
/// to route to, `smt-only` is terminal. Neither may report a pass there; see
/// [`unconfirmed_disproof_reason`].
///
/// The `axiom` clause mirrors the green side's, for the same reason: an asserted
/// axiom is a user-declared assumption the concrete evaluator does not honor,
/// so the evaluation would be deciding the axiom rather than the witness.
///
/// On this side it is DEFENSIVE rather than load-bearing, and the difference is
/// worth stating because the green side's identical clause is not. The outcome
/// it describes -- "a disproof conditional on an asserted axiom carries no
/// check" -- is produced upstream: `prove_surf_property` routes a property
/// citing `clarabel.qp.ideal_optimality` or `clarabel.qp.assume_psd` to
/// `qp_ideal::prove` and returns, so no axiom-qualified property reaches Tier B
/// here at all. The clause also cannot fire on its own terms: both call sites
/// build the hedged disproof with no assumptions, and a
/// `DisprovedModuloRealArithmetic` base derives its disclosed qualifiers purely
/// from that assumption list, which the caller appends only after this decision
/// is made. The green side differs because its lane runs on an outcome whose
/// assumptions are already attached.
///
/// It is kept because the live mechanism is an early return somewhere else: if
/// that routing ever changes, this reads the published caveats and still
/// declines.
pub fn disproof_confirmation_required(tier: &str, disclosed: &[&str]) -> bool {
    (tier == "auto" || tier == "smt-only")
        && disclosed.contains(&crate::discharge::Qualifier::RealArith.as_str())
        && !disclosed.contains(&crate::discharge::Qualifier::Axiom.as_str())
}

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

    // --- chelis#3971: the disproof side of the same lane ---

    const DISPROVED: &str = "disproved_modulo_real_arithmetic";

    // The confirmation is ONE concrete evaluation of the solver's own witness,
    // not a second tier, so unlike the green side's refutation search it is
    // owed under `smt-only` as well. The terminal engine lanes are still never
    // widened, and `fuzz-only` never reached Tier B at all.
    #[test]
    fn disproof_confirmation_is_owed_by_auto_and_smt_only() {
        let over_reals = ["real_arithmetic"];
        for tier in ["auto", "smt-only"] {
            assert!(
                disproof_confirmation_required(tier, &over_reals),
                "{tier} reports the disproof, so it owes the witness a check"
            );
        }
        for tier in ["fuzz-only", "induction-only", "beacon-only"] {
            assert!(
                !disproof_confirmation_required(tier, &over_reals),
                "{tier} is a terminal lane of its own"
            );
        }
        assert!(
            !disproof_confirmation_required("auto", &["fuzz_base"]),
            "a disproof that is not over the reals is already a definite failure"
        );
        assert!(
            !disproof_confirmation_required("auto", &["axiom", "real_arithmetic"]),
            "a disproof under a user-asserted axiom would be checked against an \
             evaluator that does not honor the axiom"
        );
    }

    // The disclosure names the DECISION the check was run against, read off the
    // recorded verdict token. Without that, the disproof side inherits the green
    // side's "proved over the reals" sentence, which states the opposite of what
    // happened.
    #[test]
    fn the_disclosure_names_which_side_of_the_lane_it_came_from() {
        assert_eq!(
            MachineArithmeticCheck::confirmed(DISPROVED).disclosure(),
            "disproved over the reals; confirmed at a machine input"
        );
        assert_eq!(
            MachineArithmeticCheck::unconfirmed(DISPROVED, "no machine value equals it")
                .disclosure(),
            "disproved over the reals; not confirmed in machine arithmetic: no machine value equals it"
        );
        assert_eq!(
            MachineArithmeticCheck::unchecked(0, DISPROVED, "unreadable model").disclosure(),
            "disproved over the reals; machine arithmetic unchecked: unreadable model"
        );
        // Same `Unchecked` status, opposite side: the shared variant must not
        // describe a proof as a disproof or the reverse.
        assert_eq!(
            MachineArithmeticCheck::unchecked(0, "proven_modulo_real_arithmetic", "starved")
                .disclosure(),
            "proved over the reals; machine arithmetic unchecked: starved"
        );
        assert_eq!(
            MachineArithmeticCheck::corroborated(100, "proven_modulo_real_arithmetic").disclosure(),
            "proved over the reals; machine arithmetic corroborated on 100 samples"
        );
    }

    // A confirmation and an unconfirmation take no samples: the disproof side
    // evaluates the solver's exhibited point once. A nonzero count here would
    // read as a sampling result the check never ran.
    #[test]
    fn the_disproof_side_records_no_samples() {
        assert_eq!(MachineArithmeticCheck::confirmed(DISPROVED).samples, 0);
        assert_eq!(
            MachineArithmeticCheck::unconfirmed(DISPROVED, "reason").samples,
            0
        );
        assert_eq!(
            MachineArithmeticCheck::confirmed(DISPROVED).reason,
            None,
            "a confirmation's reason lives on the record, not on the check"
        );
    }

    // Which outcomes a tier-explanation consumer must treat as the lane's
    // choice rather than as a fall-back. Both displacing statuses REPLACE the
    // Tier B record; the three others leave it where it was, so they explain
    // nothing about the reported tier.
    #[test]
    fn only_a_displacement_explains_the_reported_tier() {
        assert!(
            MachineArithmeticCheck::refuted(100, "proven_modulo_real_arithmetic")
                .displaced_the_tier_b_decision()
        );
        assert!(
            MachineArithmeticCheck::unconfirmed(DISPROVED, "no machine value equals it")
                .displaced_the_tier_b_decision()
        );
        assert!(
            !MachineArithmeticCheck::corroborated(100, "proven_modulo_real_arithmetic")
                .displaced_the_tier_b_decision()
        );
        assert!(!MachineArithmeticCheck::confirmed(DISPROVED).displaced_the_tier_b_decision());
        for verdict in ["proven_modulo_real_arithmetic", DISPROVED] {
            assert!(
                !MachineArithmeticCheck::unchecked(0, verdict, "nothing established")
                    .displaced_the_tier_b_decision(),
                "{verdict}: a lane that established nothing left the record alone"
            );
        }
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
