//! The discharge-engine registry and fitness-based dispatcher (WI-9).
//!
//! This is the Wave 2 routing layer that sits on top of the WI-3/WI-4 discharge
//! seam ([`crate::discharge`]). It replaces hardcoded engine selection with a
//! registry of [`DischargeEngine`]s and a deterministic, fitness-based dispatch:
//! given a [`Goal`], the registry selects the first registered engine whose
//! [`DischargeEngine::fitness`] returns true and discharges through it.
//!
//! # The Beacon rail (`docs/design/phase2_seam_contract.md` §5)
//!
//! The registry exposes a PUBLIC registration entry point ([`DischargeRegistry::register`])
//! so an OUT-OF-TREE shell engine (Beacon) can register itself without touching
//! this crate. This is the reason the registry stores boxed trait objects
//! ([`Box<dyn DischargeEngine>`]) rather than a closed enum: an out-of-tree
//! engine cannot be a variant of an in-tree enum, but it can be boxed behind the
//! trait. Beacon's interval engine claims [`GoalShape::BoxRange`] and registers
//! ahead of any weaker fallback so the box/range lane routes to it.
//!
//! # Deterministic selection and tie-break
//!
//! Selection scans the registered engines in REGISTRATION ORDER and takes the
//! FIRST whose `fitness(goal)` is true. Registration order IS the priority: a
//! caller that wants engine A preferred over engine B for an overlapping goal
//! shape registers A first. This makes the tie-break explicit and registerable
//! rather than accidental: two engines that both fit a goal resolve to the
//! earlier-registered one, deterministically, on every dispatch.
//!
//! # The no-fit path (honesty-critical)
//!
//! A goal that NO registered engine fits never silently passes and never
//! greens. [`DischargeRegistry::dispatch`] returns the canonical no-fit
//! [`Discharge`]: [`Soundness::Untrusted`] with an EMPTY [`QualifierSet`] and a
//! [`TierBResult::Error`] result. This is byte-identical in shape to what a
//! mismatched in-tree engine already returns (e.g. cvc5 handed a `BoxRange`
//! goal), and it projects through the WI-6 verdict algebra to
//! [`crate::composition::CompositeVerdict::Unsupported`] (an `Unestablished`
//! terminal), never `Proven`. No new soundness or verdict variant is
//! introduced; the existing bottom of the lattice carries the no-fit outcome.
//!
//! # Dual-lane preservation (solver-free default build)
//!
//! [`DischargeRegistry::with_builtin_engines`] registers the in-tree engines
//! PER FEATURE so that neither feature lane changes behavior for existing goals:
//!
//! - Under `--features smt` it registers [`crate::discharge::Cvc5Engine`], which
//!   wraps the same `solve_property` pipeline, so an SMT goal yields the
//!   identical [`TierBResult`] as the pre-WI-9 hardcoded cvc5 call.
//! - In the DEFAULT (non-smt) build there is no cvc5 engine (it is
//!   `#[cfg(feature = "smt")]`-gated, keeping the build solver-free). To preserve
//!   the non-smt SMT-goal path -- which today calls [`crate::tier_b::solve_property`]
//!   DIRECTLY rather than going through any engine -- the registry instead
//!   registers [`SolvePropertyEngine`], a solver-free SMT engine that wraps that
//!   exact same `solve_property` call. `solve_property` is unconditional and is
//!   itself solver-free in the non-smt build (its cvc5 branch is feature-gated
//!   away), so this introduces NO cvc5-named symbol into the default binary and
//!   returns the byte-identical `TierBResult` the direct call returned.

use crate::beacon_shim::{BeaconShim, WireDagByteStore};
use crate::discharge::{Discharge, DischargeEngine, Goal, QualifierSet, Soundness};
use crate::tier_b::TierBResult;

/// A registry of [`DischargeEngine`]s with fitness-based, deterministic
/// dispatch. See the module docs for the selection rule, the no-fit path, and
/// the dual-lane preservation guarantee.
#[derive(Default)]
pub struct DischargeRegistry {
    engines: Vec<Box<dyn DischargeEngine>>,
}

impl DischargeRegistry {
    /// An empty registry with no engines. A goal dispatched against an empty
    /// registry takes the no-fit path (see [`Self::dispatch`]).
    pub fn new() -> Self {
        Self {
            engines: Vec::new(),
        }
    }

    /// A registry pre-populated with the in-tree engines for the active feature
    /// lane. This is the constructor the in-tree dispatch sites use; it
    /// preserves both feature lanes byte-identically (see the module docs):
    ///
    /// - `--features smt`: registers [`crate::discharge::Cvc5Engine`].
    /// - default (non-smt): registers [`SolvePropertyEngine`] (solver-free).
    ///
    /// An out-of-tree consumer that wants the box/range lane (Beacon) starts
    /// from here and [`register`](Self::register)s its own engine on top.
    pub fn with_builtin_engines() -> Self {
        let mut registry = Self::new();
        // WI-15: the Clarabel SoS certificate engine registers AHEAD of cvc5 so a
        // univariate-poly-nonneg-on-interval goal is tried by Clarabel FIRST (its
        // narrow fitness matches exactly that subset). On a verified exact
        // certificate it returns a definite `Proved` -- the try-until-discharge
        // dispatcher returns it immediately at `CertificateBearing@Exact`. On
        // `Unknown` (no certificate / failed repair / boundary degeneracy) it is a
        // non-verdict, so the dispatcher FALLS THROUGH to cvc5, whose NRA may
        // still decide the goal. Only registered when the Clarabel SDP proposer is
        // present (the targets with the `sdp` backend wired); elsewhere there is
        // no production proposer, so registering would only add an always-Unknown
        // fall-through hop.
        #[cfg(all(feature = "clarabel", any(target_os = "linux", target_os = "macos")))]
        registry.register(Box::new(crate::clarabel_sos::ClarabelSosEngine::clarabel()));
        #[cfg(feature = "smt")]
        registry.register(Box::new(crate::discharge::Cvc5Engine::new()));
        #[cfg(not(feature = "smt"))]
        registry.register(Box::new(SolvePropertyEngine::new()));
        registry
    }

    /// Extend this registry with the Beacon subprocess shim, using the given
    /// content-addressed byte store. The shim is registered only if a binary
    /// is discoverable (explicit path or `CHELIS_BEACON_BIN` env). If no binary
    /// is found, the registry is unchanged and BoxRange goals take the existing
    /// no-fit path.
    pub fn with_beacon(mut self, store: WireDagByteStore) -> Self {
        if let Some(shim) = BeaconShim::from_env(store) {
            self.register(Box::new(shim));
        }
        self
    }

    /// Register an engine. PUBLIC entry point for out-of-tree engines (the
    /// Beacon rail, `docs/design/phase2_seam_contract.md` §5).
    ///
    /// Registration order is priority: an engine registered earlier is selected
    /// ahead of a later one when both fit a goal (see [`Self::dispatch`]). A
    /// caller that needs its engine preferred registers it before any
    /// overlapping fallback.
    pub fn register(&mut self, engine: Box<dyn DischargeEngine>) {
        self.engines.push(engine);
    }

    /// The number of registered engines.
    pub fn len(&self) -> usize {
        self.engines.len()
    }

    /// Whether the registry has no engines.
    pub fn is_empty(&self) -> bool {
        self.engines.is_empty()
    }

    /// Select the first registered engine whose `fitness(goal)` is true, in
    /// registration order. `None` if no engine fits. This is the deterministic
    /// selection / tie-break primitive; [`Self::dispatch`] uses it.
    fn select(&self, goal: &Goal) -> Option<&dyn DischargeEngine> {
        self.engines
            .iter()
            .find(|engine| engine.fitness(goal))
            .map(AsRef::as_ref)
    }

    /// The stable name of the engine that WOULD be selected for `goal`, or
    /// `None` if no engine fits. Diagnostic surface; does not discharge.
    ///
    /// This names the FIRST fitting engine in registration order -- the engine
    /// [`Self::dispatch`] tries first. With try-until-discharge a LATER fitting
    /// engine may actually produce the returned verdict (if the first returns a
    /// non-verdict and the dispatcher falls through); the engine that did so is
    /// recorded in the returned discharge's evidence, not here.
    pub fn selected_engine_name(&self, goal: &Goal) -> Option<&'static str> {
        self.select(goal).map(DischargeEngine::name)
    }

    /// Discharge `goal` with TRY-UNTIL-DISCHARGE over the fitting engines, in
    /// registration order (WS-5).
    ///
    /// The dispatcher tries each fitting engine in turn:
    ///
    /// - A DEFINITE verdict ([`TierBResult::Proved`] / [`TierBResult::Disproved`])
    ///   is returned IMMEDIATELY, with that engine's OWN `(soundness, qualifier)`
    ///   untouched. The dispatcher NEVER falls through past a definite verdict,
    ///   and a later engine NEVER upgrades or launders an earlier engine's
    ///   result -- each discharge carries exactly the guarantee its own engine
    ///   attached.
    /// - A NON-VERDICT ([`TierBResult::Timeout`] / [`TierBResult::Unknown`] /
    ///   [`TierBResult::Error`]) -- the engine fit the goal's SHAPE but could not
    ///   DISCHARGE it -- falls through to the NEXT fitting engine. This is the
    ///   cvc5 -> Z3 capability-split consumer: a goal cvc5 returns `Unknown` on
    ///   can be discharged by Z3 (or vice versa), so a non-verdict is not the
    ///   end of the line while another fitting engine remains.
    ///
    /// If NO engine fits the goal shape at all, the canonical
    /// [`no_fit_discharge`] is returned (the seam-contract §5 honesty floor). If
    /// fitting engines exist but EVERY one returns a non-verdict, the result is
    /// [`exhausted_discharge`]: still [`Soundness::Untrusted`] + empty
    /// [`QualifierSet`] + [`TierBResult::Error`] -- byte-identical in lattice
    /// membership to no-fit, projecting to
    /// [`crate::composition::CompositeVerdict::Unsupported`], never a green --
    /// but with an honest reason that fitting engines were tried and none
    /// produced a verdict (rather than the misleading "no engine fits").
    ///
    /// The laundering guard is structural: only a non-verdict ever falls
    /// through, and a non-verdict is always `Untrusted`/empty under
    /// [`crate::discharge::classify_smt_outcome`], so the exhausted result can
    /// only ever be `Untrusted` -- it can never carry a badge a fallen-through
    /// engine did not earn.
    pub fn dispatch(&self, goal: &Goal, timeout_ms: u64) -> Discharge {
        let mut any_fit = false;
        let mut last_non_verdict: Option<Discharge> = None;
        for engine in &self.engines {
            if !engine.fitness(goal) {
                continue;
            }
            any_fit = true;
            let discharge = engine.discharge(goal, timeout_ms);
            if is_definite_verdict(discharge.result()) {
                // Definite verdict: return it as-is, never fall through past it.
                return discharge;
            }
            // Non-verdict: remember it (honest reason) and try the next fitting
            // engine. It is Untrusted/empty by classification, so keeping it
            // cannot launder a badge.
            last_non_verdict = Some(discharge);
        }
        if !any_fit {
            return no_fit_discharge(goal);
        }
        // Fitting engines existed but none discharged a definite verdict.
        exhausted_discharge(goal, last_non_verdict.as_ref())
    }
}

/// Whether a tier-B outcome is a DEFINITE verdict the dispatcher must NOT fall
/// through past: a [`TierBResult::Proved`] or [`TierBResult::Disproved`]. A
/// timeout/unknown/error is a non-verdict (the engine fit the shape but could
/// not discharge), which the dispatcher falls through.
fn is_definite_verdict(result: &TierBResult) -> bool {
    matches!(result, TierBResult::Proved | TierBResult::Disproved(_))
}

/// The discharge for a goal whose fitting engines were ALL exhausted without a
/// definite verdict (WS-5 try-until-discharge). Like [`no_fit_discharge`] it is
/// the honesty floor -- [`Soundness::Untrusted`], an empty [`QualifierSet`], a
/// [`TierBResult::Error`], projecting to
/// [`crate::composition::CompositeVerdict::Unsupported`] -- but its reason says
/// the fitting engines were tried and none produced a verdict, which is what
/// actually happened (no engine MISFIT; they all declined to decide). The
/// last-tried engine's own non-verdict result is preserved as the carried
/// result when available, so its honest reason (a timeout vs an unknown vs a
/// lowering error) is not discarded; otherwise a synthesized Error is used.
pub fn exhausted_discharge(goal: &Goal, last_non_verdict: Option<&Discharge>) -> Discharge {
    let reason = format!(
        "all fitting discharge engines were exhausted without a verdict for goal shape `{}`",
        goal_shape_label(goal)
    );
    // Preserve the last engine's honest non-verdict result (and a note of which
    // engine it came from) when one is available; otherwise synthesize an
    // Error. Either way the lattice membership is Untrusted/empty/Error.
    // chelis#496: the canonical attribution is the top-level `"engine"` key
    // (here the dispatcher pseudo-source `exhausted_fallthrough`); the reason and
    // the last engine's own evidence live under `"backend"`.
    let (result, backend) = match last_non_verdict {
        Some(d) => (
            d.result().clone(),
            serde_json::json!({
                "reason": reason,
                "last_engine_evidence": d.evidence().clone(),
            }),
        ),
        None => (
            TierBResult::Error(reason.clone()),
            serde_json::json!({ "reason": reason }),
        ),
    };
    Discharge::with_engine_attribution(
        "exhausted_fallthrough",
        Soundness::Untrusted,
        QualifierSet::new(),
        result,
        backend,
    )
    .expect("untrusted discharge with an empty qualifier set is always valid")
}

/// Build the canonical no-fit [`Discharge`] for a goal that no registered engine
/// fits: [`Soundness::Untrusted`], an empty [`QualifierSet`], and a
/// [`TierBResult::Error`] explaining the miss.
///
/// This is the honesty floor. It is constructed through [`Discharge::new`] (so
/// the integrity invariant holds: an untrusted discharge with an empty qualifier
/// set is always valid) and is byte-identical in lattice membership to the
/// outcome a mismatched in-tree engine already returns. It never carries a
/// proof qualifier and never claims any soundness above the bottom of the
/// lattice.
pub fn no_fit_discharge(goal: &Goal) -> Discharge {
    let reason = format!(
        "no registered discharge engine fits goal shape `{}`",
        goal_shape_label(goal)
    );
    // chelis#496: the canonical attribution is the top-level `"engine"` key
    // (the dispatcher pseudo-source `no_fit`); the reason lives under `"backend"`.
    Discharge::with_engine_attribution(
        "no_fit",
        Soundness::Untrusted,
        QualifierSet::new(),
        TierBResult::Error(reason.clone()),
        serde_json::json!({ "reason": reason }),
    )
    .expect("untrusted discharge with an empty qualifier set is always valid")
}

/// A short, stable label for a goal's shape, for the no-fit diagnostic.
fn goal_shape_label(goal: &Goal) -> &'static str {
    match &goal.shape {
        crate::discharge::GoalShape::Smt(_) => "smt",
        crate::discharge::GoalShape::BoxRange { .. } => "box_range",
    }
}

/// A solver-free SMT [`DischargeEngine`] that wraps [`crate::tier_b::solve_property`].
///
/// This is the DEFAULT (non-smt) build's SMT engine. It exists ONLY when the
/// `smt` feature is OFF: under `--features smt` the registry uses
/// [`crate::discharge::Cvc5Engine`] instead. Its sole job is to preserve the
/// pre-WI-9 non-smt SMT-goal path -- which called `solve_property` directly --
/// byte-identically while still flowing through the uniform registry dispatch.
///
/// It carries no cvc5-named symbol: `solve_property` is itself solver-free in
/// the non-smt build, so this keeps the default binary solver-free (the
/// `check_is_solver_free_on_the_corpus` gate).
#[cfg(not(feature = "smt"))]
#[derive(Debug, Clone, Copy, Default)]
pub struct SolvePropertyEngine;

#[cfg(not(feature = "smt"))]
impl SolvePropertyEngine {
    pub const fn new() -> Self {
        Self
    }

    /// Map a tier-B outcome to its `(soundness, qualifier_set)` through the
    /// shared, engine-agnostic [`crate::discharge::classify_smt_outcome`]: the
    /// decision is over the REALS, so a proved/disproved result is
    /// `SoundApproximate` carrying [`crate::discharge::Qualifier::RealArith`]
    /// (chelis#422), not exact; a timeout/unknown/error is untrusted with no
    /// qualifier. In the non-smt build `solve_property` only ever returns
    /// `Timeout` (or a test-forced result), so in practice this yields the
    /// untrusted branch.
    fn classify(result: &TierBResult) -> (Soundness, QualifierSet) {
        crate::discharge::classify_smt_outcome(result)
    }
}

#[cfg(not(feature = "smt"))]
impl DischargeEngine for SolvePropertyEngine {
    fn name(&self) -> &'static str {
        // Deliberately NOT a cvc5 name: the default build is solver-free.
        "solve_property"
    }

    fn fitness(&self, goal: &Goal) -> bool {
        matches!(goal.shape, crate::discharge::GoalShape::Smt(_))
    }

    fn discharge(&self, goal: &Goal, timeout_ms: u64) -> Discharge {
        let result = match goal.as_smt() {
            Some(property) => crate::tier_b::solve_property(property, timeout_ms),
            None => TierBResult::Error(
                "solve_property engine cannot discharge a non-SMT goal shape".to_string(),
            ),
        };
        let (soundness, qualifier_set) = Self::classify(&result);
        // chelis#496: canonical top-level `"engine"` attribution; no extra
        // backend detail on the normal path.
        Discharge::with_engine_attribution(
            "solve_property",
            soundness,
            qualifier_set,
            result,
            serde_json::Value::Null,
        )
        .unwrap_or_else(|err| {
            Discharge::with_engine_attribution(
                "solve_property",
                Soundness::Untrusted,
                QualifierSet::new(),
                TierBResult::Error(err.to_string()),
                serde_json::json!({ "internal_error": err.to_string() }),
            )
            .expect("untrusted discharge with empty qualifier set is always valid")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::composition::{
        AssumptionDischarge, AssumptionRecord, CompositeVerdict, DischargeMethod, NonVacuityRecord,
        rollup_composite,
    };
    use crate::discharge::{GoalShape, IntervalBox, OutputRange, Qualifier};
    use crate::solver::{CmpOp, SmtExpr, SmtSort};
    use crate::tier_b::SmtProperty;

    fn trivially_true_property() -> SmtProperty {
        // forall x: Real . x == x
        SmtProperty {
            variables: vec![("x".to_string(), SmtSort::Real)],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Eq,
                Box::new(SmtExpr::Var("x".to_string())),
                Box::new(SmtExpr::Var("x".to_string())),
            ),
        }
    }

    fn smt_goal() -> Goal {
        Goal::smt(trivially_true_property())
    }

    fn box_range_goal() -> Goal {
        Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 0.0, 100.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 50.0,
            },
        )
        .expect("well-formed box goal")
    }

    /// A test-only engine that claims a goal shape, so the external-engine rail
    /// can be exercised in-tree WITHOUT Beacon: `BoxRange` has no real in-tree
    /// engine, so a mock proves the registration path end to end.
    #[derive(Debug)]
    struct MockEngine {
        name: &'static str,
        fits_shape: WhichShape,
        soundness: Soundness,
        qualifiers: QualifierSet,
        result: TierBResult,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum WhichShape {
        Smt,
        BoxRange,
    }

    impl MockEngine {
        /// A sound-over-approximation BoxRange engine (Beacon's stand-in): a
        /// `SoundApproximate` discharge carrying `SoundOverApproximation`.
        fn box_range_sound(name: &'static str) -> Self {
            Self {
                name,
                fits_shape: WhichShape::BoxRange,
                soundness: Soundness::SoundApproximate,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::SoundOverApproximation]),
                result: TierBResult::Proved,
            }
        }

        /// An SMT-fitting engine returning a given non-verdict (Timeout /
        /// Unknown / Error). Classified Untrusted/empty by
        /// [`crate::discharge::classify_smt_outcome`] -- the dispatcher falls
        /// through it (WS-5 try-until-discharge).
        fn smt_non_verdict(name: &'static str, result: TierBResult) -> Self {
            debug_assert!(
                !matches!(result, TierBResult::Proved | TierBResult::Disproved(_)),
                "smt_non_verdict must carry a NON-verdict result"
            );
            Self {
                name,
                fits_shape: WhichShape::Smt,
                soundness: Soundness::Untrusted,
                qualifiers: QualifierSet::new(),
                result,
            }
        }

        /// An SMT-fitting engine returning a DEFINITE over-reals verdict
        /// (`Proved` / `Disproved`), classified SoundApproximate + RealArith,
        /// mirroring how a real SMT engine (cvc5 / Z3) classifies it. The
        /// dispatcher must NOT fall through past this.
        fn smt_real_arith_verdict(name: &'static str, result: TierBResult) -> Self {
            debug_assert!(
                matches!(result, TierBResult::Proved | TierBResult::Disproved(_)),
                "smt_real_arith_verdict must carry a DEFINITE verdict result"
            );
            Self {
                name,
                fits_shape: WhichShape::Smt,
                soundness: Soundness::SoundApproximate,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::RealArith]),
                result,
            }
        }
    }

    impl DischargeEngine for MockEngine {
        fn name(&self) -> &'static str {
            self.name
        }

        fn fitness(&self, goal: &Goal) -> bool {
            matches!(
                (self.fits_shape, &goal.shape),
                (WhichShape::Smt, GoalShape::Smt(_))
                    | (WhichShape::BoxRange, GoalShape::BoxRange { .. })
            )
        }

        fn discharge(&self, _goal: &Goal, _timeout_ms: u64) -> Discharge {
            // chelis#496: route through the canonical attribution helper too, so
            // the mock cannot drift from the convention it exists to exercise.
            Discharge::with_engine_attribution(
                self.name,
                self.soundness,
                self.qualifiers.clone(),
                self.result.clone(),
                serde_json::Value::Null,
            )
            .expect("mock discharge must satisfy the integrity invariant")
        }
    }

    // --- the external-engine rail works WITHOUT Beacon (in-tree mock) ---

    #[test]
    fn registered_mock_engine_is_selected_for_its_box_range_goal() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::box_range_sound("mock_interval")));

        let goal = box_range_goal();
        assert_eq!(
            registry.selected_engine_name(&goal),
            Some("mock_interval"),
            "the registered BoxRange engine must be selected for a BoxRange goal"
        );
        let discharge = registry.dispatch(&goal, 1_000);
        assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
        assert!(
            discharge
                .qualifier_set()
                .contains(Qualifier::SoundOverApproximation)
        );
        assert_eq!(*discharge.result(), TierBResult::Proved);
    }

    // chelis#422 / WI-B8 seam: a Beacon-shaped discharge -- `SoundApproximate`
    // carrying `SoundOverApproximation` -- must project to the `sound_approximate`
    // verdict through the SAME door the consumer seam uses
    // (`base_verdict_from_discharge`), never `proven`. This is what makes a
    // future out-of-tree Beacon interval discharge reach `composite_verdict` as
    // `sound_approximate` rather than being flattened to a proof.
    #[test]
    fn beacon_shaped_discharge_projects_to_sound_approximate_not_proven() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::box_range_sound("mock_interval")));
        let discharge = registry.dispatch(&box_range_goal(), 1_000);
        let verdict = crate::composition::base_verdict_from_discharge(
            discharge.soundness(),
            discharge.qualifier_set(),
        );
        assert_eq!(
            verdict,
            crate::composition::CompositeVerdict::SoundApproximate,
            "a Beacon interval discharge discloses `sound_approximate`, never a proof"
        );
        assert_ne!(verdict, crate::composition::CompositeVerdict::Proven);
        assert_ne!(
            verdict,
            crate::composition::CompositeVerdict::ProvenModuloRealArithmetic
        );
        let qualifiers = crate::composition::composed_qualifier_strings(
            discharge.soundness(),
            discharge.qualifier_set(),
            &[],
        );
        assert_eq!(qualifiers, vec!["sound_over_approximation"]);
    }

    // --- no-fit: never a silent pass, never a green ---

    #[test]
    fn empty_registry_yields_no_fit_unsupported_discharge() {
        let registry = DischargeRegistry::new();
        let goal = box_range_goal();
        assert!(registry.is_empty());
        assert_eq!(registry.selected_engine_name(&goal), None);

        let discharge = registry.dispatch(&goal, 1_000);
        assert_no_fit_lattice_membership(&discharge);
    }

    #[test]
    fn box_range_goal_with_only_smt_engine_is_no_fit() {
        // Only an SMT-fitting engine is registered; a BoxRange goal has no fit
        // and must take the no-fit path, not be coerced onto the SMT engine.
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine {
            name: "mock_smt",
            fits_shape: WhichShape::Smt,
            soundness: Soundness::Exact,
            qualifiers: QualifierSet::from_iter_kinds([Qualifier::Exact]),
            result: TierBResult::Proved,
        }));

        let goal = box_range_goal();
        assert_eq!(registry.selected_engine_name(&goal), None);
        let discharge = registry.dispatch(&goal, 1_000);
        assert_no_fit_lattice_membership(&discharge);
    }

    /// Pin the EXACT no-fit lattice membership the brief and the seam contract
    /// (§5) require: `Soundness::Untrusted` + empty `QualifierSet` +
    /// `TierBResult::Error`, projecting to `CompositeVerdict::Unsupported`
    /// (never `Proven`, never a silent pass).
    fn assert_no_fit_lattice_membership(discharge: &Discharge) {
        assert_eq!(
            discharge.soundness(),
            Soundness::Untrusted,
            "no-fit must be untrusted (bottom of the lattice)"
        );
        assert!(
            discharge.qualifier_set().is_empty(),
            "no-fit must carry no qualifier (no laundered badge)"
        );
        assert!(
            matches!(discharge.result(), TierBResult::Error(_)),
            "no-fit must be an Error result, never Proved/Disproved"
        );

        // Project through the WI-6 verdict algebra. The no-fit discharge's
        // outcome is an `unsupported` status (an untrusted, non-green result);
        // even composed onto an otherwise-PROVEN base it degrades the composite
        // to `Unsupported` and never launders into `Proven`.
        let verdict = composite_with_no_fit_dependency(discharge);
        assert_eq!(
            verdict,
            CompositeVerdict::Unsupported,
            "the no-fit discharge must render Unsupported"
        );
        assert_ne!(
            verdict,
            CompositeVerdict::Proven,
            "the no-fit discharge must never render Proven, even over a proven base"
        );
    }

    /// Roll a Proven base together with one assumption whose discharge mirrors
    /// the no-fit outcome (status derived from the discharge, not hardcoded), to
    /// show the no-fit result cannot be laundered into a green composite.
    fn composite_with_no_fit_dependency(no_fit: &Discharge) -> CompositeVerdict {
        // Derive the status the prove flow assigns to this discharge: an
        // untrusted, non-green result is `unsupported`. This is NOT hardcoded to
        // the literal "unsupported" -- it is computed from the discharge so the
        // test exercises the discharge -> status mapping.
        let status = discharge_status(no_fit);
        assert_eq!(
            status, "unsupported",
            "a no-fit (untrusted/empty/Error) discharge must map to the `unsupported` status"
        );
        let record = AssumptionRecord::new(
            "no_fit_dependency",
            Some(AssumptionDischarge::new(
                DischargeMethod::Smt,
                serde_json::json!({ "status": status }),
            )),
            Some(NonVacuityRecord::established(
                serde_json::json!({ "result": "sat" }),
            )),
        );
        rollup_composite(CompositeVerdict::Proven, &[record])
    }

    /// The prove-flow status a discharge maps to, derived from its result. A
    /// determinate green/refutation result is `proved`/`failed`; everything
    /// else (an untrusted/non-green discharge, i.e. the no-fit case) is
    /// `unsupported`. chelis#422: a cvc5 decision is over the reals, so a
    /// proved/disproved result is `SoundApproximate` (not `Exact`); the status
    /// keys off the result, not the soundness tier.
    fn discharge_status(discharge: &Discharge) -> &'static str {
        match (discharge.soundness(), discharge.result()) {
            (Soundness::Untrusted, _) => "unsupported",
            (_, TierBResult::Proved) => "proved",
            (_, TierBResult::Disproved(_)) => "failed",
            _ => "unsupported",
        }
    }

    // --- deterministic selection / tie-break (registration order) ---

    #[test]
    fn earlier_registered_engine_wins_when_two_fit() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::box_range_sound("first")));
        registry.register(Box::new(MockEngine::box_range_sound("second")));

        let goal = box_range_goal();
        // First-registered is the deterministic winner.
        assert_eq!(registry.selected_engine_name(&goal), Some("first"));
        let discharge = registry.dispatch(&goal, 1_000);
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("first"),
            "the earlier-registered fitting engine must discharge"
        );
    }

    #[test]
    fn registration_order_is_priority_reversed_order_flips_winner() {
        // The tie-break IS registration order: registering "second" first makes
        // it the winner. Proves priority is registerable, not accidental.
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::box_range_sound("second")));
        registry.register(Box::new(MockEngine::box_range_sound("first")));

        assert_eq!(
            registry.selected_engine_name(&box_range_goal()),
            Some("second")
        );
    }

    #[test]
    fn dispatch_is_deterministic_across_repeated_calls() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::box_range_sound("first")));
        registry.register(Box::new(MockEngine::box_range_sound("second")));
        let goal = box_range_goal();
        for _ in 0..16 {
            assert_eq!(registry.selected_engine_name(&goal), Some("first"));
        }
    }

    // --- negative: registering nothing never panics and never greens ---

    #[test]
    fn empty_registry_smt_goal_is_also_no_fit_not_a_green() {
        let registry = DischargeRegistry::new();
        let discharge = registry.dispatch(&smt_goal(), 1_000);
        assert_no_fit_lattice_membership(&discharge);
    }

    // --- builtin-engine constructor: per-lane registration ---

    #[cfg(feature = "smt")]
    #[test]
    fn builtin_registry_routes_smt_goal_to_cvc5() {
        let registry = DischargeRegistry::with_builtin_engines();
        let goal = smt_goal();
        assert_eq!(registry.selected_engine_name(&goal), Some("cvc5"));
        // A trivially-true property proves through the cvc5 lane. chelis#422:
        // the cvc5 decision is over the reals, so the discharge is
        // `SoundApproximate` carrying `RealArith`, not exact.
        let discharge = registry.dispatch(&goal, 5_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
        assert!(
            discharge
                .qualifier_set()
                .contains(crate::discharge::Qualifier::RealArith)
        );
    }

    #[cfg(feature = "smt")]
    #[test]
    fn builtin_registry_box_range_goal_is_no_fit_in_smt_lane() {
        // cvc5 does not fit BoxRange; with only cvc5 registered, a BoxRange goal
        // is no-fit -> Unsupported (the seam-contract §5 guarantee).
        let registry = DischargeRegistry::with_builtin_engines();
        let discharge = registry.dispatch(&box_range_goal(), 5_000);
        assert_no_fit_lattice_membership(&discharge);
    }

    #[cfg(not(feature = "smt"))]
    #[test]
    fn builtin_registry_routes_smt_goal_to_solve_property_engine() {
        // The default (non-smt) lane: the SMT goal routes to the solver-free
        // solve_property engine, NOT to cvc5 (which is absent). solve_property
        // returns Timeout in the non-smt build, byte-identical to the pre-WI-9
        // direct call.
        let registry = DischargeRegistry::with_builtin_engines();
        let goal = smt_goal();
        assert_eq!(registry.selected_engine_name(&goal), Some("solve_property"));
        let discharge = registry.dispatch(&goal, 1_000);
        assert_eq!(
            *discharge.result(),
            TierBResult::Timeout,
            "non-smt solve_property stub returns Timeout for an SMT goal"
        );
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
    }

    #[cfg(not(feature = "smt"))]
    #[test]
    fn builtin_registry_box_range_goal_is_no_fit_in_nonsmt_lane() {
        let registry = DischargeRegistry::with_builtin_engines();
        let discharge = registry.dispatch(&box_range_goal(), 1_000);
        assert_no_fit_lattice_membership(&discharge);
    }

    // ============================================================
    // WS-5 try-until-discharge fall-through (feature-free, mock engines)
    // ============================================================

    /// The cvc5 -> Z3 capability-split consumer, in miniature: a first SMT
    /// engine returns `Unknown` (a non-verdict), so the dispatcher FALLS THROUGH
    /// to a second SMT engine that `Proved`s the goal. The returned verdict is
    /// the SECOND engine's `Proved`, carrying the SECOND engine's own
    /// `(soundness, qualifier)` -- the first engine's non-verdict does not block
    /// the discharge, and the second engine's badge is not laundered up from the
    /// first.
    #[test]
    fn falls_through_a_non_verdict_to_the_next_engine_that_discharges() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::smt_non_verdict(
            "first_unknown",
            TierBResult::Unknown,
        )));
        registry.register(Box::new(MockEngine::smt_real_arith_verdict(
            "second_proves",
            TierBResult::Proved,
        )));

        let discharge = registry.dispatch(&smt_goal(), 1_000);
        assert_eq!(
            *discharge.result(),
            TierBResult::Proved,
            "the fall-through must reach the second engine's Proved"
        );
        // The verdict carries the SECOND engine's guarantee, not the first's.
        assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
        assert!(discharge.qualifier_set().contains(Qualifier::RealArith));
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("second_proves"),
            "the discharging engine is the one that produced the verdict"
        );
    }

    /// Each of the three non-verdict kinds (Timeout / Unknown / Error) falls
    /// through. Pinned as a table so no non-verdict kind is left uncovered: a
    /// future TierBResult variant that should fall through but does not is
    /// caught here.
    #[test]
    fn every_non_verdict_kind_falls_through() {
        for first in [
            TierBResult::Timeout,
            TierBResult::Unknown,
            TierBResult::Error("first engine could not lower".to_string()),
        ] {
            let mut registry = DischargeRegistry::new();
            registry.register(Box::new(MockEngine::smt_non_verdict(
                "first",
                first.clone(),
            )));
            registry.register(Box::new(MockEngine::smt_real_arith_verdict(
                "second",
                TierBResult::Disproved(serde_json::json!({"x": "0"})),
            )));
            let discharge = registry.dispatch(&smt_goal(), 1_000);
            assert!(
                matches!(discharge.result(), TierBResult::Disproved(_)),
                "first non-verdict {first:?} must fall through to the second's Disproved"
            );
            assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
        }
    }

    /// LAUNDERING GUARD (critical): a DEFINITE verdict is NEVER fallen through.
    /// The first SMT engine `Disproved`s the goal; a second SMT engine that
    /// WOULD `Prove` it is registered AFTER. The dispatcher must stop at the
    /// first engine's Disproved and NEVER reach the second -- a later engine can
    /// never overturn (or launder) an earlier engine's definite verdict.
    #[test]
    fn never_falls_through_past_a_definite_disproved() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::smt_real_arith_verdict(
            "first_disproves",
            TierBResult::Disproved(serde_json::json!({"x": "0"})),
        )));
        // This engine would Prove the goal, but it must never be consulted.
        registry.register(Box::new(MockEngine::smt_real_arith_verdict(
            "second_would_prove",
            TierBResult::Proved,
        )));

        let discharge = registry.dispatch(&smt_goal(), 1_000);
        assert!(
            matches!(discharge.result(), TierBResult::Disproved(_)),
            "the first engine's Disproved is final; the dispatcher must not fall through to a Proved"
        );
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("first_disproves"),
            "the first definite verdict wins; the later engine is never consulted"
        );
    }

    /// The mirror of the guard: a definite `Proved` is also final and is not
    /// fallen through to a later engine that would `Disproved`.
    #[test]
    fn never_falls_through_past_a_definite_proved() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::smt_real_arith_verdict(
            "first_proves",
            TierBResult::Proved,
        )));
        registry.register(Box::new(MockEngine::smt_real_arith_verdict(
            "second_would_disprove",
            TierBResult::Disproved(serde_json::json!({"x": "0"})),
        )));

        let discharge = registry.dispatch(&smt_goal(), 1_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("first_proves")
        );
    }

    /// LAUNDERING GUARD: a WEAKER later engine never upgrades an earlier
    /// engine's result. Here the first engine returns a definite verdict, and
    /// the dispatcher returns it untouched -- the second engine (even with a
    /// stronger-looking qualifier) is never reached. This pins that the
    /// returned discharge's `(soundness, qualifier)` is EXACTLY the producing
    /// engine's, never a union or upgrade across engines.
    #[test]
    fn a_later_engine_never_upgrades_an_earlier_definite_verdict() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::smt_real_arith_verdict(
            "real_arith_first",
            TierBResult::Proved,
        )));
        // A second engine that, if (wrongly) consulted and merged, could try to
        // attach `Exact`. It must never be reached, and the result must NOT
        // carry Exact.
        registry.register(Box::new(MockEngine {
            name: "exact_second",
            fits_shape: WhichShape::Smt,
            soundness: Soundness::Exact,
            qualifiers: QualifierSet::from_iter_kinds([Qualifier::Exact]),
            result: TierBResult::Proved,
        }));

        let discharge = registry.dispatch(&smt_goal(), 1_000);
        assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
        assert!(discharge.qualifier_set().contains(Qualifier::RealArith));
        assert!(
            !discharge.qualifier_set().contains(Qualifier::Exact),
            "the earlier verdict's badge must NOT be upgraded by a later engine"
        );
    }

    /// When fitting engines exist but ALL return non-verdicts, the dispatcher
    /// returns the exhausted-fallthrough discharge: Untrusted + empty +
    /// projecting to Unsupported (never a green), with the honest "exhausted"
    /// reason -- NOT a fabricated verdict and NOT the misleading "no engine
    /// fits".
    #[test]
    fn all_fitting_engines_non_verdict_is_untrusted_unsupported() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::smt_non_verdict(
            "a",
            TierBResult::Unknown,
        )));
        registry.register(Box::new(MockEngine::smt_non_verdict(
            "b",
            TierBResult::Timeout,
        )));

        let discharge = registry.dispatch(&smt_goal(), 1_000);
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
        // The lattice membership is the no-fit floor: it projects to
        // Unsupported and never to a proof.
        let verdict = composite_with_no_fit_dependency(&discharge);
        assert_eq!(verdict, CompositeVerdict::Unsupported);
        assert_ne!(verdict, CompositeVerdict::Proven);
        // Honest reason: exhausted, not "no engine fits". chelis#496: the
        // canonical top-level attribution key is `engine`.
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("exhausted_fallthrough"),
            "the exhausted case is distinguished from no-fit in the evidence"
        );
    }

    /// A goal NO engine fits still takes the canonical no-fit path (distinct
    /// from the exhausted-fallthrough case): the dispatcher never confuses
    /// "shape unfit" with "fit but undecided".
    #[test]
    fn no_fitting_engine_takes_the_no_fit_path_not_exhausted() {
        let mut registry = DischargeRegistry::new();
        // Only an SMT engine; a BoxRange goal has no fit at all.
        registry.register(Box::new(MockEngine::smt_non_verdict(
            "smt_only",
            TierBResult::Unknown,
        )));
        let discharge = registry.dispatch(&box_range_goal(), 1_000);
        assert_no_fit_lattice_membership(&discharge);
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("no_fit"),
            "an unfit shape is no_fit, never exhausted_fallthrough"
        );
    }

    /// A single fitting engine that returns a definite verdict behaves exactly
    /// as before the fall-through change (no regression): the verdict is
    /// returned directly, no fall-through, no exhausted path.
    #[test]
    fn single_fitting_engine_with_a_verdict_is_unchanged() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine::smt_real_arith_verdict(
            "solo",
            TierBResult::Proved,
        )));
        let discharge = registry.dispatch(&smt_goal(), 1_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("solo")
        );
    }

    /// Fall-through respects the BoxRange lane too: a non-verdict BoxRange
    /// engine falls through to a later fitting BoxRange engine that discharges.
    /// (Beacon could register two interval engines; the weaker-first one
    /// declining must not block the stronger one.)
    #[test]
    fn fall_through_works_on_the_box_range_lane() {
        let mut registry = DischargeRegistry::new();
        registry.register(Box::new(MockEngine {
            name: "box_unknown",
            fits_shape: WhichShape::BoxRange,
            soundness: Soundness::Untrusted,
            qualifiers: QualifierSet::new(),
            result: TierBResult::Unknown,
        }));
        registry.register(Box::new(MockEngine::box_range_sound("box_sound")));

        let discharge = registry.dispatch(&box_range_goal(), 1_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert!(
            discharge
                .qualifier_set()
                .contains(Qualifier::SoundOverApproximation)
        );
        assert_eq!(
            discharge.evidence().get("engine").and_then(|v| v.as_str()),
            Some("box_sound")
        );
    }

    // --- WI-15: the Clarabel SoS engine through the DISPATCHED try-until-discharge
    // path (clarabel feature + the targets with the sdp backend) ---

    #[cfg(all(feature = "clarabel", any(target_os = "linux", target_os = "macos")))]
    mod clarabel_dispatch {
        use super::*;
        use crate::clarabel_sos::ClarabelSosEngine;

        fn var(name: &str) -> SmtExpr {
            SmtExpr::Var(name.to_string())
        }

        fn cmp(op: CmpOp, l: SmtExpr, r: SmtExpr) -> SmtExpr {
            SmtExpr::Cmp(op, Box::new(l), Box::new(r))
        }

        fn mul(l: SmtExpr, r: SmtExpr) -> SmtExpr {
            SmtExpr::Arith(crate::solver::ArithOp::Mul, Box::new(l), Box::new(r))
        }

        fn sub(l: SmtExpr, r: SmtExpr) -> SmtExpr {
            SmtExpr::Arith(crate::solver::ArithOp::Sub, Box::new(l), Box::new(r))
        }

        /// `<body> >= 0` for `x in [lo, hi]`.
        fn poly_goal(body: SmtExpr, lo: f64, hi: f64) -> Goal {
            Goal::smt(SmtProperty {
                variables: vec![("x".to_string(), SmtSort::Real)],
                preconditions: vec![
                    cmp(CmpOp::Ge, var("x"), SmtExpr::RealLit(lo)),
                    cmp(CmpOp::Le, var("x"), SmtExpr::RealLit(hi)),
                ],
                postcondition: cmp(CmpOp::Ge, body, SmtExpr::RealLit(0.0)),
            })
        }

        /// A registry with Clarabel ahead of a MOCK SMT engine (the cvc5 stand-in
        /// on the fall-through), so the dispatched try-until-discharge path is
        /// exercised without needing the `smt` feature in this lane. The mock
        /// returns a chosen definite verdict for an SMT goal.
        fn clarabel_then_mock(mock_result: TierBResult) -> DischargeRegistry {
            let mut registry = DischargeRegistry::new();
            registry.register(Box::new(ClarabelSosEngine::clarabel()));
            registry.register(Box::new(MockEngine {
                name: "mock_cvc5",
                fits_shape: WhichShape::Smt,
                soundness: Soundness::SoundApproximate,
                qualifiers: QualifierSet::from_iter_kinds([Qualifier::RealArith]),
                result: mock_result,
            }));
            registry
        }

        #[test]
        fn matched_poly_is_proved_by_clarabel_at_exact_no_fallthrough() {
            // x^2 >= 0 on [-1, 1]: Clarabel certifies it, returns a DEFINITE
            // Proved -> the dispatcher returns it immediately at
            // CertificateBearing@Exact, never reaching the mock fall-through.
            let goal = poly_goal(mul(var("x"), var("x")), -1.0, 1.0);
            // The mock would Disprove if reached; proving Clarabel short-circuits.
            let registry = clarabel_then_mock(TierBResult::Disproved(serde_json::json!({})));
            assert_eq!(registry.selected_engine_name(&goal), Some("clarabel_sos"));
            let discharge = registry.dispatch(&goal, 5_000);
            assert_eq!(*discharge.result(), TierBResult::Proved);
            assert_eq!(discharge.soundness(), Soundness::Exact);
            assert!(
                discharge
                    .qualifier_set()
                    .contains(Qualifier::CertificateBearing)
            );
            // The verdict is Clarabel's, not the mock's (no laundered fall-through).
            assert_eq!(
                discharge.evidence().get("engine").and_then(|v| v.as_str()),
                Some("clarabel_sos")
            );
        }

        #[test]
        fn clarabel_unknown_falls_through_to_the_next_smt_engine() {
            // A goal Clarabel cannot certify but is still in its fitness subset:
            // a cubic x^3 on [0, 1] is nonnegative but ODD-degree handling here
            // may not produce a verifying cert for every such goal -> Clarabel
            // returns Unknown (a non-verdict), so the dispatcher FALLS THROUGH to
            // the mock SMT engine, which here Proves it. The returned verdict is
            // the MOCK's, proving the fall-through actually occurred.
            //
            // To make the fall-through deterministic regardless of whether
            // Clarabel happens to certify x^3, use a goal OUTSIDE Clarabel's
            // certifiable reach but still a recognized poly: the constant FALSE
            // sub-case is covered separately; here we force the Clarabel-Unknown
            // arm with a goal whose proposer returns no cert, then assert the mock
            // verdict is returned. x^3 - x on [0,1] is NEGATIVE in part of (0,1)
            // (e.g. x=1/2 -> -3/8), so Clarabel finds no nonneg cert -> Unknown,
            // and the mock (forced to Prove) supplies the dispatched verdict.
            let goal = poly_goal(
                sub(mul(mul(var("x"), var("x")), var("x")), var("x")),
                0.0,
                1.0,
            );
            let registry = clarabel_then_mock(TierBResult::Proved);
            // Clarabel is tried first (it fits the poly subset)...
            assert_eq!(registry.selected_engine_name(&goal), Some("clarabel_sos"));
            let discharge = registry.dispatch(&goal, 5_000);
            // ...but it returns Unknown (the goal is not nonneg on [0,1]), so the
            // dispatcher falls through to the mock, whose Proved is returned. The
            // verdict carries the MOCK's badge (RealArith), NOT Clarabel's
            // CertificateBearing -- no laundering across the fall-through.
            assert_eq!(*discharge.result(), TierBResult::Proved);
            assert!(discharge.qualifier_set().contains(Qualifier::RealArith));
            assert!(
                !discharge
                    .qualifier_set()
                    .contains(Qualifier::CertificateBearing),
                "the fall-through verdict must carry the mock's badge, not Clarabel's"
            );
            assert_eq!(
                discharge.evidence().get("engine").and_then(|v| v.as_str()),
                Some("mock_cvc5")
            );
        }

        #[test]
        fn false_goal_is_not_proved_through_the_dispatched_path() {
            // x^2 - 1 >= 0 on [-1, 1] is FALSE (-1 at x=0). Clarabel finds no
            // cert -> Unknown -> falls through to the mock. The mock here returns
            // Unknown too (a sound cvc5 would Disprove or Unknown, never Prove a
            // false goal), so the WHOLE dispatch is EXHAUSTED: Untrusted, empty,
            // Error -> Unsupported, NEVER Proved. The soundness-critical end-to-end
            // negative: a false goal is never proved through the dispatched path.
            let goal = poly_goal(
                sub(mul(var("x"), var("x")), SmtExpr::RealLit(1.0)),
                -1.0,
                1.0,
            );
            let registry = clarabel_then_mock(TierBResult::Unknown);
            let discharge = registry.dispatch(&goal, 5_000);
            assert_ne!(
                *discharge.result(),
                TierBResult::Proved,
                "a false goal must never be proved through the dispatched path"
            );
            assert_ne!(discharge.soundness(), Soundness::Exact);
            assert!(
                !discharge
                    .qualifier_set()
                    .contains(Qualifier::CertificateBearing)
            );
            // Exhausted (both engines non-verdict) -> the Unsupported honesty floor.
            assert_eq!(discharge.soundness(), Soundness::Untrusted);
            assert!(discharge.qualifier_set().is_empty());
        }
    }
}
