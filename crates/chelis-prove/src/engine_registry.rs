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
        #[cfg(feature = "smt")]
        registry.register(Box::new(crate::discharge::Cvc5Engine::new()));
        #[cfg(not(feature = "smt"))]
        registry.register(Box::new(SolvePropertyEngine::new()));
        registry
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
    pub fn selected_engine_name(&self, goal: &Goal) -> Option<&'static str> {
        self.select(goal).map(DischargeEngine::name)
    }

    /// Discharge `goal` through the first fitting engine, or take the no-fit
    /// path if none fits.
    ///
    /// The no-fit result is the canonical [`no_fit_discharge`]: an
    /// [`Soundness::Untrusted`] discharge with an empty [`QualifierSet`] and a
    /// [`TierBResult::Error`], which the WI-6 algebra renders
    /// [`crate::composition::CompositeVerdict::Unsupported`]. It is never a
    /// silent pass and never a green.
    pub fn dispatch(&self, goal: &Goal, timeout_ms: u64) -> Discharge {
        match self.select(goal) {
            Some(engine) => engine.discharge(goal, timeout_ms),
            None => no_fit_discharge(goal),
        }
    }
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
    Discharge::new(
        Soundness::Untrusted,
        QualifierSet::new(),
        TierBResult::Error(reason.clone()),
        serde_json::json!({ "dispatch": "no_fit", "reason": reason }),
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

    /// Map a tier-B outcome to its `(soundness, qualifier_set)`. This mirrors
    /// the cvc5 engine's classification: a proved/disproved result is an exact
    /// decision; a timeout/unknown/error is untrusted and carries no qualifier.
    /// In the non-smt build `solve_property` only ever returns `Timeout` (or a
    /// test-forced result), so in practice this yields the untrusted branch.
    fn classify(result: &TierBResult) -> (Soundness, QualifierSet) {
        use crate::discharge::Qualifier;
        match result {
            TierBResult::Proved | TierBResult::Disproved(_) => (
                Soundness::Exact,
                QualifierSet::from_iter_kinds([Qualifier::Exact]),
            ),
            TierBResult::Timeout | TierBResult::Unknown | TierBResult::Error(_) => {
                (Soundness::Untrusted, QualifierSet::new())
            }
        }
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
        let evidence = serde_json::json!({ "engine": "solve_property" });
        Discharge::new(soundness, qualifier_set, result, evidence).unwrap_or_else(|err| {
            Discharge::new(
                Soundness::Untrusted,
                QualifierSet::new(),
                TierBResult::Error(err.to_string()),
                serde_json::json!({ "engine": "solve_property", "internal_error": err.to_string() }),
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
            Discharge::new(
                self.soundness,
                self.qualifiers.clone(),
                self.result.clone(),
                serde_json::json!({ "engine": self.name }),
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

    /// The prove-flow status a discharge maps to, derived from its soundness and
    /// result. An exact, green result is `proved`; everything else (an
    /// untrusted/non-green discharge, i.e. the no-fit case) is `unsupported`.
    fn discharge_status(discharge: &Discharge) -> &'static str {
        match (discharge.soundness(), discharge.result()) {
            (Soundness::Exact, TierBResult::Proved) => "proved",
            (Soundness::Exact, TierBResult::Disproved(_)) => "failed",
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
        // Byte-identical to the pre-WI-9 direct cvc5 call: a trivially-true
        // property proves as exact.
        let discharge = registry.dispatch(&goal, 5_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::Exact);
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
}
