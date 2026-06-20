//! Engine-independent discharge seam (WI-4 / WI-5).
//!
//! This module is the substrate every verification engine plugs into. It sits
//! ABOVE the existing [`crate::solver::Solver`] / [`crate::solver::SmtExpr`]
//! boundary (which already abstracts a single SMT backend). The seam expresses:
//!
//! - [`Goal`]: a canonical, engine-independent statement to discharge. It
//!   carries an OPTIONAL in-memory IR back-reference ([`Goal::ir`]) which is a
//!   handle to the internal compiler IR. In Phase 1 the cvc5 path leaves this
//!   `None`; the graph-extraction seam (WI-3, deferred) populates it later. It
//!   is deliberately NOT the serialized `WireDag`.
//! - [`Soundness`] and [`Qualifier`] / [`QualifierSet`]: the guarantee an
//!   engine attaches to a discharge. The seven qualifier kinds are incomparable
//!   on one axis, so they form a set, not a chain (the WI-6 verdict algebra
//!   rolls them up). Soundness is the orthogonal "is the result trustworthy as
//!   a proof" axis.
//! - [`Discharge`]: what an engine returns: the outcome plus its
//!   `(soundness, qualifier_set)` and the underlying evidence.
//! - [`DischargeEngine`]: the trait each engine implements. cvc5 is one engine
//!   ([`Cvc5Engine`]) that internally keeps the
//!   `SmtProperty -> SmtExpr -> Solver` pipeline, so the SmtExpr lowering sites
//!   stay cvc5-internal.
//!
//! WI-5 note: [`Goal`] is designed once with the interval-box-input plus
//! output-range-assertion variant ([`GoalShape::BoxRange`]). Phase 1 lands the
//! goal TYPE only. There is deliberately no property-surface SYNTAX for
//! authoring such goals yet (no producer exists; none is promised this phase).

use crate::tier_b::{SmtProperty, TierBResult};

/// Opaque, type-erased in-memory back-reference to the internal compiler IR.
///
/// In Phase 1 this is always unpopulated for the cvc5 path. The
/// graph-extraction seam (WI-3, deferred) will give this a concrete payload
/// (a handle to / index into the internal `chelis_ir::Dag`). It is modelled as
/// an opaque handle now so adding the payload later does not change the
/// [`Goal`] shape or pull a `chelis-ir` dependency into this crate before a
/// consumer needs it.
///
/// It is NOT the serialized `WireDag`: this seam never reaches for the wire
/// schema or its `schema_version`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IrHandle {
    /// Stable identifier of the IR graph this goal was extracted from, once a
    /// producer populates it. `None` on the Phase 1 cvc5 path.
    node: Option<u64>,
}

impl IrHandle {
    /// The unpopulated handle used on the Phase 1 cvc5 path.
    pub const fn unpopulated() -> Self {
        Self { node: None }
    }

    /// Construct a handle referencing a specific IR node/graph index. Reserved
    /// for the WI-3 graph-extraction producer; unused on the cvc5 path.
    pub const fn from_node(node: u64) -> Self {
        Self { node: Some(node) }
    }

    /// Whether this handle has been populated by a producer.
    pub const fn is_populated(&self) -> bool {
        self.node.is_some()
    }

    /// The referenced node index, if populated.
    pub const fn node(&self) -> Option<u64> {
        self.node
    }
}

/// An interval box over named inputs: each name is bounded by `[lo, hi]`.
///
/// WI-5 type surface only. The lower bound must not exceed the upper bound;
/// [`Goal::box_range`] rejects an inverted interval as an ill-formed goal.
#[derive(Debug, Clone, PartialEq)]
pub struct IntervalBox {
    /// One `(name, lo, hi)` constraint per input dimension.
    pub dims: Vec<(String, f64, f64)>,
}

/// An assertion that a named output lies within `[lo, hi]`.
///
/// WI-5 type surface only.
#[derive(Debug, Clone, PartialEq)]
pub struct OutputRange {
    pub output: String,
    pub lo: f64,
    pub hi: f64,
}

/// The canonical shape of a [`Goal`].
#[derive(Debug, Clone, PartialEq)]
pub enum GoalShape {
    /// A structured SMT property: preconditions plus a postcondition to prove.
    /// This is the shape the cvc5 engine discharges in Phase 1.
    Smt(SmtProperty),
    /// WI-5: interval-box inputs with an output-range assertion. Beacon's
    /// native verdict form. TYPE ONLY in Phase 1 (no authoring syntax, no
    /// engine discharges it yet).
    BoxRange {
        inputs: IntervalBox,
        output: OutputRange,
    },
}

/// A canonical, engine-independent goal to discharge.
#[derive(Debug, Clone, PartialEq)]
pub struct Goal {
    /// What is being asserted.
    pub shape: GoalShape,
    /// Optional in-memory IR back-reference. Unpopulated on the Phase 1 cvc5
    /// path; populated later by the WI-3 graph-extraction seam.
    pub ir: IrHandle,
}

/// Error constructing or routing a [`Goal`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum GoalError {
    /// An interval-box dimension has `lo > hi`, or an output range has
    /// `lo > hi`: the goal asserts an empty region and is ill-formed.
    #[error("ill-formed goal: {0}")]
    IllFormed(String),
}

/// Whether `[lo, hi]` is a non-empty interval. A NaN bound is incomparable, so
/// `partial_cmp` returns `None` and the interval is treated as ill-formed;
/// `lo > hi` (an inverted interval) is likewise rejected.
fn interval_is_non_empty(lo: f64, hi: f64) -> bool {
    matches!(
        lo.partial_cmp(&hi),
        Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
    )
}

impl Goal {
    /// Build a structured-SMT goal from an existing [`SmtProperty`]. The IR
    /// handle is left unpopulated (the Phase 1 cvc5 path).
    pub fn smt(property: SmtProperty) -> Self {
        Self {
            shape: GoalShape::Smt(property),
            ir: IrHandle::unpopulated(),
        }
    }

    /// Build a WI-5 box / output-range goal. Validates that every interval is
    /// non-empty (`lo <= hi`); an inverted interval is rejected as an
    /// ill-formed goal rather than silently accepted. TYPE-LEVEL surface only:
    /// no engine discharges this shape in Phase 1.
    pub fn box_range(inputs: IntervalBox, output: OutputRange) -> Result<Self, GoalError> {
        for (name, lo, hi) in &inputs.dims {
            if !interval_is_non_empty(*lo, *hi) {
                return Err(GoalError::IllFormed(format!(
                    "input box dimension `{name}` has lo > hi (or a NaN bound)"
                )));
            }
        }
        if !interval_is_non_empty(output.lo, output.hi) {
            return Err(GoalError::IllFormed(format!(
                "output range `{}` has lo > hi (or a NaN bound)",
                output.output
            )));
        }
        Ok(Self {
            shape: GoalShape::BoxRange { inputs, output },
            ir: IrHandle::unpopulated(),
        })
    }

    /// Attach an IR back-reference (WI-3 producer surface; unused on the cvc5
    /// path in Phase 1).
    pub fn with_ir(mut self, ir: IrHandle) -> Self {
        self.ir = ir;
        self
    }

    /// Borrow the underlying [`SmtProperty`] when this is a structured-SMT
    /// goal, else `None`.
    pub fn as_smt(&self) -> Option<&SmtProperty> {
        match &self.shape {
            GoalShape::Smt(p) => Some(p),
            GoalShape::BoxRange { .. } => None,
        }
    }
}

/// The soundness of a discharge: how trustworthy the result is as a proof.
///
/// This is the orthogonal axis to [`QualifierSet`]; the WI-6 verdict algebra
/// rolls a dependency set up to the MINIMUM soundness across its discharges.
/// `Ord` ranks weaker before stronger, so `min` is the weakest-link fold.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Soundness {
    /// No trust: the goal could not be discharged (timeout, unknown, lowering
    /// error). Never a green result.
    Untrusted,
    /// Empirical evidence only (e.g. fuzz sampling): supports the goal but is
    /// not a proof.
    Empirical,
    /// A sound over-approximation: the result holds but may be conservative.
    SoundApproximate,
    /// An exact proof: the result holds with no approximation.
    Exact,
}

/// One guarantee kind an engine can attach to a discharge. The seven kinds are
/// incomparable on a single axis, so a discharge carries a [`QualifierSet`] and
/// the WI-6 algebra unions them across a dependency set.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum Qualifier {
    /// Exact decision procedure (e.g. cvc5 UNSAT on the negation).
    Exact,
    /// Delta-complete decision (e.g. a dReal-style delta-SAT result).
    DeltaComplete,
    /// Backed by a certified special-function envelope.
    SpecialFunctionCertified,
    /// Sound over-approximation (e.g. interval/abstract-interpretation bound).
    SoundOverApproximation,
    /// Carries an independently checkable certificate (e.g. an SoS witness).
    CertificateBearing,
    /// Established by randomized fuzz sampling only.
    Fuzz,
    /// Asserted as an axiom (trusted, not derived).
    Axiom,
}

impl Qualifier {
    pub fn as_str(self) -> &'static str {
        match self {
            Qualifier::Exact => "exact",
            Qualifier::DeltaComplete => "delta_complete",
            Qualifier::SpecialFunctionCertified => "special_function_certified",
            Qualifier::SoundOverApproximation => "sound_over_approximation",
            Qualifier::CertificateBearing => "certificate_bearing",
            Qualifier::Fuzz => "fuzz",
            Qualifier::Axiom => "axiom",
        }
    }

    /// Whether this qualifier may only ride a [`Soundness::Exact`] discharge.
    /// `exact` is the one kind that claims a no-approximation proof, so it is
    /// rejected on any weaker soundness (the integrity invariant tested below).
    fn requires_exact_soundness(self) -> bool {
        matches!(self, Qualifier::Exact)
    }
}

/// A set of [`Qualifier`] kinds, kept sorted and deduplicated so equal sets
/// compare equal regardless of insertion order. The WI-6 rollup unions these.
#[derive(Debug, Clone, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct QualifierSet {
    qualifiers: Vec<Qualifier>,
}

impl QualifierSet {
    pub fn new() -> Self {
        Self::default()
    }

    /// Build a set from an iterator of qualifiers (sorted + deduplicated).
    pub fn from_iter_kinds<I: IntoIterator<Item = Qualifier>>(iter: I) -> Self {
        let mut set = Self::new();
        for q in iter {
            set.insert(q);
        }
        set
    }

    /// Insert a qualifier, keeping the set sorted and deduplicated.
    pub fn insert(&mut self, q: Qualifier) {
        if let Err(pos) = self.qualifiers.binary_search(&q) {
            self.qualifiers.insert(pos, q);
        }
    }

    pub fn contains(&self, q: Qualifier) -> bool {
        self.qualifiers.binary_search(&q).is_ok()
    }

    pub fn is_empty(&self) -> bool {
        self.qualifiers.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = Qualifier> + '_ {
        self.qualifiers.iter().copied()
    }

    /// Union with another set (WI-6 rollup primitive).
    pub fn union(&self, other: &Self) -> Self {
        let mut out = self.clone();
        for q in other.iter() {
            out.insert(q);
        }
        out
    }
}

/// Error building a [`Discharge`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum DischargeError {
    /// A qualifier was attached that the discharge's soundness cannot support.
    /// Specifically: a non-exact-soundness discharge (e.g. a fuzz result)
    /// cannot carry the `exact` qualifier; that would launder a weaker
    /// guarantee into a stronger badge.
    #[error("qualifier `{qualifier}` requires exact soundness but discharge is `{soundness:?}`")]
    QualifierExceedsSoundness {
        qualifier: &'static str,
        soundness: Soundness,
    },
}

/// What an engine returns for a [`Goal`]: the outcome, the guarantee it carries,
/// and the underlying evidence.
///
/// Constructed only through [`Discharge::new`], which enforces the integrity
/// invariant that a discharge cannot carry a qualifier its soundness does not
/// support.
///
/// `Discharge` deliberately does not derive `Serialize`/`Deserialize`, while
/// its parts (`Soundness`, `QualifierSet`) do. Because the integrity invariant
/// lives in [`Discharge::new`], a `Discharge` must never be reconstructed
/// field-by-field from separately (de)serialized parts: that bypasses the
/// constructor and could rejoin a weak soundness with a strong qualifier
/// outside the seam. Any future wire form of a discharge (WI-6 and later) must
/// round-trip through [`Discharge::new`] (or a `Deserialize` impl that
/// re-validates through it), never by independent field assembly.
#[derive(Debug, Clone, PartialEq)]
pub struct Discharge {
    soundness: Soundness,
    qualifier_set: QualifierSet,
    /// The underlying tier-B outcome, so existing callers can pattern-match the
    /// proved/disproved/timeout/unknown/error result exactly as before.
    result: TierBResult,
    /// Engine-specific evidence (model, solver name, etc.).
    evidence: serde_json::Value,
}

impl Discharge {
    /// Build a discharge, enforcing the integrity invariant: every qualifier in
    /// the set must be supportable by `soundness`. A `fuzz` (or any non-exact)
    /// discharge that tries to carry `exact` is rejected here, in the
    /// constructor, rather than silently producing a laundered badge.
    pub fn new(
        soundness: Soundness,
        qualifier_set: QualifierSet,
        result: TierBResult,
        evidence: serde_json::Value,
    ) -> Result<Self, DischargeError> {
        if soundness != Soundness::Exact {
            for q in qualifier_set.iter() {
                if q.requires_exact_soundness() {
                    return Err(DischargeError::QualifierExceedsSoundness {
                        qualifier: q.as_str(),
                        soundness,
                    });
                }
            }
        }
        Ok(Self {
            soundness,
            qualifier_set,
            result,
            evidence,
        })
    }

    pub fn soundness(&self) -> Soundness {
        self.soundness
    }

    pub fn qualifier_set(&self) -> &QualifierSet {
        &self.qualifier_set
    }

    pub fn result(&self) -> &TierBResult {
        &self.result
    }

    /// Consume the discharge, returning the underlying tier-B result. Lets the
    /// migrated callers keep their exact `match` on `TierBResult` unchanged.
    pub fn into_result(self) -> TierBResult {
        self.result
    }

    pub fn evidence(&self) -> &serde_json::Value {
        &self.evidence
    }
}

/// An engine that can discharge [`Goal`]s. cvc5 is one impl; later engines
/// (Z3, Clarabel, Beacon, ...) plug in here without touching callers.
pub trait DischargeEngine {
    /// Stable engine identifier for evidence and diagnostics.
    fn name(&self) -> &'static str;

    /// Whether this engine is a fit for the goal's shape. The dispatcher
    /// (WI-9, later) uses this to route; callers may use it to skip an engine.
    fn fitness(&self, goal: &Goal) -> bool;

    /// Discharge the goal, returning its outcome and guarantee.
    fn discharge(&self, goal: &Goal, timeout_ms: u64) -> Discharge;
}

/// The cvc5 discharge engine. Internally keeps the existing
/// `SmtProperty -> SmtExpr -> Solver` pipeline ([`crate::tier_b::solve_property`]),
/// so the SmtExpr lowering sites stay cvc5-internal and behavior is unchanged.
#[derive(Debug, Clone, Copy, Default)]
pub struct Cvc5Engine;

impl Cvc5Engine {
    pub const fn new() -> Self {
        Self
    }

    /// Map a tier-B outcome to its `(soundness, qualifier_set)`. A proved or
    /// disproved cvc5 result is an EXACT decision over the chosen logic; a
    /// timeout, unknown, or lowering error is untrusted (never a proof) and
    /// carries no qualifier.
    fn classify(result: &TierBResult) -> (Soundness, QualifierSet) {
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

impl DischargeEngine for Cvc5Engine {
    fn name(&self) -> &'static str {
        "cvc5"
    }

    fn fitness(&self, goal: &Goal) -> bool {
        // cvc5 discharges structured-SMT goals. The WI-5 box/output-range shape
        // is Beacon's native form; cvc5 does not own it in Phase 1.
        matches!(goal.shape, GoalShape::Smt(_))
    }

    fn discharge(&self, goal: &Goal, timeout_ms: u64) -> Discharge {
        let result = match goal.as_smt() {
            Some(property) => crate::tier_b::solve_property(property, timeout_ms),
            None => {
                TierBResult::Error("cvc5 engine cannot discharge a non-SMT goal shape".to_string())
            }
        };
        let (soundness, qualifier_set) = Self::classify(&result);
        let evidence = serde_json::json!({ "solver": "cvc5" });
        // The classification only ever pairs the `exact` qualifier with
        // `Soundness::Exact`, so this constructor cannot fail here; surfacing
        // the error as a non-proof discharge keeps the seam total.
        Discharge::new(soundness, qualifier_set, result, evidence).unwrap_or_else(|err| {
            Discharge::new(
                Soundness::Untrusted,
                QualifierSet::new(),
                TierBResult::Error(err.to_string()),
                serde_json::json!({ "solver": "cvc5", "internal_error": err.to_string() }),
            )
            .expect("untrusted discharge with empty qualifier set is always valid")
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::solver::{CmpOp, SmtExpr, SmtSort};

    fn trivially_true_property() -> SmtProperty {
        // forall x: Real . x == x  -- proved by asserting the negation UNSAT.
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

    fn false_property() -> SmtProperty {
        // postcondition 1.0 < 0.0 is false: cvc5 disproves it.
        SmtProperty {
            variables: vec![],
            preconditions: vec![],
            postcondition: SmtExpr::Cmp(
                CmpOp::Lt,
                Box::new(SmtExpr::RealLit(1.0)),
                Box::new(SmtExpr::RealLit(0.0)),
            ),
        }
    }

    #[test]
    fn goal_smt_leaves_ir_handle_unpopulated_in_phase_1() {
        let goal = Goal::smt(trivially_true_property());
        assert!(!goal.ir.is_populated());
        assert!(goal.as_smt().is_some());
    }

    #[test]
    fn cvc5_engine_fitness_accepts_smt_goal_rejects_box_range() {
        let engine = Cvc5Engine::new();
        let smt_goal = Goal::smt(trivially_true_property());
        assert!(engine.fitness(&smt_goal));

        let box_goal = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 0.0, 100.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 50.0,
            },
        )
        .expect("well-formed box goal");
        assert!(!engine.fitness(&box_goal));
    }

    // --- WI-5 box/output-range goal TYPE (positive + negative twin) ---

    #[test]
    fn box_range_goal_accepts_non_empty_intervals() {
        let goal = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 50.0, 150.0), ("t".to_string(), 0.0, 1.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 1000.0,
            },
        )
        .expect("non-empty intervals are well-formed");
        assert!(matches!(goal.shape, GoalShape::BoxRange { .. }));
        assert!(!goal.ir.is_populated());
    }

    #[test]
    fn box_range_goal_rejects_inverted_input_interval() {
        let err = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 150.0, 50.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 1.0,
            },
        )
        .expect_err("lo > hi must be rejected");
        assert!(matches!(err, GoalError::IllFormed(_)));
    }

    #[test]
    fn box_range_goal_rejects_nan_input_bound() {
        let err = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), f64::NAN, 50.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 1.0,
            },
        )
        .expect_err("a NaN bound is incomparable and must be rejected");
        assert!(matches!(err, GoalError::IllFormed(_)));
    }

    #[test]
    fn box_range_goal_rejects_inverted_output_range() {
        let err = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 0.0, 1.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 10.0,
                hi: 1.0,
            },
        )
        .expect_err("output lo > hi must be rejected");
        assert!(matches!(err, GoalError::IllFormed(_)));
    }

    // --- Discharge integrity invariant (positive + negative twin) ---

    #[test]
    fn exact_qualifier_is_allowed_on_exact_soundness() {
        let discharge = Discharge::new(
            Soundness::Exact,
            QualifierSet::from_iter_kinds([Qualifier::Exact]),
            TierBResult::Proved,
            serde_json::json!({}),
        )
        .expect("exact qualifier on exact soundness is valid");
        assert_eq!(discharge.soundness(), Soundness::Exact);
        assert!(discharge.qualifier_set().contains(Qualifier::Exact));
    }

    #[test]
    fn fuzz_discharge_cannot_carry_exact_qualifier() {
        let err = Discharge::new(
            Soundness::Empirical,
            QualifierSet::from_iter_kinds([Qualifier::Fuzz, Qualifier::Exact]),
            TierBResult::Proved,
            serde_json::json!({}),
        )
        .expect_err("empirical/fuzz soundness cannot carry the exact qualifier");
        assert!(matches!(
            err,
            DischargeError::QualifierExceedsSoundness {
                qualifier: "exact",
                soundness: Soundness::Empirical,
            }
        ));
    }

    #[test]
    fn sound_approximate_discharge_cannot_carry_exact_qualifier() {
        let err = Discharge::new(
            Soundness::SoundApproximate,
            QualifierSet::from_iter_kinds([Qualifier::Exact]),
            TierBResult::Proved,
            serde_json::json!({}),
        )
        .expect_err("a sound over-approximation is not an exact proof");
        assert!(matches!(
            err,
            DischargeError::QualifierExceedsSoundness { .. }
        ));
    }

    #[test]
    fn fuzz_discharge_with_fuzz_qualifier_is_valid() {
        let discharge = Discharge::new(
            Soundness::Empirical,
            QualifierSet::from_iter_kinds([Qualifier::Fuzz]),
            TierBResult::Timeout,
            serde_json::json!({}),
        )
        .expect("fuzz qualifier on empirical soundness is valid");
        assert!(discharge.qualifier_set().contains(Qualifier::Fuzz));
        assert!(!discharge.qualifier_set().contains(Qualifier::Exact));
    }

    #[test]
    fn qualifier_set_dedups_and_orders() {
        let a = QualifierSet::from_iter_kinds([Qualifier::Fuzz, Qualifier::Axiom, Qualifier::Fuzz]);
        let b = QualifierSet::from_iter_kinds([Qualifier::Axiom, Qualifier::Fuzz]);
        assert_eq!(a, b);
        assert_eq!(a.iter().count(), 2);
    }

    #[test]
    fn soundness_min_is_weakest_link() {
        // The WI-6 rollup folds with `min`; weaker sorts before stronger.
        assert!(Soundness::Untrusted < Soundness::Empirical);
        assert!(Soundness::Empirical < Soundness::SoundApproximate);
        assert!(Soundness::SoundApproximate < Soundness::Exact);
        assert_eq!(
            Soundness::Exact.min(Soundness::Empirical),
            Soundness::Empirical
        );
    }

    // --- cvc5 engine happy path (smt feature only: needs a live solver) ---

    #[cfg(feature = "smt")]
    #[test]
    fn cvc5_engine_proves_a_trivial_goal_as_exact() {
        let engine = Cvc5Engine::new();
        let goal = Goal::smt(trivially_true_property());
        let discharge = engine.discharge(&goal, 5_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::Exact);
        assert!(discharge.qualifier_set().contains(Qualifier::Exact));
    }

    #[cfg(feature = "smt")]
    #[test]
    fn cvc5_engine_disproves_a_false_goal_as_exact() {
        let engine = Cvc5Engine::new();
        let goal = Goal::smt(false_property());
        let discharge = engine.discharge(&goal, 5_000);
        assert!(matches!(discharge.result(), TierBResult::Disproved(_)));
        assert_eq!(discharge.soundness(), Soundness::Exact);
    }

    #[cfg(feature = "smt")]
    #[test]
    fn cvc5_engine_refuses_box_range_goal_as_untrusted() {
        let engine = Cvc5Engine::new();
        let goal = Goal::box_range(
            IntervalBox {
                dims: vec![("s".to_string(), 0.0, 1.0)],
            },
            OutputRange {
                output: "price".to_string(),
                lo: 0.0,
                hi: 1.0,
            },
        )
        .expect("well-formed box goal");
        // cvc5 is not a fit; discharging anyway must not fabricate a proof.
        let discharge = engine.discharge(&goal, 5_000);
        assert!(matches!(discharge.result(), TierBResult::Error(_)));
        assert_eq!(discharge.soundness(), Soundness::Untrusted);
        assert!(discharge.qualifier_set().is_empty());
    }
}
