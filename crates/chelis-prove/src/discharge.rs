//! Engine-independent discharge seam (WI-4 / WI-5).
//!
//! This module is the substrate every verification engine plugs into. It sits
//! ABOVE the existing [`crate::solver::Solver`] / [`crate::solver::SmtExpr`]
//! boundary (which already abstracts a single SMT backend). The seam expresses:
//!
//! - [`Goal`]: a canonical, engine-independent statement to discharge. It
//!   carries an OPTIONAL IR back-reference ([`Goal::ir`]) which addresses the
//!   serialized `WireDag` v1 artifact a consumer (Beacon) deserializes. In
//!   Phase 1 the cvc5 path leaves this unpopulated; the graph-extraction seam
//!   (WI-3) populates it with the artifact's content hash + a root index.
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

/// A content-addressed back-reference to a serialized `WireDag` v1 artifact.
///
/// This is exactly what an out-of-tree consumer (Beacon) resolves: it parses
/// the serialized `WireDag` JSON bytes, validates `schema_version <=
/// WIRE_DAG_SCHEMA_VERSION` (currently `2` since chelis#178; a lower version
/// is forward-compatible via additive defaults, a higher one fails closed),
/// computes a sha256 over those bytes, and selects the output of interest by
/// `root_index`. So the handle addresses that artifact by:
///
/// - [`dag_hash`](Self::dag_hash): the lowercase-hex sha256 of the serialized
///   `WireDag` v1 bytes. A consumer recomputes the same digest over the bytes
///   it received and compares for byte-identity; lowercase hex round-trips
///   cleanly through JSON and is cheap to compare.
/// - [`root_index`](Self::root_index): which `WireDag.roots` entry this goal's
///   single scalar output corresponds to.
///
/// In Phase 1 this is always unpopulated for the cvc5 path. The
/// graph-extraction seam (WI-3) populates it via [`from_wire_dag`](Self::from_wire_dag)
/// once it has serialized and hashed the artifact. The handle stays a plain
/// hash + index: it does NOT hold a `chelis_ir::Dag` or a `WireDag` value, so
/// populating it pulls no live IR dependency into this crate; the producer
/// computes the hash and hands it here.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct IrHandle {
    /// Lowercase-hex sha256 of the serialized `WireDag` v1 artifact this goal
    /// was extracted from, once a producer populates it. `None` on the Phase 1
    /// cvc5 path.
    dag_hash: Option<String>,
    /// Which `WireDag.roots` entry this goal's output corresponds to. `None`
    /// on the Phase 1 cvc5 path.
    root_index: Option<u64>,
}

impl IrHandle {
    /// The unpopulated handle used on the Phase 1 cvc5 path.
    pub fn unpopulated() -> Self {
        Self {
            dag_hash: None,
            root_index: None,
        }
    }

    /// Construct a handle addressing a serialized `WireDag` v1 artifact by its
    /// content hash and the root index this goal's output selects. Reserved for
    /// the WI-3 graph-extraction producer; unused on the cvc5 path. `dag_hash`
    /// is the lowercase-hex sha256 of the serialized artifact bytes.
    pub fn from_wire_dag(dag_hash: String, root_index: u64) -> Self {
        Self {
            dag_hash: Some(dag_hash),
            root_index: Some(root_index),
        }
    }

    /// Whether this handle has been populated by a producer.
    pub const fn is_populated(&self) -> bool {
        self.dag_hash.is_some()
    }

    /// The content hash of the addressed `WireDag` v1 artifact, if populated.
    pub fn dag_hash(&self) -> Option<&str> {
        self.dag_hash.as_deref()
    }

    /// The root index this goal's output selects, if populated.
    pub const fn root_index(&self) -> Option<u64> {
        self.root_index
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
    /// Optional back-reference to the serialized `WireDag` v1 artifact this
    /// goal was extracted from (content hash + root index). Unpopulated on the
    /// Phase 1 cvc5 path; populated by the WI-3 graph-extraction seam.
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
    /// The SMT base proof is over the REALS, while runtime arithmetic is
    /// machine arithmetic (int widths lower to an unbounded integer sort,
    /// `f32`/`f64` to `Real`; no overflow, NaN, or IEEE rounding). The proof
    /// holds over the reals but is a sound over-approximation of the machine
    /// claim, so it discloses the gap rather than claiming exact machine
    /// soundness (chelis#422). The legacy `arith_model:"real"` side field
    /// mirrors this qualifier.
    RealArith,
    /// Carries an independently checkable certificate (e.g. an SoS witness).
    CertificateBearing,
    /// Established by randomized fuzz sampling only.
    Fuzz,
    /// The BASE proof obligation itself was established by randomized fuzz
    /// sampling only -- no SMT (or other deductive) proof underlies it
    /// (chelis#422). Distinct from [`Qualifier::Fuzz`], which marks a
    /// fuzz-validated CONTRACT assumption discharged UNDER an otherwise-exact
    /// base. A `FuzzBase` rollup can never render a `proven_*` badge: a
    /// fuzz-only base is not proven, so it must not read as
    /// proven-modulo-anything.
    FuzzBase,
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
            Qualifier::RealArith => "real_arithmetic",
            Qualifier::CertificateBearing => "certificate_bearing",
            Qualifier::Fuzz => "fuzz",
            Qualifier::FuzzBase => "fuzz_base",
            Qualifier::Axiom => "axiom",
        }
    }

    /// The MINIMUM [`Soundness`] a discharge must carry to be allowed to wear
    /// this qualifier. A discharge whose soundness is strictly below this floor
    /// is rejected by [`Discharge::new`]: attaching a guarantee a weaker result
    /// cannot back would launder the weaker result into a stronger badge.
    ///
    /// The floors follow each kind's establishment demand, weakest-link first:
    ///
    /// - [`Qualifier::Exact`] and [`Qualifier::CertificateBearing`] claim a
    ///   no-approximation result (an exact decision, or an independently checked
    ///   exact witness), so they demand [`Soundness::Exact`].
    /// - [`Qualifier::DeltaComplete`], [`Qualifier::SpecialFunctionCertified`],
    ///   [`Qualifier::SoundOverApproximation`], and [`Qualifier::RealArith`] are
    ///   sound but conservative (a delta-relaxed decision, a certified envelope,
    ///   an over-approximating bound, or a proof over the reals standing in for
    ///   machine arithmetic), so they demand [`Soundness::SoundApproximate`].
    /// - [`Qualifier::Fuzz`] and [`Qualifier::FuzzBase`] are empirical sampling,
    ///   so they demand [`Soundness::Empirical`].
    /// - [`Qualifier::Axiom`] is asserted, not established, so it places no
    ///   establishment demand: its floor is [`Soundness::Untrusted`], the bottom
    ///   of the lattice, and it rides any soundness.
    fn min_soundness(self) -> Soundness {
        match self {
            Qualifier::Exact | Qualifier::CertificateBearing => Soundness::Exact,
            Qualifier::DeltaComplete
            | Qualifier::SpecialFunctionCertified
            | Qualifier::SoundOverApproximation
            | Qualifier::RealArith => Soundness::SoundApproximate,
            Qualifier::Fuzz | Qualifier::FuzzBase => Soundness::Empirical,
            Qualifier::Axiom => Soundness::Untrusted,
        }
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
    /// Each qualifier has a minimum soundness floor (see
    /// [`Qualifier::min_soundness`]); a discharge whose soundness falls strictly
    /// below that floor cannot carry the qualifier, because that would launder a
    /// weaker result into a stronger badge (e.g. a fuzz result wearing the
    /// `exact` badge, or an empirical result wearing `delta_complete`).
    #[error(
        "qualifier `{qualifier}` requires at least `{required:?}` soundness but discharge is `{soundness:?}`"
    )]
    QualifierExceedsSoundness {
        qualifier: &'static str,
        required: Soundness,
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
    /// the set must be supportable by `soundness`. Each qualifier has a minimum
    /// soundness floor ([`Qualifier::min_soundness`]); any qualifier whose floor
    /// is strictly above `soundness` is rejected here, in the constructor,
    /// rather than silently producing a laundered badge. For example a `fuzz`
    /// (empirical) discharge cannot carry `exact`, and an `empirical` discharge
    /// cannot carry `delta_complete`.
    pub fn new(
        soundness: Soundness,
        qualifier_set: QualifierSet,
        result: TierBResult,
        evidence: serde_json::Value,
    ) -> Result<Self, DischargeError> {
        for q in qualifier_set.iter() {
            let required = q.min_soundness();
            if soundness < required {
                return Err(DischargeError::QualifierExceedsSoundness {
                    qualifier: q.as_str(),
                    required,
                    soundness,
                });
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

/// The ENGINE-AGNOSTIC classification of an SMT [`TierBResult`] into its
/// `(soundness, qualifier_set)` (chelis#422, WS-5).
///
/// Every SMT engine that lowers a goal to an over-the-reals decision procedure
/// shares this map, because the soundness story is the SAME regardless of which
/// solver produced the result: ints lower to an unbounded integer sort and
/// `f32`/`f64` to `Real`, so a `Proved`/`Disproved` holds over the REALS but is
/// a sound over-approximation of the machine claim. It is therefore
/// [`Soundness::SoundApproximate`] carrying [`Qualifier::RealArith`] (the
/// disclosure that the proof is over reals), and a timeout/unknown/lowering
/// error is [`Soundness::Untrusted`] with no qualifier (never a proof).
///
/// cvc5 ([`Cvc5Engine`]) and Z3 (the `z3`-feature engine) both route through
/// this function, so a Z3 `Proved` and a cvc5 `Proved` carry byte-identical
/// `(soundness, qualifier_set)` and project through the verdict algebra to the
/// same `proven_modulo_real_arithmetic` / `disproved_modulo_real_arithmetic`.
/// This is what lets the dispatcher fall through cvc5 -> Z3 without one engine
/// laundering the other's guarantee: neither can mint a stronger badge than
/// this shared classification allows.
pub fn classify_smt_outcome(result: &TierBResult) -> (Soundness, QualifierSet) {
    match result {
        TierBResult::Proved | TierBResult::Disproved(_) => (
            Soundness::SoundApproximate,
            QualifierSet::from_iter_kinds([Qualifier::RealArith]),
        ),
        TierBResult::Timeout | TierBResult::Unknown | TierBResult::Error(_) => {
            (Soundness::Untrusted, QualifierSet::new())
        }
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
#[cfg(feature = "smt")]
pub struct Cvc5Engine;

#[cfg(feature = "smt")]
impl Cvc5Engine {
    pub const fn new() -> Self {
        Self
    }

    /// Map a tier-B outcome to its `(soundness, qualifier_set)` through the
    /// shared, engine-agnostic [`classify_smt_outcome`]. The cvc5 decision is
    /// over the REALS, not machine arithmetic (chelis#422): int widths lower to
    /// an unbounded integer sort and `f32`/`f64` to `Real`, so a proved result
    /// holds over the reals but is a sound over-approximation of the machine
    /// claim -- `SoundApproximate` carrying [`Qualifier::RealArith`]. A
    /// timeout/unknown/lowering error is untrusted with no qualifier. A
    /// `Disproved` is classified the same way (a real-arithmetic decision is
    /// not exact); its `RealArith` qualifier drives the verdict algebra to the
    /// hedged `disproved_modulo_real_arithmetic`, symmetric to
    /// `proven_modulo_real_arithmetic`, never a flat definite `failed`. Z3
    /// shares this exact map, so the two engines' discharges are
    /// interchangeable under the verdict algebra.
    fn classify(result: &TierBResult) -> (Soundness, QualifierSet) {
        classify_smt_outcome(result)
    }
}

#[cfg(feature = "smt")]
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
        #[cfg_attr(not(feature = "carcara"), allow(unused_mut))]
        let mut evidence = serde_json::json!({ "solver": "cvc5" });

        // WI-16 Carcara audit dimension: when the `carcara` feature is on and
        // cvc5 returned a proof (Proved == UNSAT), independently re-check that
        // proof with Carcara and FOLD the outcome into the evidence. This does
        // NOT change `soundness`/`qualifier_set`: a cvc5 `Proved` keeps the
        // classification's `SoundApproximate` / `RealArith` badge (the auditor
        // adds an independent-audit dimension, it does not manufacture a
        // stronger badge). A re-check FAILURE is recorded in the evidence as an
        // auditor disagreement (write-only today; see `carcara_audit`), never
        // folded into a confirmed state.
        #[cfg(feature = "carcara")]
        if matches!(result, TierBResult::Proved)
            && let Some(property) = goal.as_smt()
        {
            let audit = crate::carcara_audit::audit_cvc5_proof(property, timeout_ms);
            evidence["carcara_audit"] = serde_json::to_value(&audit)
                .unwrap_or(serde_json::Value::String("serialization_error".to_string()));
        }

        // The classification only ever pairs the `real_arithmetic` qualifier
        // with `Soundness::SoundApproximate` (its floor) or returns an empty
        // set at `Untrusted`, so this constructor cannot fail here; surfacing
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

    #[cfg(feature = "smt")]
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
        assert!(goal.ir.dag_hash().is_none());
        assert!(goal.ir.root_index().is_none());
        assert!(goal.as_smt().is_some());
    }

    #[test]
    fn ir_handle_unpopulated_addresses_nothing() {
        let handle = IrHandle::unpopulated();
        assert!(!handle.is_populated());
        assert!(handle.dag_hash().is_none());
        assert!(handle.root_index().is_none());
        assert_eq!(handle, IrHandle::default());
    }

    #[test]
    fn ir_handle_from_wire_dag_carries_hash_and_root_index() {
        // The WI-3 producer surface: a content hash (lowercase-hex sha256 of
        // the serialized WireDag v1 bytes) plus the root index the goal's
        // output selects. The handle holds only the hash + index -- never a
        // Dag or WireDag value.
        let hash = "a".repeat(64);
        let handle = IrHandle::from_wire_dag(hash.clone(), 2);
        assert!(handle.is_populated());
        assert_eq!(handle.dag_hash(), Some(hash.as_str()));
        assert_eq!(handle.root_index(), Some(2));
    }

    #[test]
    fn goal_with_ir_attaches_a_populated_handle() {
        let handle = IrHandle::from_wire_dag("b".repeat(64), 0);
        let goal = Goal::smt(trivially_true_property()).with_ir(handle.clone());
        assert!(goal.ir.is_populated());
        assert_eq!(goal.ir, handle);
    }

    #[cfg(feature = "smt")]
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
                required: Soundness::Exact,
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

    /// Exhaustive integrity oracle for the per-qualifier minimum-soundness map.
    ///
    /// This is the primitive every later no-laundering guarantee rests on, so it
    /// gets its own table-driven negative+positive oracle: for each of the seven
    /// qualifiers, `Discharge::new` must be REJECTED at every soundness strictly
    /// below the qualifier's minimum, and ACCEPTED at/above it.
    ///
    /// The expected minimums are written out by hand here (NOT read back from
    /// `min_soundness`), so a silent edit to the production map is caught by this
    /// oracle rather than tautologically agreeing with itself.
    #[test]
    fn discharge_new_enforces_per_qualifier_minimum_soundness_exhaustively() {
        // Weakest-to-strongest, matching the `Soundness` `Ord`.
        let all_soundness = [
            Soundness::Untrusted,
            Soundness::Empirical,
            Soundness::SoundApproximate,
            Soundness::Exact,
        ];

        // (qualifier, its expected minimum soundness floor) for all 7 kinds.
        // The minimums are written out by hand, NOT read from `min_soundness`,
        // so a silent edit to the production map is caught here.
        let table = [
            (Qualifier::Exact, Soundness::Exact),
            (Qualifier::CertificateBearing, Soundness::Exact),
            (Qualifier::DeltaComplete, Soundness::SoundApproximate),
            (
                Qualifier::SpecialFunctionCertified,
                Soundness::SoundApproximate,
            ),
            (
                Qualifier::SoundOverApproximation,
                Soundness::SoundApproximate,
            ),
            (Qualifier::Fuzz, Soundness::Empirical),
            (Qualifier::Axiom, Soundness::Untrusted),
        ];

        // Guard: the table must cover every qualifier kind exactly once, so a
        // newly added qualifier cannot slip past this oracle uncovered.
        assert_eq!(
            table.len(),
            7,
            "the qualifier vocabulary is closed at 7 kinds; update the table if it changes"
        );

        for (qualifier, expected_min) in table {
            for soundness in all_soundness {
                let result = Discharge::new(
                    soundness,
                    QualifierSet::from_iter_kinds([qualifier]),
                    TierBResult::Proved,
                    serde_json::json!({}),
                );
                if soundness < expected_min {
                    let err = result
                        .expect_err("below-minimum soundness must be rejected by Discharge::new");
                    assert_eq!(
                        err,
                        DischargeError::QualifierExceedsSoundness {
                            qualifier: qualifier.as_str(),
                            required: expected_min,
                            soundness,
                        },
                        "qualifier {qualifier:?} at {soundness:?} must report its \
                         minimum {expected_min:?}"
                    );
                } else {
                    assert!(
                        result.is_ok(),
                        "qualifier {qualifier:?} must be ACCEPTED at/above its \
                         minimum {expected_min:?}, but {soundness:?} was rejected"
                    );
                }
            }
        }
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
    fn cvc5_engine_proves_a_trivial_goal_over_reals() {
        // chelis#422: the cvc5 decision is over the REALS, not machine
        // arithmetic, so a proved goal is `SoundApproximate` carrying
        // `RealArith` -- a sound over-approximation that discloses the gap --
        // NOT an exact decision.
        let engine = Cvc5Engine::new();
        let goal = Goal::smt(trivially_true_property());
        let discharge = engine.discharge(&goal, 5_000);
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
        assert!(discharge.qualifier_set().contains(Qualifier::RealArith));
        assert!(
            !discharge.qualifier_set().contains(Qualifier::Exact),
            "an over-reals proof must not claim exact machine soundness"
        );
    }

    // WI-16: with the carcara feature on, a Proved goal in the audited
    // fragment carries an independent-audit evidence dimension WITHOUT any
    // change to its soundness/qualifier. Under the chelis#422 taxonomy a cvc5
    // proof is `SoundApproximate` / `RealArith` (a decision over the reals, not
    // exact machine arithmetic); the auditor adds an audit dimension, it does
    // not move that badge.
    #[cfg(feature = "carcara")]
    #[test]
    fn cvc5_proved_goal_carries_a_confirmed_carcara_audit_dimension() {
        let engine = Cvc5Engine::new();
        // x == x is linear/EUF, so cvc5 emits an Alethe proof Carcara confirms.
        let goal = Goal::smt(trivially_true_property());
        let discharge = engine.discharge(&goal, 10_000);
        // The cvc5 result and its badge are unchanged by the audit: the auditor
        // adds an evidence dimension, it does not move soundness/qualifier.
        assert_eq!(*discharge.result(), TierBResult::Proved);
        assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
        assert!(discharge.qualifier_set().contains(Qualifier::RealArith));
        assert!(
            !discharge.qualifier_set().contains(Qualifier::Exact),
            "the audit must not promote an over-reals proof to an exact badge"
        );
        // The audit dimension is present and is a confirmation (either a full
        // `confirmed` or `confirmed_modulo_rewrites` depending on whether cvc5's
        // proof of `x == x` leans on a trusted rewrite leaf; both re-verify the
        // proof structure). It must NOT be a failure or absent.
        let status = discharge
            .evidence()
            .get("carcara_audit")
            .and_then(|v| v.get("status"))
            .and_then(|s| s.as_str())
            .unwrap_or_else(|| {
                panic!(
                    "evidence must carry a carcara_audit status, got {:?}",
                    discharge.evidence()
                )
            });
        assert!(
            status == "confirmed" || status == "confirmed_modulo_rewrites",
            "carcara_audit must be a confirmation, got status `{status}`"
        );
    }

    #[cfg(feature = "smt")]
    #[test]
    fn cvc5_engine_disproves_a_false_goal_over_reals() {
        let engine = Cvc5Engine::new();
        let goal = Goal::smt(false_property());
        let discharge = engine.discharge(&goal, 5_000);
        assert!(matches!(discharge.result(), TierBResult::Disproved(_)));
        assert_eq!(discharge.soundness(), Soundness::SoundApproximate);
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
