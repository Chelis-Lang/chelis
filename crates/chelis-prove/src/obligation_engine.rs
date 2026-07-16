//! Shared obligation execution engine (RFC D-PARITY).
//!
//! This is the ONE implementation of "run the derived producer
//! obligations of a module" that BOTH the CLI prove path and the
//! chelis-tide MCP `chelis_prove` tool call. It performs collection
//! (via [`crate::obligations`]), Tier B (via [`crate::tier_b_lower`] +
//! [`crate::tier_b`]) and Tier C (via the interpreter, evaluating the
//! obligation body over sampled producer inputs). Callers render the
//! [`ObligationOutcome`]s into their own surface (CLI NDJSON, tide MCP
//! envelope) — the verification work itself is shared, so a prove run
//! through tide is identical to the CLI on the same module (the parity
//! the cross-surface test locks).

use std::collections::BTreeMap;

use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_types::types::{Prim, Type};

use crate::composition::{
    AssumptionDischarge, AssumptionRecord, CompositeVerdict, DischargeMethod, FUZZ_TOLERANCE,
    NonVacuityRecord, NonVacuityStatus, rollup_composite,
};
use crate::discharge::QualifierSet;
use crate::obligations::{ObligationMeta, ObligationProperty, ProducedPosition};
use crate::opaque::{ConstEnv, OpaqueInvariant};
use crate::tier_b::TierBResult;
use crate::tier_b_lower::ProducerParamType;

/// The verification outcome of one obligation (or a collection error).
#[derive(Debug, Clone, PartialEq)]
pub struct ObligationOutcome {
    pub name: String,
    pub meta: ObligationMeta,
    pub status: ObligationStatus,
    pub proof_tier: ObligationTier,
    pub samples: usize,
    pub seed: u64,
    /// The deductive base discharge's `(soundness, qualifiers)` when an engine
    /// discharged the base (chelis#422); `None` for a fuzz-only or non-green
    /// base. Retained so the disclosed qualifier set can be recomposed.
    base_discharge: Option<(crate::discharge::Soundness, QualifierSet)>,
    pub counterexample: Option<serde_json::Value>,
    pub shrink_steps: usize,
    pub reason: Option<String>,
    pub assumptions: Vec<AssumptionRecord>,
    pub composite_verdict: CompositeVerdict,
    /// The discharged proposition (the invariant predicate this obligation
    /// proves over every produced value) in canonical Deep text, so a consumer
    /// displays exactly what was discharged rather than re-deriving it from the
    /// type definition (chelis#436). `None` for a producer-set declaration
    /// error that carries no invariant to render.
    pub goal: Option<String>,
}

impl ObligationOutcome {
    #[allow(clippy::too_many_arguments)]
    fn new(
        name: impl Into<String>,
        meta: ObligationMeta,
        status: ObligationStatus,
        proof_tier: ObligationTier,
        samples: usize,
        seed: u64,
        base_discharge: Option<(crate::discharge::Soundness, QualifierSet)>,
        counterexample: Option<serde_json::Value>,
        reason: Option<String>,
        assumptions: Vec<AssumptionRecord>,
    ) -> Self {
        let name = name.into();
        let (status, reason) = if status == ObligationStatus::Failed && counterexample.is_none() {
            (
                ObligationStatus::Unsupported,
                Some(reason.unwrap_or_else(|| {
                    "failed obligation outcome missing counterexample".to_string()
                })),
            )
        } else {
            (status, reason)
        };
        let assumptions = if assumptions.is_empty()
            && proof_tier != ObligationTier::None
            && status != ObligationStatus::Error
        {
            default_obligation_assumptions(
                &name,
                &meta,
                status,
                proof_tier,
                samples,
                seed,
                counterexample.as_ref(),
                reason.as_deref(),
                base_discharge_is_real_arithmetic(base_discharge.as_ref()),
            )
        } else {
            assumptions
        };
        let base = match status {
            ObligationStatus::Passed => match proof_tier {
                // chelis#422: project the SMT/deductive green base from the
                // discharge's own (soundness, qualifiers) -- threaded from the
                // dispatch -- so an over-reals proof reads
                // `proven_modulo_real_arithmetic` and a Beacon interval
                // discharge reads `sound_approximate`, never a flattened
                // `proven`. A green base without a discharge is a
                // covered-or-rejected `Unsupported`, never a silent proof.
                ObligationTier::Smt => match &base_discharge {
                    Some((soundness, qualifiers)) => {
                        crate::composition::base_verdict_from_discharge(*soundness, qualifiers)
                    }
                    None => CompositeVerdict::Unsupported,
                },
                // A fuzz-tier obligation BASE pass is empirically validated,
                // not proven: seed `FuzzBase` so it renders `fuzz_validated`
                // and never `proven_*` (chelis#422).
                ObligationTier::Fuzz if samples > 0 => {
                    crate::composition::base_verdict_from_discharge(
                        crate::discharge::Soundness::Empirical,
                        &QualifierSet::from_iter_kinds([crate::discharge::Qualifier::FuzzBase]),
                    )
                }
                _ => CompositeVerdict::Unsupported,
            },
            ObligationStatus::Failed => {
                if counterexample.is_some() {
                    // chelis#422 (symmetric): a disproof whose base discharge is
                    // over the reals (carries `RealArith`) is a HEDGED failure --
                    // the counterexample may be a false counterexample at machine
                    // arithmetic -- so it reads `disproved_modulo_real_arithmetic`.
                    // A disproof with no real-arithmetic base (a fuzz
                    // counterexample is a real machine-arithmetic witness) is a
                    // definite `Failed`.
                    if base_discharge_is_real_arithmetic(base_discharge.as_ref()) {
                        CompositeVerdict::DisprovedModuloRealArithmetic
                    } else {
                        CompositeVerdict::Failed
                    }
                } else {
                    CompositeVerdict::Unsupported
                }
            }
            ObligationStatus::Unsupported | ObligationStatus::Error => {
                CompositeVerdict::Unsupported
            }
        };
        let composite_verdict = rollup_composite(base, &assumptions);
        Self {
            name,
            meta,
            status,
            proof_tier,
            samples,
            seed,
            base_discharge,
            counterexample,
            shrink_steps: 0,
            reason,
            assumptions,
            composite_verdict,
            goal: None,
        }
    }

    fn with_shrink_steps(mut self, shrink_steps: usize) -> Self {
        self.shrink_steps = shrink_steps;
        self
    }

    /// Attach the discharged proposition's canonical text (chelis#436): the
    /// invariant predicate this obligation proves, so the goal travels with the
    /// record.
    fn with_goal(mut self, goal: impl Into<String>) -> Self {
        self.goal = Some(goal.into());
        self
    }

    /// The base discharge `(soundness, qualifiers)` this outcome's verdict was
    /// composed from, synthesized to match the base derivation above (chelis#422).
    fn effective_base_discharge(&self) -> Option<(crate::discharge::Soundness, QualifierSet)> {
        if self.status != ObligationStatus::Passed {
            return None;
        }
        match self.proof_tier {
            ObligationTier::Smt => self.base_discharge.clone(),
            ObligationTier::Fuzz if self.samples > 0 => Some((
                crate::discharge::Soundness::Empirical,
                QualifierSet::from_iter_kinds([crate::discharge::Qualifier::FuzzBase]),
            )),
            _ => None,
        }
    }

    /// The full disclosed qualifier set of this obligation's composed verdict,
    /// as sorted `snake_case` strings for the `qualifiers:[...]` JSON array. A
    /// reals-hedged disproof discloses `real_arithmetic` (symmetric to the proof
    /// side); every other non-green outcome discloses none.
    #[must_use]
    pub fn disclosed_qualifiers(&self) -> Vec<&'static str> {
        if self.composite_verdict == CompositeVerdict::DisprovedModuloRealArithmetic {
            return crate::composition::disclosed_qualifier_strings_for_base(
                CompositeVerdict::DisprovedModuloRealArithmetic,
                &self.assumptions,
            );
        }
        match self.effective_base_discharge() {
            Some((soundness, qualifiers)) => crate::composition::composed_qualifier_strings(
                soundness,
                &qualifiers,
                &self.assumptions,
            ),
            None => Vec::new(),
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn default_obligation_assumptions(
    name: &str,
    meta: &ObligationMeta,
    status: ObligationStatus,
    tier: ObligationTier,
    samples: usize,
    seed: u64,
    counterexample: Option<&serde_json::Value>,
    reason: Option<&str>,
    base_real_arithmetic: bool,
) -> Vec<AssumptionRecord> {
    let method = match tier {
        ObligationTier::Smt => DischargeMethod::Smt,
        ObligationTier::Fuzz => DischargeMethod::Fuzz,
        ObligationTier::None => return Vec::new(),
    };
    let evidence = match status {
        ObligationStatus::Passed if tier == ObligationTier::Smt => serde_json::json!({
            "status": "proved",
            "obligation": name,
            "arith_model": "real",
        }),
        ObligationStatus::Passed => serde_json::json!({
            "status": "validated",
            "obligation": name,
            "samples": samples,
            "seed": seed,
            "tolerance": FUZZ_TOLERANCE,
        }),
        // chelis#422 (symmetric to the `proved` branch): a disproof discharged
        // over the REALS carries `arith_model:"real"`, so the synthesized
        // self-discharge record folds as the reals-hedged `DisprovedOverReals`
        // -- consistent with the hedged base -- instead of a definite
        // `Disproved` that would collapse the hedge to a flat `failed`. A fuzz
        // disproof (a real machine-arithmetic witness) omits the marker and
        // stays definite.
        ObligationStatus::Failed if base_real_arithmetic => serde_json::json!({
            "status": "failed",
            "obligation": name,
            "counterexample": counterexample.cloned(),
            "arith_model": "real",
        }),
        ObligationStatus::Failed => serde_json::json!({
            "status": "failed",
            "obligation": name,
            "counterexample": counterexample.cloned(),
        }),
        ObligationStatus::Unsupported => serde_json::json!({
            "status": "unsupported",
            "obligation": name,
            "reason": reason.unwrap_or("unsupported"),
        }),
        ObligationStatus::Error => return Vec::new(),
    };
    let non_vacuity = if status == ObligationStatus::Passed {
        Some(NonVacuityRecord::established(serde_json::json!({
            "method": method.as_str(),
            "result": "sat",
            "assumption_count": 0,
            "trivial": tier == ObligationTier::Smt,
            "accepted_samples": if tier == ObligationTier::Fuzz { samples } else { 0 },
            "seed": seed,
        })))
    } else {
        None
    };
    vec![
        AssumptionRecord::new(
            name.to_string(),
            Some(AssumptionDischarge::new(method, evidence)),
            non_vacuity,
        )
        .with_source(meta.source_type.clone(), meta.producer.clone())
        // WI-8: stamp the prover-side discharge tier, keyed to the obligation.
        .with_discharge_tier(crate::composition::DischargeTier::new(
            method.engine(),
            method,
            Some(name.to_string()),
        )),
    ]
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObligationStatus {
    Passed,
    Failed,
    Unsupported,
    /// A collection-time declaration error (covered-or-rejected /
    /// signature rejection); not tied to a single producer outcome.
    Error,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObligationTier {
    Smt,
    Fuzz,
    /// No tier ran (a declaration error).
    None,
}

impl ObligationTier {
    #[must_use]
    pub fn as_str(self) -> &'static str {
        match self {
            ObligationTier::Smt => "smt",
            ObligationTier::Fuzz => "fuzz",
            ObligationTier::None => "none",
        }
    }
}

/// Options for the obligation run, mirroring the prove surface.
#[derive(Debug, Clone)]
pub struct ObligationRunOptions {
    pub seed: u64,
    pub samples: usize,
    pub smt_timeout_ms: u64,
    /// `"auto"` (Tier B then C), `"smt-only"`, `"fuzz-only"`.
    pub tier: String,
    /// Obligation-name selector (`--only`); `None` runs all.
    pub only: Option<String>,
    /// The acceptance-rate floor for opaque input-binder generation in
    /// Tier C (`--invariant-min-rate`, RFC D-STARVE). `0.0` disables the
    /// starvation classifier and restores the legacy exhaustion (`Error`)
    /// path.
    pub invariant_min_rate: f64,
}

impl Default for ObligationRunOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            samples: 100,
            smt_timeout_ms: 5000,
            tier: "auto".to_string(),
            only: None,
            invariant_min_rate: 0.01,
        }
    }
}

/// The result of an obligation run from module source.
#[derive(Debug, Clone)]
pub enum ObligationRunResult {
    /// The module type-checked; obligations were collected and run.
    Ran(Vec<ObligationOutcome>),
    /// The module did NOT type-check. Obligation verification is
    /// meaningless on a type-broken module (a rejectable producer can be
    /// hidden behind an unrelated type error), so prove surfaces the check
    /// diagnostics and is an Error -- never silent success (RT3-F2).
    CheckFailed(Vec<String>),
}

/// Run obligations directly from module SOURCE (Surf `.ch` text). This is
/// the entry the chelis-tide MCP tool and the CLI call: it desugars, runs
/// the checker for inferred return types, then runs the same engine
/// (RFC D-PARITY). Returns `Err` if the source does not parse;
/// `Ok(CheckFailed)` if it does not type-check; `Ok(Ran(..))` otherwise.
pub fn run_surf_source_obligations(
    source: &str,
    options: &ObligationRunOptions,
) -> Result<ObligationRunResult, String> {
    let decls = chelis_surf::parser::parse_str(source).map_err(|e| format!("parse: {e}"))?;
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let sigs: BTreeMap<String, Type> = match chelis_types::check_typed_program(&exprs) {
        Ok(checked) => checked
            .signature_inference()
            .functions
            .iter()
            .map(|(n, s)| (n.clone(), s.checked_signature.clone()))
            .collect(),
        // A type-broken module cannot have its obligations meaningfully
        // verified: the checker-inferred sigs are unavailable, so the
        // producer set would be empty and a violating producer hidden.
        // Surface the check diagnostics as a CheckFailed result (RT3-F2).
        Err(infer) => {
            let messages = infer
                .errors
                .iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>();
            return Ok(ObligationRunResult::CheckFailed(messages));
        }
    };
    Ok(ObligationRunResult::Ran(run_module_obligations(
        &exprs, &sigs, options,
    )))
}

/// Run obligations directly from Deep module SOURCE (`.dp` text). This is
/// the Deep sibling of [`run_surf_source_obligations`]: it is the entry the
/// chelis-tide MCP tool calls for a `source_kind:"deep"` module so a prove
/// through tide is identical to the CLI `chelis prove foo.dp` path
/// (`run_deep_obligations`). A `.dp` is already Deep, so it is parsed but NOT
/// desugared; the checker then runs for inferred return types (a type-broken
/// module surfaces a `CheckFailed`, never a silent zero-obligation pass --
/// RT3-F2 parity), and the SAME `run_module_obligations` the Surf source
/// entry reaches verifies each obligation. Returns `Err` if the source does
/// not parse; `Ok(CheckFailed)` if it does not type-check; `Ok(Ran(..))`
/// otherwise.
pub fn run_deep_source_obligations(
    source: &str,
    options: &ObligationRunOptions,
) -> Result<ObligationRunResult, String> {
    let exprs = chelis_deep::parser::parse_str(source).map_err(|e| format!("parse: {e}"))?;
    let sigs: BTreeMap<String, Type> = match chelis_types::check_typed_program(&exprs) {
        Ok(checked) => checked
            .signature_inference()
            .functions
            .iter()
            .map(|(n, s)| (n.clone(), s.checked_signature.clone()))
            .collect(),
        // A type-broken module cannot have its obligations meaningfully
        // verified: the checker-inferred sigs are unavailable, so the
        // producer set would be empty and a violating producer hidden.
        // Surface the check diagnostics as a CheckFailed result (RT3-F2).
        Err(infer) => {
            let messages = infer
                .errors
                .iter()
                .map(|e| e.message.clone())
                .collect::<Vec<_>>();
            return Ok(ObligationRunResult::CheckFailed(messages));
        }
    };
    Ok(ObligationRunResult::Ran(run_module_obligations(
        &exprs, &sigs, options,
    )))
}

/// Run all derived producer obligations of a desugared Deep program.
/// `sigs` is the checker-inferred def-name -> type map (from
/// `chelis_types::check_typed_program`). Returns one outcome per
/// obligation plus one `Error` outcome per collection error.
#[must_use]
pub fn run_module_obligations(
    exprs: &[Expr],
    sigs: &BTreeMap<String, Type>,
    options: &ObligationRunOptions,
) -> Vec<ObligationOutcome> {
    let (invariants, rejections) = crate::opaque::collect_opaque_invariants_and_rejections(exprs);
    if invariants.is_empty() && rejections.is_empty() {
        return Vec::new();
    }

    let mut out = Vec::new();
    // Covered-or-rejected: an invariant-carrying opaque type whose
    // representation the prover cannot model is surfaced as an `Error`
    // outcome, NEVER silently dropped. Dropping it would let a violating
    // exported producer pass `chelis prove` with zero obligations.
    for rej in &rejections {
        out.push(ObligationOutcome::new(
            format!("invariant:{}", rej.type_name),
            ObligationMeta {
                obligation_kind: "invariant_producer".to_string(),
                source_type: rej.type_name.clone(),
                producer: String::new(),
            },
            ObligationStatus::Error,
            ObligationTier::None,
            0,
            options.seed,
            None,
            None,
            Some(rej.reason.clone()),
            Vec::new(),
        ));
    }

    let consts = resolve_module_constants(exprs, &invariants);
    let collection = crate::obligations::collect_obligations(exprs, &invariants, sigs);

    for err in &collection.errors {
        out.push(ObligationOutcome::new(
            String::new(),
            ObligationMeta {
                obligation_kind: "invariant_producer".to_string(),
                source_type: String::new(),
                producer: String::new(),
            },
            ObligationStatus::Error,
            ObligationTier::None,
            0,
            options.seed,
            None,
            None,
            Some(err.to_string()),
            Vec::new(),
        ));
    }
    for ob in &collection.obligations {
        if let Some(only) = &options.only
            && !matches_filter(&ob.name, only)
        {
            continue;
        }
        let inv = invariants
            .iter()
            .find(|i| i.type_name == ob.source_type)
            .expect("obligation references a collected invariant");
        // chelis#436: the discharged proposition (the invariant predicate) travels
        // with every obligation outcome, so a consumer displays exactly what was
        // discharged rather than re-deriving it from the type definition.
        out.push(
            run_one(exprs, inv, &invariants, ob, sigs, &consts, options).with_goal(inv.goal_text()),
        );
    }
    out
}

#[allow(clippy::too_many_arguments)]
fn run_one(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    invariants: &[OpaqueInvariant],
    ob: &ObligationProperty,
    sigs: &BTreeMap<String, Type>,
    consts: &ConstEnv,
    options: &ObligationRunOptions,
) -> ObligationOutcome {
    // Tier B.
    if options.tier == "auto" || options.tier == "smt-only" {
        let pparams = producer_param_types(sigs, &ob.producer, invariants);
        if let Some(lowered) =
            crate::tier_b_lower::lower_obligation(exprs, inv, ob, &pparams, consts)
        {
            // Route the solve through the WI-9 discharge-engine registry. The
            // registry selects the SMT engine for this SMT goal by fitness, and
            // `into_result()` yields the identical TierBResult so the match arms
            // below are unchanged. The per-lane engine choice is internal to
            // `with_builtin_engines`: under `--features smt` it is the cvc5
            // engine (wrapping the same solve_property pipeline); in the default
            // build it is the solver-free solve_property engine (wrapping the
            // same direct solve_property call). Both lanes are byte-identical to
            // the pre-WI-9 dispatch, and no cvc5-named symbol leaks into the
            // solver-free default build.
            // chelis#422: keep the full `Discharge` -- its (soundness,
            // qualifiers) -- so the green base is projected from the engine's
            // own guarantee (over-reals -> proven_modulo_real_arithmetic; a
            // Beacon interval discharge -> sound_approximate), not flattened to
            // a hardcoded `proven`.
            let discharge = crate::engine_registry::DischargeRegistry::with_builtin_engines()
                .dispatch(
                    &crate::discharge::Goal::smt(lowered.property.clone()),
                    options.smt_timeout_ms,
                );
            let base_discharge = Some((discharge.soundness(), discharge.qualifier_set().clone()));
            let discharge_result = discharge.into_result();
            match discharge_result {
                TierBResult::Proved => {
                    let non_vacuity =
                        smt_non_vacuity_record(&lowered.property, options.smt_timeout_ms);
                    let reason = match non_vacuity.status {
                        NonVacuityStatus::Established => None,
                        NonVacuityStatus::Invalid | NonVacuityStatus::Unsupported => {
                            non_vacuity.reason.clone()
                        }
                    };
                    let assumptions = obligation_assumption_records(
                        ob,
                        AssumptionDischarge::new(
                            DischargeMethod::Smt,
                            serde_json::json!({
                                "status": "proved",
                                "obligation": ob.name,
                                "arith_model": "real",
                            }),
                        ),
                        non_vacuity,
                    );
                    let status = if reason.is_some() {
                        ObligationStatus::Unsupported
                    } else {
                        ObligationStatus::Passed
                    };
                    return outcome_with_assumptions(
                        ob,
                        status,
                        ObligationTier::Smt,
                        0,
                        options.seed,
                        base_discharge,
                        None,
                        reason,
                        assumptions,
                    );
                }
                TierBResult::Disproved(model) => {
                    // chelis#422 (symmetric): thread the discharge so a disproof
                    // over the reals (carries `RealArith`) reads
                    // `disproved_modulo_real_arithmetic` -- the counterexample may
                    // be a false counterexample at machine arithmetic -- rather
                    // than a flattened definite `failed`.
                    return outcome_with_assumptions(
                        ob,
                        ObligationStatus::Failed,
                        ObligationTier::Smt,
                        0,
                        options.seed,
                        base_discharge,
                        Some(model),
                        None,
                        Vec::new(),
                    );
                }
                TierBResult::Timeout => {
                    if options.tier == "smt-only" {
                        return outcome(
                            ob,
                            ObligationStatus::Unsupported,
                            ObligationTier::Smt,
                            0,
                            options.seed,
                            None,
                            Some("smt timeout".to_string()),
                        );
                    }
                }
                TierBResult::Unknown => {
                    if options.tier == "smt-only" {
                        return outcome(
                            ob,
                            ObligationStatus::Unsupported,
                            ObligationTier::Smt,
                            0,
                            options.seed,
                            None,
                            Some("smt unknown".to_string()),
                        );
                    }
                }
                TierBResult::Error(reason) => {
                    // The obligation did not lower to a valid SMT term (a
                    // wrong-arity intrinsic, RT5-F1; a transcendental cvc5 has
                    // no kind for, chelis#434). Surface it as Unsupported with
                    // the reason rather than letting the bad term abort cvc5.
                    // In `auto` mode fall through to Tier C; in `smt-only` it
                    // is terminal, so frame it as an honest capability boundary
                    // ("does not lower to the SMT tier"), not an internal bug.
                    if options.tier == "smt-only" {
                        return outcome(
                            ob,
                            ObligationStatus::Unsupported,
                            ObligationTier::Smt,
                            0,
                            options.seed,
                            None,
                            Some(format!(
                                "obligation does not lower to the SMT tier: {reason}"
                            )),
                        );
                    }
                }
            }
        }
    }
    if options.tier == "smt-only" {
        return outcome(
            ob,
            ObligationStatus::Unsupported,
            ObligationTier::Smt,
            0,
            options.seed,
            None,
            Some("obligation does not lower to Tier B".to_string()),
        );
    }
    // Tier C.
    run_tier_c(exprs, inv, invariants, ob, sigs, consts, options)
}

#[allow(clippy::too_many_arguments)]
fn run_tier_c(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    invariants: &[OpaqueInvariant],
    ob: &ObligationProperty,
    sigs: &BTreeMap<String, Type>,
    consts: &ConstEnv,
    options: &ObligationRunOptions,
) -> ObligationOutcome {
    let seed = options.seed;
    if ob.is_constant {
        return match eval_constant_obligation(exprs, inv, ob, consts) {
            Ok(true) => outcome(
                ob,
                ObligationStatus::Passed,
                ObligationTier::Fuzz,
                1,
                seed,
                None,
                None,
            ),
            Ok(false) => outcome(
                ob,
                ObligationStatus::Failed,
                ObligationTier::Fuzz,
                1,
                seed,
                None,
                None,
            ),
            Err(e) => outcome(
                ob,
                ObligationStatus::Error,
                ObligationTier::Fuzz,
                0,
                seed,
                None,
                Some(e),
            ),
        };
    }
    let Some(Type::Fn(arg_types, _)) = sigs.get(&ob.producer) else {
        return outcome(
            ob,
            ObligationStatus::Unsupported,
            ObligationTier::Fuzz,
            0,
            seed,
            None,
            Some("producer has no callable signature to sample".to_string()),
        );
    };
    // Classify each producer parameter.
    let mut kinds = Vec::new();
    for t in arg_types {
        match t {
            Type::Prim(p) => kinds.push(ArgKind::Scalar(prim_name(p))),
            Type::Tensor(dims, _) => {
                // Fixed-shape numeric tensor input.
                let lit_dims: Option<Vec<usize>> = dims
                    .iter()
                    .map(|d| match d {
                        chelis_types::types::Dim::Lit(n) if *n >= 0 => Some(*n as usize),
                        _ => None,
                    })
                    .collect();
                match lit_dims {
                    Some(d) => kinds.push(ArgKind::Tensor(d)),
                    None => {
                        return unsupported(
                            ob,
                            seed,
                            "producer has a symbolic-shape tensor parameter (Tier C V1)",
                        );
                    }
                }
            }
            Type::Adt(name, _) => match invariants.iter().find(|i| &i.type_name == name) {
                // An invariant-carrying opaque input: the update-shaped
                // inductive step. The input is generated to satisfy its
                // invariant (the assumption), via the tiered generator.
                Some(input_inv) => kinds.push(ArgKind::Opaque(input_inv.clone())),
                None => {
                    return unsupported(
                        ob,
                        seed,
                        "producer has a non-invariant ADT parameter (Tier C V1)",
                    );
                }
            },
            _ => {
                return unsupported(
                    ob,
                    seed,
                    "producer has an unsupported parameter (Tier C V1)",
                );
            }
        }
    }
    let names = producer_param_names(exprs, &ob.producer);
    if names.len() != kinds.len() {
        return unsupported(ob, seed, "producer parameter arity mismatch");
    }

    let module_source = chelis_deep::printer::print_canonical(exprs);
    let mut rng = Lcg::new(seed);
    let gen_budget = options.samples.saturating_mul(100).max(200);
    for n in 0..options.samples {
        // Build the per-arg values. An opaque input is generated to satisfy
        // its invariant (the assumption); on starvation we report it.
        let mut arg_values = Vec::new();
        for (name, kind) in names.iter().zip(&kinds) {
            match kind {
                ArgKind::Scalar(prim) => {
                    let v = sample_scalar(prim, &mut rng);
                    arg_values.push(ArgValue {
                        expr: scalar_lit(prim, v),
                        json: serde_json::json!(v),
                        name: name.clone(),
                    });
                }
                ArgKind::Tensor(dims) => {
                    let count: usize = dims.iter().product::<usize>().max(1);
                    let vals: Vec<f64> = (0..count).map(|_| rng.next_f64(-10.0, 10.0)).collect();
                    arg_values.push(ArgValue {
                        expr: crate::opaque::tensor_value_expr_pub(dims, "f32", &vals),
                        json: serde_json::json!(vals),
                        name: name.clone(),
                    });
                }
                ArgKind::Opaque(input_inv) => {
                    let mut grng = crate::opaque::GenRng::new(rng.next_u64());
                    let producers = generation_producers(exprs, sigs, input_inv);
                    match crate::opaque::generate_binder(
                        input_inv,
                        consts,
                        &module_source,
                        &producers,
                        &mut grng,
                        options.invariant_min_rate,
                        gen_budget,
                    ) {
                        Ok(binder) => {
                            let json = serde_json::to_value(&binder.env)
                                .unwrap_or(serde_json::json!(null));
                            arg_values.push(ArgValue {
                                expr: binder.value_expr,
                                json,
                                name: name.clone(),
                            });
                        }
                        Err(diag) => {
                            // Generator starvation for the input binder.
                            if options.invariant_min_rate == 0.0 {
                                // Floor disabled: legacy exhaustion => Error.
                                return outcome(
                                    ob,
                                    ObligationStatus::Error,
                                    ObligationTier::Fuzz,
                                    n,
                                    seed,
                                    None,
                                    Some(format!(
                                        "generator exhausted for input binder of type `{}`",
                                        diag.type_name
                                    )),
                                );
                            }
                            return outcome(
                                ob,
                                ObligationStatus::Unsupported,
                                ObligationTier::Fuzz,
                                n,
                                seed,
                                None,
                                Some(diag.message()),
                            );
                        }
                    }
                }
            }
        }
        match eval_obligation_body_values(exprs, inv, ob, &arg_values, consts) {
            Ok(true) => {}
            Ok(false) => {
                let (shrunk, shrink_steps) =
                    shrink_obligation_counterexample(exprs, inv, ob, &kinds, consts, arg_values);
                let cx = obligation_counterexample(&shrunk);
                return outcome(
                    ob,
                    ObligationStatus::Failed,
                    ObligationTier::Fuzz,
                    n + 1,
                    seed,
                    Some(serde_json::Value::Object(cx)),
                    None,
                )
                .with_shrink_steps(shrink_steps);
            }
            Err(e) => {
                return outcome(
                    ob,
                    ObligationStatus::Error,
                    ObligationTier::Fuzz,
                    n,
                    seed,
                    None,
                    Some(e),
                );
            }
        }
    }
    outcome(
        ob,
        ObligationStatus::Passed,
        ObligationTier::Fuzz,
        options.samples,
        seed,
        None,
        None,
    )
}

/// A producer parameter kind for Tier C sampling.
#[derive(Clone)]
enum ArgKind {
    Scalar(String),
    Tensor(Vec<usize>),
    Opaque(OpaqueInvariant),
}

/// A sampled argument value: the Deep value expr passed to the producer
/// call, a JSON repr for counterexamples, and the param name.
#[derive(Clone)]
struct ArgValue {
    expr: Expr,
    json: serde_json::Value,
    name: String,
}

const MAX_OBLIGATION_SHRINK_STEPS: usize = 64;

fn obligation_counterexample(args: &[ArgValue]) -> serde_json::Map<String, serde_json::Value> {
    args.iter()
        .map(|a| (a.name.clone(), a.json.clone()))
        .collect()
}

fn shrink_obligation_counterexample(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    kinds: &[ArgKind],
    consts: &ConstEnv,
    mut args: Vec<ArgValue>,
) -> (Vec<ArgValue>, usize) {
    let mut steps = 0usize;
    while steps < MAX_OBLIGATION_SHRINK_STEPS {
        let mut changed = false;
        for index in 0..args.len() {
            let Some(kind) = kinds.get(index) else {
                continue;
            };
            for candidate in obligation_shrink_candidates(&args[index], kind) {
                if candidate.json == args[index].json {
                    continue;
                }
                let mut trial = args.clone();
                trial[index] = candidate;
                if matches!(
                    eval_obligation_body_values(exprs, inv, ob, &trial, consts),
                    Ok(false)
                ) {
                    args = trial;
                    steps += 1;
                    changed = true;
                    break;
                }
            }
            if changed || steps >= MAX_OBLIGATION_SHRINK_STEPS {
                break;
            }
        }
        if !changed {
            break;
        }
    }
    (args, steps)
}

fn obligation_shrink_candidates(value: &ArgValue, kind: &ArgKind) -> Vec<ArgValue> {
    match kind {
        ArgKind::Scalar(prim) if prim == "bool" => value
            .json
            .as_bool()
            .or_else(|| value.json.as_f64().map(|current| current != 0.0))
            .and_then(|current| {
                current.then(|| ArgValue {
                    expr: scalar_lit(prim, 0.0),
                    json: serde_json::json!(false),
                    name: value.name.clone(),
                })
            })
            .into_iter()
            .collect(),
        ArgKind::Scalar(prim) if crate::opaque::is_int_width(prim) => value
            .json
            .as_f64()
            .map(|current| {
                let current = current as i64;
                let mut candidates = Vec::new();
                push_unique_obligation_int_candidate(&mut candidates, value, prim, current, 0);
                push_unique_obligation_int_candidate(
                    &mut candidates,
                    value,
                    prim,
                    current,
                    current / 2,
                );
                push_unique_obligation_int_candidate(
                    &mut candidates,
                    value,
                    prim,
                    current,
                    current.signum(),
                );
                candidates
            })
            .unwrap_or_default(),
        ArgKind::Scalar(prim) => value
            .json
            .as_f64()
            .map(|current| {
                let mut candidates = Vec::new();
                push_unique_obligation_float_candidate(&mut candidates, value, prim, current, 0.0);
                push_unique_obligation_float_candidate(
                    &mut candidates,
                    value,
                    prim,
                    current,
                    current / 2.0,
                );
                candidates
            })
            .unwrap_or_default(),
        ArgKind::Tensor(dims) => value
            .json
            .as_array()
            .map(|items| {
                let data = items
                    .iter()
                    .filter_map(serde_json::Value::as_f64)
                    .collect::<Vec<_>>();
                if data.len() != dims.iter().product::<usize>().max(1) {
                    return Vec::new();
                }
                let mut candidates = Vec::new();
                if data.iter().any(|value| value.abs() > f64::EPSILON) {
                    let zeros = vec![0.0; data.len()];
                    candidates.push(obligation_tensor_arg(&value.name, dims, &zeros));
                }
                let halves = data.iter().map(|value| value / 2.0).collect::<Vec<_>>();
                if halves
                    .iter()
                    .zip(&data)
                    .any(|(candidate, current)| (candidate - current).abs() > f64::EPSILON)
                {
                    candidates.push(obligation_tensor_arg(&value.name, dims, &halves));
                }
                candidates
            })
            .unwrap_or_default(),
        ArgKind::Opaque(_) => Vec::new(),
    }
}

fn push_unique_obligation_int_candidate(
    candidates: &mut Vec<ArgValue>,
    value: &ArgValue,
    prim: &str,
    current: i64,
    candidate: i64,
) {
    if candidate == current {
        return;
    }
    let Some((lo, hi)) = crate::opaque::int_sample_bounds(prim) else {
        return;
    };
    let candidate = candidate.clamp(lo, hi);
    if candidates
        .iter()
        .any(|arg| arg.json.as_f64() == Some(candidate as f64))
    {
        return;
    }
    candidates.push(ArgValue {
        expr: scalar_lit(prim, candidate as f64),
        json: serde_json::json!(candidate as f64),
        name: value.name.clone(),
    });
}

fn push_unique_obligation_float_candidate(
    candidates: &mut Vec<ArgValue>,
    value: &ArgValue,
    prim: &str,
    current: f64,
    candidate: f64,
) {
    if (candidate - current).abs() <= f64::EPSILON
        || candidates.iter().any(|arg| {
            arg.json
                .as_f64()
                .is_some_and(|prior| (prior - candidate).abs() <= f64::EPSILON)
        })
    {
        return;
    }
    candidates.push(ArgValue {
        expr: scalar_lit(prim, candidate),
        json: serde_json::json!(candidate),
        name: value.name.clone(),
    });
}

fn obligation_tensor_arg(name: &str, dims: &[usize], values: &[f64]) -> ArgValue {
    ArgValue {
        expr: crate::opaque::tensor_value_expr_pub(dims, "f32", values),
        json: serde_json::json!(values),
        name: name.to_string(),
    }
}

fn unsupported(ob: &ObligationProperty, seed: u64, reason: &str) -> ObligationOutcome {
    outcome(
        ob,
        ObligationStatus::Unsupported,
        ObligationTier::Fuzz,
        0,
        seed,
        None,
        Some(reason.to_string()),
    )
}

fn smt_non_vacuity_record(
    smt_prop: &crate::tier_b::SmtProperty,
    timeout_ms: u64,
) -> NonVacuityRecord {
    if smt_prop.preconditions.is_empty() {
        return NonVacuityRecord::established(serde_json::json!({
            "solver": "cvc5",
            "result": "sat",
            "assumption_count": 0,
            "trivial": true,
        }));
    }
    match crate::tier_b::check_assumptions_satisfiable(smt_prop, timeout_ms) {
        crate::tier_b::AssumptionSatisfiability::Sat(model) => {
            NonVacuityRecord::established(serde_json::json!({
                "solver": "cvc5",
                "result": "sat",
                "assumption_count": smt_prop.preconditions.len(),
                "model": model,
            }))
        }
        crate::tier_b::AssumptionSatisfiability::Unsat => NonVacuityRecord::invalid(
            "non_vacuity_invalid: assumptions are unsatisfiable",
            serde_json::json!({
                "solver": "cvc5",
                "result": "unsat",
                "assumption_count": smt_prop.preconditions.len(),
            }),
        ),
        crate::tier_b::AssumptionSatisfiability::Timeout => NonVacuityRecord::unsupported(
            "non_vacuity_unestablished: smt timeout",
            serde_json::json!({
                "solver": "cvc5",
                "result": "timeout",
                "assumption_count": smt_prop.preconditions.len(),
            }),
        ),
        crate::tier_b::AssumptionSatisfiability::Unknown => NonVacuityRecord::unsupported(
            "non_vacuity_unestablished: smt unknown",
            serde_json::json!({
                "solver": "cvc5",
                "result": "unknown",
                "assumption_count": smt_prop.preconditions.len(),
            }),
        ),
        crate::tier_b::AssumptionSatisfiability::Error(reason) => NonVacuityRecord::unsupported(
            format!("non_vacuity_unestablished: smt lowering error: {reason}"),
            serde_json::json!({
                "solver": "cvc5",
                "result": "error",
                "assumption_count": smt_prop.preconditions.len(),
                "reason": reason,
            }),
        ),
    }
}

fn obligation_assumption_records(
    ob: &ObligationProperty,
    discharge: AssumptionDischarge,
    non_vacuity: NonVacuityRecord,
) -> Vec<AssumptionRecord> {
    // WI-8: stamp the prover-side discharge tier from the discharge's method
    // (which engine + guarantee), keyed to the obligation's source identity.
    let tier = crate::composition::DischargeTier::new(
        discharge.method.engine(),
        discharge.method,
        Some(ob.name.clone()),
    );
    vec![
        AssumptionRecord::new(ob.name.clone(), Some(discharge), Some(non_vacuity))
            .with_source(ob.meta.source_type.clone(), ob.meta.producer.clone())
            .with_discharge_tier(tier),
    ]
}

/// Public entry: the exported base producers of `input_inv`'s type usable
/// for constructor-based generation, computing the inferred signatures
/// from the program. Used by the user-property injection path (D-INJECT)
/// so it shares the obligation engine's producer-resolution rules.
#[must_use]
pub fn generation_producers_for(
    exprs: &[Expr],
    input_inv: &OpaqueInvariant,
) -> Vec<crate::opaque::GenProducer> {
    let sigs: BTreeMap<String, Type> = match chelis_types::check_typed_program(exprs) {
        Ok(checked) => checked
            .signature_inference()
            .functions
            .iter()
            .map(|(n, s)| (n.clone(), s.checked_signature.clone()))
            .collect(),
        Err(_) => BTreeMap::new(),
    };
    generation_producers(exprs, &sigs, input_inv)
}

/// The exported producers of the input type usable for constructor-based
/// generation (RFC D-STARVE tier 2). A base producer (no input of the
/// type) whose result is the input type, with scalar/tensor params.
fn generation_producers(
    exprs: &[Expr],
    sigs: &BTreeMap<String, Type>,
    input_inv: &OpaqueInvariant,
) -> Vec<crate::opaque::GenProducer> {
    use crate::opaque::{GenParamKind, GenProducer};
    let exports = crate::obligations::collect_exports(exprs);
    let mut out = Vec::new();
    for name in &exports {
        let Some(Type::Fn(args, ret)) = sigs.get(name) else {
            continue;
        };
        // The result must be the input type directly or Option-wrapped,
        // and the inputs must be raw (scalar/tensor) — a base producer.
        let (option_wrapped, ok_ret) = match ret.as_ref() {
            Type::Adt(n, _) if n == &input_inv.type_name => (false, true),
            Type::Adt(n, inner) if n == "Option" => match inner.first() {
                Some(Type::Adt(m, _)) if m == &input_inv.type_name => (true, true),
                _ => (false, false),
            },
            _ => (false, false),
        };
        if !ok_ret {
            continue;
        }
        let mut kinds = Vec::new();
        let mut raw_ok = true;
        for a in args {
            match a {
                Type::Prim(p) => kinds.push(GenParamKind::Scalar(prim_name(p))),
                Type::Tensor(dims, _) => {
                    let lit: Option<Vec<usize>> = dims
                        .iter()
                        .map(|d| match d {
                            chelis_types::types::Dim::Lit(n) if *n >= 0 => Some(*n as usize),
                            _ => None,
                        })
                        .collect();
                    if let Some(d) = lit {
                        kinds.push(GenParamKind::Tensor {
                            dims: d,
                            precision: "f32".to_string(),
                        });
                    } else {
                        raw_ok = false;
                        break;
                    }
                }
                // A producer that itself takes the opaque type is
                // update-shaped, not a base producer: skip for generation.
                _ => {
                    raw_ok = false;
                    break;
                }
            }
        }
        if !raw_ok {
            continue;
        }
        let param_names = producer_param_names(exprs, name);
        out.push(GenProducer {
            name: name.clone(),
            param_names,
            param_kinds: kinds,
            option_wrapped,
        });
    }
    out
}

/// Whether a threaded base discharge is an over-the-reals decision (carries
/// [`crate::discharge::Qualifier::RealArith`]). Used to hedge a disproof's
/// failure badge symmetrically to the proof side (chelis#422).
fn base_discharge_is_real_arithmetic(
    base_discharge: Option<&(crate::discharge::Soundness, QualifierSet)>,
) -> bool {
    base_discharge
        .is_some_and(|(_, qualifiers)| qualifiers.contains(crate::discharge::Qualifier::RealArith))
}

#[allow(clippy::too_many_arguments)]
fn outcome(
    ob: &ObligationProperty,
    status: ObligationStatus,
    tier: ObligationTier,
    samples: usize,
    seed: u64,
    counterexample: Option<serde_json::Value>,
    reason: Option<String>,
) -> ObligationOutcome {
    outcome_with_assumptions(
        ob,
        status,
        tier,
        samples,
        seed,
        None,
        counterexample,
        reason,
        Vec::new(),
    )
}

#[allow(clippy::too_many_arguments)]
fn outcome_with_assumptions(
    ob: &ObligationProperty,
    status: ObligationStatus,
    tier: ObligationTier,
    samples: usize,
    seed: u64,
    base_discharge: Option<(crate::discharge::Soundness, QualifierSet)>,
    counterexample: Option<serde_json::Value>,
    reason: Option<String>,
    assumptions: Vec<AssumptionRecord>,
) -> ObligationOutcome {
    ObligationOutcome::new(
        ob.name.clone(),
        ob.meta.clone(),
        status,
        tier,
        samples,
        seed,
        base_discharge,
        counterexample,
        reason,
        assumptions,
    )
}

fn producer_param_types(
    sigs: &BTreeMap<String, Type>,
    producer: &str,
    invariants: &[OpaqueInvariant],
) -> Vec<(String, ProducerParamType)> {
    match sigs.get(producer) {
        Some(Type::Fn(args, _)) => args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let pt = match a {
                    Type::Prim(p) => ProducerParamType::Scalar(prim_name(p)),
                    // An invariant-carrying opaque input parameter: the
                    // update-shaped inductive step (D-SOUND). Resolve its
                    // invariant model so Tier B can flatten + inject it.
                    Type::Adt(name, _) => {
                        match invariants.iter().find(|inv| &inv.type_name == name) {
                            Some(inv) => ProducerParamType::Opaque(inv.clone()),
                            None => ProducerParamType::Other,
                        }
                    }
                    _ => ProducerParamType::Other,
                };
                (format!("__arg{i}"), pt)
            })
            .collect(),
        _ => vec![],
    }
}

fn prim_name(p: &Prim) -> String {
    // Use the canonical `Prim::name()` spelling (the single type-system
    // source decode.rs also reads), NOT `format!("{p:?}").to_lowercase()`.
    // The Debug spelling coincidentally lowercases to the canonical name
    // for every variant today, but a future variant whose Debug spelling
    // diverges from its canonical name would feed a wrong dtype string into
    // scalar sampling / tensor-precision selection.
    p.name().to_string()
}

fn matches_filter(name: &str, pattern: &str) -> bool {
    if pattern == name {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    name.contains(pattern)
}

// --- constant resolution + obligation body evaluation (shared) ---

fn resolve_module_constants(exprs: &[Expr], invariants: &[OpaqueInvariant]) -> ConstEnv {
    let mut env = ConstEnv::new();
    let mut referenced: Vec<String> = Vec::new();
    // Constants referenced by the invariant predicates...
    for inv in invariants {
        for v in crate::predicate_free_vars(&inv.predicate) {
            if v != inv.binder && !referenced.contains(&v) {
                referenced.push(v);
            }
        }
    }
    // ...and the in-module zero-arg scalar defs (value bindings or
    // zero-parameter functions) a producer GUARD may compare against
    // (CR-8: `def hi() -> f32 = 1.0; ... if x <= hi() ...`, or the value
    // binding `hi = 1.0; ... x <= hi ...`). Resolving them here makes the
    // Tier B `reduce` pass inline the constant into the guard.
    for name in module_zero_arg_scalar_defs(exprs) {
        if !referenced.contains(&name) {
            referenced.push(name);
        }
    }
    if referenced.is_empty() {
        return env;
    }
    let source = chelis_deep::printer::print_canonical(exprs);
    for name in referenced {
        if let Some(value) = eval_scalar_const(&source, &name) {
            env.insert(name, value);
        }
    }
    env
}

/// Names of in-module defs that are zero-arg constants: a value binding
/// `(def name <value>)` whose value is not a `fn`, or a zero-parameter
/// function `(def name (fn (params) ...))` with no params. The caller
/// resolves each to a scalar by evaluation (non-scalar ones simply fail to
/// resolve and are dropped).
fn module_zero_arg_scalar_defs(exprs: &[Expr]) -> Vec<String> {
    fn walk(exprs: &[Expr], out: &mut Vec<String>) {
        for expr in exprs {
            if list_tag(expr) == Some("def")
                && let Some(name) = node_children(expr).first().and_then(sym_text)
                && let Some(body) = node_children(expr).get(1)
            {
                let is_zero_arg = if list_tag(body) == Some("fn") {
                    // A fn with an empty params node.
                    node_children(body)
                        .first()
                        .is_some_and(|p| node_children(p).is_empty())
                } else {
                    // A non-fn value binding.
                    true
                };
                if is_zero_arg && !out.iter().any(|n| n == name) {
                    out.push(name.to_string());
                }
            }
            if let Expr::List(l, _) = expr {
                walk(&l.elements[2.min(l.elements.len())..], out);
            }
        }
    }
    let mut out = Vec::new();
    walk(exprs, &mut out);
    out
}

fn eval_scalar_const(source: &str, name: &str) -> Option<f64> {
    // An in-module constant may be a value binding (`(var name)`) or a
    // zero-argument constant function (`(app (var name))`, the desugaring
    // of `def eps() -> f32 = 0.01`).
    let exprs: Vec<Expr> = chelis_deep::parser::parse_str(source)
        .ok()?
        .iter()
        .map(strip_invariant_meta)
        .collect();

    // Fast path: if the const def body is a plain literal, read it
    // directly (CR-8). A top-level value binding `hi = 1.0` is itself a
    // root, so the eval-probe path below trips the lowered-root-count
    // selection mismatch; extracting the literal sidesteps that.
    if let Some(v) = literal_const_value(&exprs, name) {
        return Some(v);
    }
    for probe_body in [deep_node("app", vec![deep_var(name)]), deep_var(name)] {
        let probe = "__chelis_const_probe";
        let probe_def = node_def(probe, probe_body);
        let program = inject_const_probe(&exprs, probe_def);
        let deep_probe = chelis_deep::printer::print_canonical(&program);
        let Ok(result) = chelis_compiler_api::compiler::eval_selected(
            EvalRequest {
                source_kind: SourceKind::Deep,
                source: deep_probe,
                bindings: Default::default(),
            },
            &[probe.to_string()],
        ) else {
            continue;
        };
        if let [root] = result.roots.as_slice()
            && let ExecutionValue::Tensor { value } = &root.value
            && value.shape.is_empty()
            && value.data.len() == 1
        {
            return Some(value.data[0]);
        }
    }
    None
}

/// Read a constant def's value directly when its body is a plain numeric
/// literal: `(def name (lit {} <num>))` (a value binding) or
/// `(def name (fn (params) (lit {} <num>)))` (a zero-arg constant fn).
/// Returns `None` for any non-literal body (those go through evaluation).
fn literal_const_value(exprs: &[Expr], name: &str) -> Option<f64> {
    fn lit_number(expr: &Expr) -> Option<f64> {
        // A bare atom or a `(lit {} <num>)` node.
        match expr {
            Expr::Atom(Atom::Float(v), _) => Some(*v),
            Expr::Atom(Atom::Int(v), _) => Some(*v as f64),
            _ => {
                if list_tag(expr) == Some("lit") {
                    match node_children(expr).first() {
                        Some(Expr::Atom(Atom::Float(v), _)) => Some(*v),
                        Some(Expr::Atom(Atom::Int(v), _)) => Some(*v as f64),
                        _ => None,
                    }
                } else {
                    None
                }
            }
        }
    }
    fn find(exprs: &[Expr], name: &str) -> Option<f64> {
        for expr in exprs {
            if list_tag(expr) == Some("def")
                && node_children(expr).first().and_then(sym_text) == Some(name)
                && let Some(body) = node_children(expr).get(1)
            {
                // Value binding: the body is the literal.
                if let Some(v) = lit_number(body) {
                    return Some(v);
                }
                // Zero-arg constant fn: `(fn (params) <lit>)`.
                if list_tag(body) == Some("fn") {
                    let kids = node_children(body);
                    if kids.first().is_some_and(|p| node_children(p).is_empty())
                        && let Some(fn_body) = kids.get(1)
                        && let Some(v) = lit_number(fn_body)
                    {
                        return Some(v);
                    }
                }
            }
            if let Expr::List(l, _) = expr
                && let Some(v) = find(&l.elements[2.min(l.elements.len())..], name)
            {
                return Some(v);
            }
        }
        None
    }
    find(exprs, name)
}

/// Inject a probe def into the first `(module ...)` wrapper (or top level).
fn inject_const_probe(exprs: &[Expr], def: Expr) -> Vec<Expr> {
    let mut out = Vec::with_capacity(exprs.len());
    let mut injected = false;
    for expr in exprs {
        if !injected
            && list_tag(expr) == Some("module")
            && let Expr::List(l, span) = expr
        {
            let mut elements = l.elements.clone();
            elements.push(def.clone());
            out.push(Expr::List(List { elements }, *span));
            injected = true;
        } else {
            out.push(expr.clone());
        }
    }
    if !injected {
        out.push(def);
    }
    out
}

/// Validate a CONSTANT producer's obligation through the ONE produced-value
/// chokepoint (U1 review-3 unification). A constant producer's value is a
/// concrete record; evaluate it and validate its representation
/// structurally via [`validate_value`] -> [`validate_produced_env`], so a
/// non-finite constant field is rejected fail-CLOSED exactly like a sampled
/// producer's value (the old host-runtime predicate eval had no finiteness
/// guard).
fn eval_constant_obligation(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    consts: &ConstEnv,
) -> Result<bool, String> {
    let predicate = crate::opaque::lower_predicate_flattened(inv, &inv.binder, consts)
        .ok_or_else(|| "invariant predicate does not lower for validation".to_string())?;
    let module_source = chelis_deep::printer::print_canonical(exprs);
    let call = deep_var(&ob.producer);
    let value = eval_producer_value(&module_source, inv, call)?;
    validate_value(&value, &ob.position, inv, &predicate)
}

/// Insert `new_defs` into the `(module ...)` wrapper that contains the
/// deftype for `type_name`, so the synthetic obligation defs are in the
/// defining module (field access on the opaque type is then in-module and
/// legal). If no module wrapper is found (top-level program), append at
/// top level.
fn inject_into_defining_module(exprs: &[Expr], type_name: &str, new_defs: Vec<Expr>) -> Vec<Expr> {
    fn module_defines(expr: &Expr, type_name: &str) -> bool {
        if list_tag(expr) == Some("deftype")
            && node_children(expr).first().and_then(sym_text) == Some(type_name)
        {
            return true;
        }
        if let Expr::List(l, _) = expr {
            return l.elements.iter().any(|c| module_defines(c, type_name));
        }
        false
    }
    let mut out = Vec::with_capacity(exprs.len());
    let mut injected = false;
    for expr in exprs {
        if !injected
            && list_tag(expr) == Some("module")
            && module_defines(expr, type_name)
            && let Expr::List(l, span) = expr
        {
            let mut elements = l.elements.clone();
            elements.extend(new_defs.clone());
            out.push(Expr::List(List { elements }, *span));
            injected = true;
        } else {
            out.push(expr.clone());
        }
    }
    if !injected {
        out.extend(new_defs);
    }
    out
}

/// Evaluate the obligation over richer argument VALUES (opaque records,
/// tensors, scalars) by EVALUATING THE PRODUCER and validating the
/// produced value's representation against the invariant through the ONE
/// shared chokepoint [`validate_value`] -> [`validate_produced_env`]
/// (U1 review-3 unification).
///
/// There is no separate scalar branch. The historical all-scalar branch
/// routed through the host runtime predicate eval, which never reached a
/// finiteness guard, so a NaN scalar field under a `!=`/`not(==)`
/// invariant shipped fail-OPEN (`NaN != C == true`). Every produced value
/// -- scalar-only, tensor-bearing, or nested-record -- is now evaluated
/// structurally and validated through the single chokepoint, which walks
/// EVERY representation leaf, rejects any non-finite leaf fail-CLOSED, then
/// evaluates the strict invariant predicate. This matches the generator's
/// `validate_env` (same finiteness helper, same strict evaluator), so
/// proposal and acceptance agree by construction.
fn eval_obligation_body_values(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    args: &[ArgValue],
    consts: &ConstEnv,
) -> Result<bool, String> {
    let call = {
        let mut app = vec![deep_var(&ob.producer)];
        for a in args {
            app.push(a.expr.clone());
        }
        deep_node("app", app)
    };

    let predicate = crate::opaque::lower_predicate_flattened(inv, &inv.binder, consts)
        .ok_or_else(|| "invariant predicate does not lower for validation".to_string())?;
    let module_source = chelis_deep::printer::print_canonical(exprs);
    // Evaluate the producer ONCE and validate the produced value's
    // representation structurally against the position + invariant
    // (CR-1 / CR-4 / CR-6): the structured ExecutionValue already
    // represents Option as `Adt{ctor:Some|None}`, tuples as `Tuple`, and
    // records as `Adt{ctor, fields}`, so the inner position is applied
    // recursively and None is a real discriminant -- no NaN sentinel and
    // no spurious record-read off a tuple.
    let value = eval_producer_value(&module_source, inv, call)?;
    validate_value(&value, &ob.position, inv, &predicate)
}

/// Evaluate the producer call inside the defining module (invariant
/// metadata stripped so the shape-sensitive predicate does not block IR
/// lowering) and return the full structured result value.
fn eval_producer_value(
    module_source: &str,
    inv: &OpaqueInvariant,
    call: Expr,
) -> Result<ExecutionValue, String> {
    let probe = "__chelis_value_probe";
    let probe_def = node_def(probe, call);
    let program = inject_into_module_stripped(module_source, &inv.type_name, vec![probe_def])?;
    let source = chelis_deep::printer::print_canonical(&program);
    let result = chelis_compiler_api::compiler::eval_selected(
        EvalRequest {
            source_kind: SourceKind::Deep,
            source,
            bindings: Default::default(),
        },
        &[probe.to_string()],
    )
    .map_err(|e| {
        e.errors
            .iter()
            .map(|d| d.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    match result.roots.as_slice() {
        [root] => Ok(root.value.clone()),
        _ => Err("producer value probe did not return one root".to_string()),
    }
}

/// Validate a produced [`ExecutionValue`] at `position` against the
/// invariant. The structured value carries Option/tuple/record structure
/// directly, so we traverse it (CR-1: the inner position is applied;
/// CR-4/CR-6: a `None` result is the real `Adt{ctor:"None"}` discriminant,
/// never a NaN sentinel, so a legitimate NaN inside a `Some` record fails
/// the invariant rather than passing vacuously).
fn validate_value(
    value: &ExecutionValue,
    position: &ProducedPosition,
    inv: &OpaqueInvariant,
    predicate: &crate::solver::SmtExpr,
) -> Result<bool, String> {
    match position {
        ProducedPosition::Direct => validate_produced_env(value, inv, predicate),
        ProducedPosition::InsideOption(inner) => match value {
            ExecutionValue::Adt { ctor, fields } if ctor == "None" => {
                let _ = fields;
                Ok(true) // None result: vacuously satisfied.
            }
            ExecutionValue::Adt { ctor, fields } if ctor == "Some" => {
                let payload = fields
                    .first()
                    .ok_or_else(|| "Some has no payload".to_string())?;
                validate_value(payload, inner, inv, predicate)
            }
            other => Err(format!(
                "expected an Option value at an Option position, got {other:?}"
            )),
        },
        ProducedPosition::TupleComponents(comps) => {
            let ExecutionValue::Tuple { value: items } = value else {
                return Err(format!(
                    "expected a tuple value at a tuple position, got {value:?}"
                ));
            };
            for (idx, inner) in comps {
                let comp = items
                    .get(*idx)
                    .ok_or_else(|| format!("tuple has no component {idx}"))?;
                if !validate_value(comp, inner, inv, predicate)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }
}

/// Flatten an opaque record [`ExecutionValue`] (`Adt{ctor, fields}` in the
/// type's declared field order) into the dotted-path env the lowered
/// predicate reads. A NaN scalar/element is kept as-is so the strict
/// invariant comparison fails on it (CR-4/CR-6 fail-closed).
fn opaque_record_env(
    value: &ExecutionValue,
    inv: &OpaqueInvariant,
) -> Result<BTreeMap<String, f64>, String> {
    let ExecutionValue::Adt { ctor, fields } = value else {
        return Err(format!(
            "expected an opaque record value at a Direct position, got {value:?}"
        ));
    };
    // Compare the produced value's ctor to the declared one (matching how
    // the Option/tuple arms in `validate_value` compare ctor), FAIL-CLOSED
    // on mismatch. Without this, a Direct-position value that is a DIFFERENT
    // same-arity ADT than `inv.ctor_name` would be flattened field-by-field
    // and validated as if it were the opaque record (a spurious Passed).
    // The producer return type is checker-pinned so a wrong-ctor value
    // should not reach here, but the asymmetry would be a latent fail-open.
    if ctor != &inv.ctor_name {
        return Err(format!(
            "opaque record ctor mismatch: produced `{ctor}` but the invariant declares `{}`",
            inv.ctor_name
        ));
    }
    if fields.len() != inv.fields.len() {
        return Err(format!(
            "opaque record arity mismatch: {} runtime fields vs {} declared",
            fields.len(),
            inv.fields.len()
        ));
    }
    let mut env = BTreeMap::new();
    for ((fname, fty), fval) in inv.fields.iter().zip(fields.iter()) {
        let field_path = format!("{}.{}", inv.binder, fname);
        flatten_field_value(fval, fty, &field_path, &mut env)?;
    }
    Ok(env)
}

fn flatten_field_value(
    value: &ExecutionValue,
    fty: &crate::opaque::FieldType,
    field_path: &str,
    env: &mut BTreeMap<String, f64>,
) -> Result<(), String> {
    match fty {
        crate::opaque::FieldType::Tensor { dims, .. } => {
            let ExecutionValue::Tensor { value } = value else {
                return Err("tensor field is not a tensor value".to_string());
            };
            let count = dims.iter().product::<usize>().max(1);
            if value.data.len() != count {
                return Err("tensor field shape mismatch".to_string());
            }
            for (i, v) in value.data.iter().enumerate() {
                env.insert(format!("{field_path}.{i}"), *v);
            }
        }
        crate::opaque::FieldType::Scalar(_) => {
            let v = match value {
                ExecutionValue::Float64 { value } => *value,
                ExecutionValue::Int64 { value } => *value as f64,
                ExecutionValue::Bool { value } => {
                    if *value {
                        1.0
                    } else {
                        0.0
                    }
                }
                // A 1-element tensor is a legitimate scalar carrier. A
                // MULTI-element tensor is a declared-vs-produced shape drift:
                // FAIL-CLOSED naming the mismatch rather than silently using
                // `data[0]` and dropping the rest (which could hide a NaN or
                // out-of-band remaining element that is never checked).
                ExecutionValue::Tensor { value } if value.data.len() == 1 => value.data[0],
                ExecutionValue::Tensor { value } => {
                    return Err(format!(
                        "scalar field shape mismatch: declared Scalar but produced a tensor with {} elements",
                        value.data.len()
                    ));
                }
                other => return Err(format!("scalar field is not a scalar value: {other:?}")),
            };
            env.insert(field_path.to_string(), v);
        }
        crate::opaque::FieldType::Record(inner_fields) => {
            // A nested record value: Adt{fields} in declared order.
            let ExecutionValue::Adt { fields, .. } = value else {
                return Err("nested record field is not a record value".to_string());
            };
            if fields.len() != inner_fields.len() {
                return Err("nested record arity mismatch".to_string());
            }
            for ((iname, ity), ival) in inner_fields.iter().zip(fields.iter()) {
                flatten_field_value(ival, ity, &format!("{field_path}.{iname}"), env)?;
            }
        }
    }
    Ok(())
}

/// The ONE produced-value validation chokepoint (U1 review-3
/// unification). Flatten a produced opaque record [`ExecutionValue`] into
/// the dotted-path env the lowered predicate reads (walking EVERY
/// representation leaf -- scalar fields, every tensor element, and nested
/// records), reject fail-CLOSED if ANY leaf is non-finite (NaN OR Inf),
/// then evaluate the strict invariant predicate. The scalar, tensor, and
/// nested-record produced-value paths all route here; the generator's
/// `validate_env` shares the SAME finiteness helper
/// ([`crate::opaque::any_non_finite`]) and the SAME strict evaluation, so
/// proposal and acceptance agree by construction.
fn validate_produced_env(
    value: &ExecutionValue,
    inv: &OpaqueInvariant,
    predicate: &crate::solver::SmtExpr,
) -> Result<bool, String> {
    let env = opaque_record_env(value, inv)?;
    // CR2-2 / U1 (fail-CLOSED on non-finite): a NaN/Inf representation leaf
    // is never a valid inhabitant of the opaque domain, REGARDLESS of the
    // predicate's shape. The strict evaluator gives `NaN != C == true`
    // under IEEE, so a `!=`/negation-shaped invariant would otherwise pass
    // fail-OPEN on a NaN leaf. Reject before the predicate runs (the
    // predicate's own truth value is the thing the NaN corrupts). This is
    // the single shared finiteness helper -- the generator's `validate_env`
    // calls the identical helper, so no second path can bypass it.
    if crate::opaque::any_non_finite(env.values().copied()) {
        return Ok(false);
    }
    let hash: std::collections::HashMap<String, f64> =
        env.iter().map(|(k, v)| (k.clone(), *v)).collect();
    // STRICT validation (CR-2 / CR-5 / CR-10): the produced value's
    // invariant is checked with exact `==`/`!=`, never the fuzz tolerance.
    Ok(crate::concrete_eval::eval_bool_strict(predicate, &hash))
}

fn node_def(name: &str, body: Expr) -> Expr {
    deep_node("def", vec![deep_sym(name), body])
}

/// Insert defs into the defining module with the invariant metadata
/// stripped (so the shape-sensitive predicate does not block IR lowering).
fn inject_into_module_stripped(
    module_source: &str,
    type_name: &str,
    defs: Vec<Expr>,
) -> Result<Vec<Expr>, String> {
    let exprs = chelis_deep::parser::parse_str(module_source)
        .map_err(|e| format!("reparse module: {e}"))?;
    let stripped: Vec<Expr> = exprs.iter().map(strip_invariant_meta).collect();
    Ok(inject_into_defining_module(&stripped, type_name, defs))
}

/// Drop the `invariant`/`invariant_amenability` deftype metadata keys,
/// recursively (keeps `opaque: true`).
fn strip_invariant_meta(expr: &Expr) -> Expr {
    match expr {
        Expr::List(list, span) => {
            let mut elements: Vec<Expr> = list.elements.iter().map(strip_invariant_meta).collect();
            if list.elements.first().and_then(sym_text) == Some("deftype")
                && let Some(Expr::Map(map, mspan)) = elements.get(1)
            {
                let kept: Vec<(String, Expr)> = map
                    .entries
                    .iter()
                    .filter(|(k, _)| k != "invariant" && k != "invariant_amenability")
                    .cloned()
                    .collect();
                elements[1] = Expr::Map(MetaMap { entries: kept }, *mspan);
            }
            Expr::List(List { elements }, *span)
        }
        other => other.clone(),
    }
}

/// Build a typed scalar literal Deep expr for a producer argument. Integer
/// widths are recognized through the single-source `is_int_width` (review
/// 5) and built width-appropriately: int32 is the literal default, the
/// other widths cast an int32 literal to the target width (matching the
/// generator's `int_lit`), so an int8/int16/int64 producer argument is a
/// well-typed integer, not a silently-mistyped float.
fn scalar_lit(prim: &str, v: f64) -> Expr {
    if crate::opaque::is_int_width(prim) {
        deep_int_lit_for(v as i64, prim)
    } else if prim == "bool" {
        deep_bool_lit(v != 0.0)
    } else {
        deep_float_lit(v)
    }
}

fn producer_param_names(exprs: &[Expr], producer: &str) -> Vec<String> {
    fn find(exprs: &[Expr], producer: &str) -> Option<Vec<String>> {
        for expr in exprs {
            if list_tag(expr) == Some("def")
                && let Some(name) = node_children(expr).first().and_then(sym_text)
                && name == producer
                && let Some(fn_node) = node_children(expr).get(1)
                && list_tag(fn_node) == Some("fn")
                && let Some(params) = node_children(fn_node).first()
            {
                let mut out = Vec::new();
                for p in node_children(params) {
                    if let Some(n) = sym_text(p) {
                        out.push(n.to_string());
                    } else if let Expr::List(l, _) = p
                        && let Some(Expr::Atom(Atom::Symbol(s), _)) = l.elements.first()
                    {
                        out.push(s.clone());
                    }
                }
                return Some(out);
            }
            if let Expr::List(l, _) = expr
                && let Some(found) = find(&l.elements[2.min(l.elements.len())..], producer)
            {
                return Some(found);
            }
        }
        None
    }
    find(exprs, producer).unwrap_or_default()
}

fn sample_scalar(kind: &str, rng: &mut Lcg) -> f64 {
    // Integer widths sample within the width's representable range via the
    // single-source `int_sample_bounds` (review 5): an int8 producer arg
    // samples in [-128, 127], never an unrepresentable value.
    if let Some((lo, hi)) = crate::opaque::int_sample_bounds(kind) {
        return rng.next_i64(lo, hi) as f64;
    }
    match kind {
        "bool" => {
            if rng.next_bool() {
                1.0
            } else {
                0.0
            }
        }
        _ => rng.next_f64(-10.0, 10.0),
    }
}

// --- Deep builders ---

fn deep_sym(s: &str) -> Expr {
    Expr::Atom(Atom::Symbol(s.to_string()), Span::new(0, 0))
}
fn deep_node(tag: &str, children: Vec<Expr>) -> Expr {
    let mut elements = vec![
        deep_sym(tag),
        Expr::Map(MetaMap::default(), Span::new(0, 0)),
    ];
    elements.extend(children);
    Expr::List(List { elements }, Span::new(0, 0))
}
fn deep_var(name: &str) -> Expr {
    deep_node("var", vec![deep_sym(name)])
}
fn deep_typed_lit(type_prim: &str, value: Expr) -> Expr {
    let mut entries = MetaMap::default();
    entries.entries.push((
        "type".to_string(),
        deep_node("t-prim", vec![deep_sym(type_prim)]),
    ));
    Expr::List(
        List {
            elements: vec![deep_sym("lit"), Expr::Map(entries, Span::new(0, 0)), value],
        },
        Span::new(0, 0),
    )
}
fn deep_float_lit(v: f64) -> Expr {
    deep_typed_lit("f32", Expr::Atom(Atom::Float(v), Span::new(0, 0)))
}
/// A width-appropriate integer literal Deep expr: an int32 literal for the
/// default width, otherwise an int32 literal cast to the target width
/// (review 5). `prim` must be an integer width (`is_int_width`).
fn deep_int_lit_for(v: i64, prim: &str) -> Expr {
    let lit = deep_typed_lit("int32", Expr::Atom(Atom::Int(v), Span::new(0, 0)));
    if prim == "int32" {
        lit
    } else {
        deep_node("cast", vec![lit, deep_node("t-prim", vec![deep_sym(prim)])])
    }
}
fn deep_bool_lit(v: bool) -> Expr {
    deep_typed_lit("bool", Expr::Atom(Atom::Bool(v), Span::new(0, 0)))
}
fn list_tag(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::List(l, _) => l.elements.first().and_then(sym_text),
        _ => None,
    }
}
fn node_children(expr: &Expr) -> &[Expr] {
    match expr {
        Expr::List(l, _) if l.elements.len() >= 2 => &l.elements[2..],
        _ => &[],
    }
}
fn sym_text(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    }
}

struct Lcg {
    state: u64,
}
impl Lcg {
    fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x9E37_79B9_7F4A_7C15,
        }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }
    fn next_bool(&mut self) -> bool {
        self.next_u64() & 1 == 1
    }
    fn next_i64(&mut self, min: i64, max: i64) -> i64 {
        let span = (max - min + 1) as u64;
        min + (self.next_u64() % span) as i64
    }
    fn next_f64(&mut self, min: f64, max: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
        min + (max - min) * unit
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod finding_tests {
    //! Unit tests for the PR #386 fresh-context review findings #7, #8, #11.
    //! These drive the private flatten/validate helpers directly with
    //! constructed values so the fail-closed boundaries are exercised at the
    //! unit level (no full CLI pipeline needed).
    // `super::*` already brings `ExecutionValue`, `OpaqueInvariant`, `Prim`,
    // `BTreeMap`, and the private flatten/validate helpers into scope.
    use super::*;
    use crate::opaque::FieldType;
    use chelis_compiler_api::schema::TensorValue;
    use chelis_pred::PredAmenability;

    /// A minimal single-scalar-field opaque invariant. `opaque_record_env`
    /// and `flatten_field_value` read only `ctor_name`, `binder`, and
    /// `fields`; the `predicate`/`amenability` are placeholders here.
    fn scalar_inv(type_name: &str, ctor_name: &str, field: &str) -> OpaqueInvariant {
        OpaqueInvariant {
            type_name: type_name.to_string(),
            ctor_name: ctor_name.to_string(),
            fields: vec![(field.to_string(), FieldType::Scalar("f32".to_string()))],
            // Never read by the flatten helpers under test; a placeholder
            // `fn` node keeps the struct well-formed.
            predicate: deep_node("fn", vec![]),
            binder: "p".to_string(),
            amenability: PredAmenability::Linear,
        }
    }

    // --- Finding #7: ctor mismatch at a Direct position must fail-closed ---

    #[test]
    fn f7_wrong_ctor_same_arity_adt_is_rejected() {
        // The invariant declares ctor `Probability`, but the produced value
        // is a DIFFERENT same-arity ADT (`Velocity`). Without the ctor check
        // this would flatten field-by-field and validate as if it were the
        // opaque record (a spurious Passed). It must FAIL-CLOSED with an
        // error naming the mismatch.
        let inv = scalar_inv("Probability", "Probability", "value");
        let wrong = ExecutionValue::Adt {
            ctor: "Velocity".to_string(),
            fields: vec![ExecutionValue::Float64 { value: 0.5 }],
        };
        let err = opaque_record_env(&wrong, &inv)
            .expect_err("a wrong-ctor same-arity ADT must be rejected, not flattened");
        assert!(
            err.contains("ctor mismatch")
                && err.contains("Velocity")
                && err.contains("Probability"),
            "the error must name the produced and declared ctor: {err}"
        );
    }

    #[test]
    fn f7_matching_ctor_still_flattens() {
        // The correct case (produced ctor == inv.ctor_name) must still work:
        // the field flattens into the binder-dotted env.
        let inv = scalar_inv("Probability", "Probability", "value");
        let right = ExecutionValue::Adt {
            ctor: "Probability".to_string(),
            fields: vec![ExecutionValue::Float64 { value: 0.5 }],
        };
        let env =
            opaque_record_env(&right, &inv).expect("the matching-ctor case must flatten cleanly");
        assert_eq!(
            env.get("p.value").copied(),
            Some(0.5),
            "the scalar field flattens to its binder-dotted path"
        );
    }

    // --- Finding #8: multi-element tensor for a Scalar field fails-closed ---

    #[test]
    fn f8_multi_element_tensor_for_scalar_field_fails_closed() {
        // The invariant declares a Scalar field, but the produced value is a
        // 3-element tensor (the 2nd element is NaN). The pre-fix code used
        // only `data[0]` (0.5) and silently dropped the NaN. It must now
        // FAIL-CLOSED naming the shape mismatch.
        let fty = FieldType::Scalar("f32".to_string());
        let mut env = BTreeMap::new();
        let multi = ExecutionValue::Tensor {
            value: TensorValue {
                shape: vec![3],
                data: vec![0.5, f64::NAN, 0.5],
            },
        };
        let err = flatten_field_value(&multi, &fty, "p.value", &mut env)
            .expect_err("a multi-element tensor for a Scalar field must fail-closed");
        assert!(
            err.contains("shape mismatch") && err.contains('3'),
            "the error must name the shape drift and the element count: {err}"
        );
        assert!(
            env.is_empty(),
            "no scalar value is recorded when the shape drifts"
        );
    }

    #[test]
    fn f8_single_element_tensor_for_scalar_field_still_works() {
        // A genuine 1-element tensor carrier for a Scalar field is still
        // legitimate and flattens to that element.
        let fty = FieldType::Scalar("f32".to_string());
        let mut env = BTreeMap::new();
        let single = ExecutionValue::Tensor {
            value: TensorValue {
                shape: vec![1],
                data: vec![0.5],
            },
        };
        flatten_field_value(&single, &fty, "p.value", &mut env)
            .expect("a 1-element tensor scalar carrier still works");
        assert_eq!(env.get("p.value").copied(), Some(0.5));
    }

    #[test]
    fn f8_true_scalar_for_scalar_field_still_works() {
        // A plain Float64 scalar (the common case) is unaffected by the fix.
        let fty = FieldType::Scalar("f32".to_string());
        let mut env = BTreeMap::new();
        flatten_field_value(
            &ExecutionValue::Float64 { value: 0.25 },
            &fty,
            "p.value",
            &mut env,
        )
        .expect("a true scalar still flattens");
        assert_eq!(env.get("p.value").copied(), Some(0.25));
    }

    // --- Finding #11: prim_name returns the canonical Prim::name() ---

    #[test]
    fn f11_prim_name_is_the_canonical_name() {
        // The canonical spelling, not the Debug-lowercased spelling, for a
        // representative width (and a couple of others for good measure).
        assert_eq!(prim_name(&Prim::Bf16), "bf16");
        assert_eq!(prim_name(&Prim::F32), "f32");
        assert_eq!(prim_name(&Prim::Int64), "int64");
        // Every variant's canonical name must match `prim_name` exactly. This
        // list must enumerate the WHOLE `Prim` vocabulary (including the f8
        // widths) or the "every variant" claim is hollow.
        for p in [
            Prim::F32,
            Prim::F64,
            Prim::F16,
            Prim::Bf16,
            Prim::F8e4m3,
            Prim::Int8,
            Prim::Int16,
            Prim::Int32,
            Prim::Int64,
            Prim::Bool,
            Prim::String,
        ] {
            assert_eq!(
                prim_name(&p),
                p.name(),
                "prim_name must route through the canonical Prim::name()"
            );
        }
    }
}
