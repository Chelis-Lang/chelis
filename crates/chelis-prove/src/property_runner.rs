//! Shared user-property runner (U4 review-3 unification).
//!
//! This is the ONE implementation of "discover the user `@property`
//! declarations of a module and verify each" that BOTH the CLI prove path
//! and the chelis-tide MCP `chelis_prove` tool call. It discovers
//! `@property` declarations the SAME way for `.ch` (parse + flatten
//! modules) and `.dp` (Deep metadata scan) -- it does NOT gate on a
//! hardcoded property name -- runs each through the same Tier B (SMT) ->
//! Tier C (fuzz) engine, with assumption injection for invariant-carrying
//! opaque binders, and returns one [`PropertyOutcome`] per property.
//!
//! Callers render the outcomes into their own surface (the CLI's NDJSON
//! property records, tide's MCP envelope) and fold them into a verdict --
//! the verification work itself is shared, so a prove run through tide is
//! identical to the CLI on the same module (the parity the cross-surface
//! test locks). This mirrors [`crate::obligation_engine`], which already
//! shares the derived-obligation run across the two surfaces.

use chelis_deep::DeepTag;
use std::{
    cell::RefCell,
    collections::{BTreeMap, BTreeSet},
};

use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr, List as DeepList, MetaMap};
use chelis_surf::ast::{
    BinOp, Decl, Expr, LetBinding, LetPattern, Literal, Param, PropertyOption, TypeExpr,
};

mod smt_lower;
use smt_lower::{
    ContractAbstraction, DeepInlineCtx, InlineCtx, deep_expr_to_smt, surf_arith, surf_expr_to_smt,
};

mod injection;
use crate::beacon_contract_prover::BeaconContractProver;
use crate::composition::{
    AssumptionDischarge, AssumptionRecord, CompositeVerdict, DischargeMethod, FUZZ_TOLERANCE,
    NonVacuityRecord, NonVacuityStatus, base_verdict_from_discharge, rollup_composite,
};
use crate::contracts::{
    NORMAL_CDF_RANGE, NORMAL_CDF_REFLECTION, standard_contract_registry,
    standard_contract_registry_with_prover,
};
use crate::discharge::QualifierSet;

/// The verification status of one user property.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PropertyStatus {
    Passed,
    Failed,
    Unsupported,
    Error,
}

/// The tier that produced the property's verdict.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PropertyTier {
    Smt,
    /// Mathematical induction with separately dispatched base and step SMT goals.
    Induction,
    Fuzz,
    /// No tier ran (a sampling/declaration error).
    None,
}

impl PropertyTier {
    pub fn as_str(self) -> &'static str {
        match self {
            PropertyTier::Smt => "smt",
            PropertyTier::Induction => "induction",
            PropertyTier::Fuzz => "fuzz",
            PropertyTier::None => "none",
        }
    }
}

/// One solver-dispatched induction obligation. These records are additive
/// machine evidence; neither an assertion nor a caller-supplied classification
/// can manufacture a green induction result (chelis#978).
#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct InductionCaseEvidence {
    pub status: String,
    pub arith_model: String,
    pub goal: crate::tier_b::SmtProperty,
    pub soundness: crate::discharge::Soundness,
    pub qualifiers: QualifierSet,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub non_vacuity: Option<NonVacuityRecord>,
}

#[derive(Debug, Clone, PartialEq, serde::Serialize)]
pub struct InductionEvidence {
    pub variable: String,
    pub base: InductionCaseEvidence,
    pub step: InductionCaseEvidence,
}

/// The verification outcome of one user property.
#[derive(Debug, Clone, PartialEq)]
pub struct PropertyOutcome {
    pub name: String,
    pub status: PropertyStatus,
    pub proof_tier: PropertyTier,
    pub samples: usize,
    pub seed: u64,
    pub counterexample: Option<serde_json::Value>,
    pub shrink_steps: usize,
    pub reason: Option<String>,
    /// `true` when the property was verified through the assumption-injection
    /// path (an invariant-carrying opaque binder). Only affects rendering.
    pub injected: bool,
    /// Assumptions used by this proof, with their discharge evidence.
    pub assumptions: Vec<AssumptionRecord>,
    /// The deductive base discharge's `(soundness, qualifiers)`, when an
    /// engine discharged the base (chelis#422). Retained so `append_assumptions`
    /// re-derives the same base badge it was first built with. `None` for a
    /// fuzz-only or non-green base (the base is derived from `proof_tier`).
    base_discharge: Option<(crate::discharge::Soundness, QualifierSet)>,
    /// Weakest-link verdict after composing the proof and its assumptions.
    pub composite_verdict: CompositeVerdict,
    /// The discharged proposition (the property body) in canonical Surf/Deep
    /// text, so a consumer displays exactly what was discharged rather than
    /// re-parsing it out of source and risking drift (chelis#436). `None` only
    /// when the outcome carries no body to render (a declaration/discovery
    /// error that never reached a property body); a real verification outcome
    /// always carries its goal.
    pub goal: Option<String>,
    /// Additive machine evidence for a Tier-C run. `None` for deductive
    /// outcomes and declaration failures that never selected a sampler.
    pub sampling_method: Option<String>,
    pub attempted_samples: usize,
    /// Samples that satisfied every guard and reached the property body. This
    /// is distinct from `samples`, which is zero on terminal error/unsupported
    /// outcomes even when useful sampling work preceded the terminal state.
    pub accepted_samples: usize,
    pub rejected_samples: usize,
    /// Present only after the induction classifier constructed and dispatched
    /// real base and step obligations.
    pub induction_evidence: Option<InductionEvidence>,
}

impl PropertyOutcome {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        name: impl Into<String>,
        status: PropertyStatus,
        proof_tier: PropertyTier,
        samples: usize,
        seed: u64,
        counterexample: Option<serde_json::Value>,
        reason: Option<String>,
        injected: bool,
        assumptions: Vec<AssumptionRecord>,
    ) -> Self {
        Self::with_base_discharge(
            name,
            status,
            proof_tier,
            samples,
            seed,
            counterexample,
            reason,
            injected,
            assumptions,
            None,
        )
    }

    /// Like [`PropertyOutcome::new`], but carrying the deductive base
    /// discharge's `(soundness, qualifiers)` so the SMT/Beacon green base is
    /// projected from the engine's own guarantee rather than a hardcoded
    /// `proven` (chelis#422). The SMT-tier Proved sites pass `Some(..)`; every
    /// other site keeps `new` (fuzz / terminal / unsupported bases).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn with_base_discharge(
        name: impl Into<String>,
        status: PropertyStatus,
        proof_tier: PropertyTier,
        samples: usize,
        seed: u64,
        counterexample: Option<serde_json::Value>,
        reason: Option<String>,
        injected: bool,
        assumptions: Vec<AssumptionRecord>,
        base_discharge: Option<(crate::discharge::Soundness, QualifierSet)>,
    ) -> Self {
        let base = base_verdict(
            &status,
            proof_tier,
            samples,
            counterexample.as_ref(),
            base_discharge.as_ref(),
        );
        let composite_verdict = rollup_composite(base, &assumptions);
        Self {
            name: name.into(),
            status,
            proof_tier,
            samples,
            seed,
            counterexample,
            shrink_steps: 0,
            reason,
            injected,
            assumptions,
            base_discharge,
            composite_verdict,
            goal: None,
            sampling_method: None,
            attempted_samples: 0,
            accepted_samples: 0,
            rejected_samples: 0,
            induction_evidence: None,
        }
    }

    pub(super) fn with_shrink_steps(mut self, shrink_steps: usize) -> Self {
        self.shrink_steps = shrink_steps;
        self
    }

    /// Attach the discharged proposition's canonical text (chelis#436). Called
    /// once per outcome with the property body rendered through the canonical
    /// Surf/Deep formatter, so the goal travels with the record.
    pub(super) fn with_goal(mut self, goal: impl Into<String>) -> Self {
        self.goal = Some(goal.into());
        self
    }

    fn with_sampling(
        mut self,
        method: impl Into<String>,
        attempted_samples: usize,
        accepted_samples: usize,
    ) -> Self {
        self.sampling_method = Some(method.into());
        self.attempted_samples = attempted_samples;
        self.accepted_samples = accepted_samples;
        self.rejected_samples = attempted_samples.saturating_sub(accepted_samples);
        self
    }

    /// The base discharge `(soundness, qualifiers)` this outcome's verdict was
    /// composed from, synthesized to match [`base_verdict`]: a deductive base
    /// uses the threaded discharge; a fuzz-tier pass synthesizes the `FuzzBase`
    /// guarantee; everything else has no green base.
    fn effective_base_discharge(&self) -> Option<(crate::discharge::Soundness, QualifierSet)> {
        if self.status != PropertyStatus::Passed {
            return None;
        }
        match self.proof_tier {
            PropertyTier::Smt | PropertyTier::Induction => self.base_discharge.clone(),
            PropertyTier::Fuzz if self.samples > 0 => Some((
                crate::discharge::Soundness::Empirical,
                QualifierSet::from_iter_kinds([crate::discharge::Qualifier::FuzzBase]),
            )),
            _ => None,
        }
    }

    /// The full disclosed qualifier set of this outcome's composed verdict, as
    /// sorted snake_case strings for the `qualifiers:[...]` JSON array
    /// (chelis#422, D2). A green outcome discloses its composed union; a
    /// reals-hedged disproof discloses `real_arithmetic` (symmetric to the proof
    /// side, so the machine-arithmetic caveat is visible on the failure too);
    /// every other non-green outcome discloses none.
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

    fn append_assumptions(&mut self, assumptions: Vec<AssumptionRecord>) {
        if assumptions.is_empty() {
            return;
        }
        self.assumptions.extend(assumptions);
        let base = base_verdict(
            &self.status,
            self.proof_tier,
            self.samples,
            self.counterexample.as_ref(),
            self.base_discharge.as_ref(),
        );
        self.composite_verdict = rollup_composite(base, &self.assumptions);
    }

    /// Whether this outcome is a genuine pass. A pass is `Passed` with at
    /// least one sample (or an SMT proof, which carries `samples == 0` but
    /// `proof_tier == Smt`). A `Passed` with zero fuzz samples is NOT a
    /// genuine pass -- it is the vacuous/timeout sentinel and the fold must
    /// treat it as not-ok (U4).
    pub fn is_pass(&self) -> bool {
        if matches!(
            self.composite_verdict,
            CompositeVerdict::Failed
                | CompositeVerdict::DisprovedModuloRealArithmetic
                | CompositeVerdict::Invalid
                | CompositeVerdict::Unsupported
        ) {
            return false;
        }
        self.status == PropertyStatus::Passed
            && (matches!(self.proof_tier, PropertyTier::Smt | PropertyTier::Induction)
                || self.samples > 0)
    }

    /// The display status label, bucketed through [`is_pass`] so a
    /// zero-sample `Passed` sentinel renders as `"unsupported"` (not a
    /// genuine pass) consistently across surfaces (F8 review-4). Both the
    /// CLI's NDJSON render and the tide MCP JSON render use this single
    /// method, so a property's reported status can never diverge between the
    /// two surfaces.
    pub fn display_status(&self) -> &'static str {
        if self.status == PropertyStatus::Error {
            return "error";
        }
        match self.composite_verdict {
            // A hedged disproof is a failure (the property did not hold over the
            // reals); the coarse label is `failed`. The precise
            // `composite_verdict` field still carries the hedged badge.
            CompositeVerdict::Failed | CompositeVerdict::DisprovedModuloRealArithmetic => {
                return "failed";
            }
            CompositeVerdict::Invalid | CompositeVerdict::Unsupported => return "unsupported",
            // Every green badge -- proven, the proven_modulo_* disclosures, the
            // sound-over-approximation base, and the fuzz-only base -- falls
            // through to the `is_pass` check below and reports `"passed"`
            // (chelis#422). The badge, not the status, carries the
            // not-proven / disclosed-caveat distinction.
            CompositeVerdict::Proven
            | CompositeVerdict::ProvenModuloRealArithmetic
            | CompositeVerdict::ProvenModuloCertifiedEnvelope
            | CompositeVerdict::ProvenModuloFuzzValidatedContract
            | CompositeVerdict::ProvenModuloAssertedAxiom
            | CompositeVerdict::SoundApproximate
            | CompositeVerdict::FuzzValidatedEmpirical => {}
        }
        if self.is_pass() {
            "passed"
        } else {
            match self.status {
                PropertyStatus::Failed => "failed",
                PropertyStatus::Unsupported => "unsupported",
                PropertyStatus::Error => "error",
                // A `Passed` that is NOT a genuine pass (zero-sample fuzz
                // sentinel) is reported as unsupported -- it lowers the
                // overall verdict and is visible, matching the fold.
                PropertyStatus::Passed => "unsupported",
            }
        }
    }
}

fn base_verdict(
    status: &PropertyStatus,
    proof_tier: PropertyTier,
    samples: usize,
    counterexample: Option<&serde_json::Value>,
    base_discharge: Option<&(crate::discharge::Soundness, QualifierSet)>,
) -> CompositeVerdict {
    match status {
        PropertyStatus::Passed => match proof_tier {
            // chelis#422: a deductive-tier green base is projected from the
            // DISCHARGE's own (soundness, qualifiers) -- threaded here from the
            // dispatch instead of being flattened to a hardcoded `proven` --
            // so cvc5-over-reals reads `proven_modulo_real_arithmetic` and a
            // Beacon interval discharge reads `sound_approximate`. A green base
            // MUST carry its discharge; a missing one is a covered-or-rejected
            // `Unsupported`, never a silent proof.
            PropertyTier::Smt | PropertyTier::Induction => match base_discharge {
                Some((soundness, qualifiers)) => {
                    base_verdict_from_discharge(*soundness, qualifiers)
                }
                None => CompositeVerdict::Unsupported,
            },
            // A fuzz-tier base pass is empirically validated, NOT proven: seed
            // `FuzzBase` so it renders `fuzz_validated` and can never read
            // `proven_*` (chelis#422).
            PropertyTier::Fuzz if samples > 0 => base_verdict_from_discharge(
                crate::discharge::Soundness::Empirical,
                &QualifierSet::from_iter_kinds([crate::discharge::Qualifier::FuzzBase]),
            ),
            _ => CompositeVerdict::Unsupported,
        },
        PropertyStatus::Failed => {
            if counterexample.is_some() {
                // chelis#422 (symmetric to the Proved side): a disproof whose
                // base discharge is over the reals (carries `RealArith`) is a
                // HEDGED failure -- the counterexample may be a false
                // counterexample at machine arithmetic -- so it reads
                // `disproved_modulo_real_arithmetic`. A disproof with no
                // real-arithmetic base (a fuzz counterexample is a real
                // machine-arithmetic witness, threaded with no base_discharge) is
                // a definite `Failed`.
                if base_discharge_is_real_arithmetic(base_discharge) {
                    CompositeVerdict::DisprovedModuloRealArithmetic
                } else {
                    CompositeVerdict::Failed
                }
            } else {
                CompositeVerdict::Unsupported
            }
        }
        PropertyStatus::Unsupported | PropertyStatus::Error => CompositeVerdict::Unsupported,
    }
}

/// Whether a threaded base discharge is an over-the-reals decision (carries
/// [`crate::discharge::Qualifier::RealArith`]). Used to hedge a disproof's
/// failure badge symmetrically to the proof side.
fn base_discharge_is_real_arithmetic(
    base_discharge: Option<&(crate::discharge::Soundness, QualifierSet)>,
) -> bool {
    base_discharge
        .is_some_and(|(_, qualifiers)| qualifiers.contains(crate::discharge::Qualifier::RealArith))
}

/// Options for a property run, mirroring the prove surface.
#[derive(Debug, Clone)]
pub struct PropertyRunOptions {
    pub seed: u64,
    pub samples: usize,
    pub smt_timeout_ms: u64,
    /// `"auto"` (Tier B then C), `"smt-only"`, `"induction-only"`, `"fuzz-only"`.
    pub tier: String,
    /// Property-name selector (`--only`); `None` runs all.
    pub only: Option<String>,
    /// Acceptance-rate floor for invariant binder generation (D-STARVE).
    pub invariant_min_rate: f64,
    /// Max sampling attempts; `None` derives it from `samples`.
    pub max_attempts: Option<usize>,
}

impl Default for PropertyRunOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            samples: 100,
            smt_timeout_ms: 5000,
            tier: "auto".to_string(),
            only: None,
            invariant_min_rate: 0.01,
            max_attempts: None,
        }
    }
}

impl PropertyRunOptions {
    fn effective_seed(&self, property_seed: Option<u64>) -> u64 {
        if self.seed != 0 {
            self.seed
        } else {
            property_seed.unwrap_or(0)
        }
    }

    /// The seed the injection path uses. The CLI injection path keyed off
    /// the explicit `--seed` (defaulting to 0), independent of any
    /// per-property seed option; preserve that for parity.
    fn injection_seed(&self) -> u64 {
        self.seed
    }
}

/// The result of a property run from module source.
#[derive(Debug, Clone)]
pub enum PropertyRunResult {
    /// Discovered and ran the user properties (possibly none).
    Ran(Vec<PropertyOutcome>),
}

/// Discover and run every user `@property` declaration in Surf `.ch`
/// source. Returns `Err` if the source does not parse.
pub fn run_surf_source_properties(
    source: &str,
    options: &PropertyRunOptions,
) -> Result<PropertyRunResult, String> {
    let parsed = chelis_surf::parser::parse_str(source).map_err(|e| format!("parse: {e}"))?;
    let flat = flatten_module_decls(&parsed);
    run_surf_decls_properties(&flat, &flat, &parsed, options)
}

/// Run properties discovered in `entry_decls` while lowering/evaluating
/// against `all_decls`. The CLI prove path uses this for Reef-linked inputs:
/// imports are available to Tier B/C, but only the user's entry properties are
/// reported.
pub fn run_surf_decls_properties(
    all_decls: &[Decl],
    entry_decls: &[Decl],
    module_decls: &[Decl],
    options: &PropertyRunOptions,
) -> Result<PropertyRunResult, String> {
    run_surf_decls_properties_with_contract_decls(
        all_decls,
        entry_decls,
        module_decls,
        &[],
        options,
    )
}

/// Run linked Surf properties with an explicit trusted implementation slice
/// for standard contracts. The CLI Reef path passes dependency-owned linker
/// declarations here (including chelis-std and Nautilus); unlinked source
/// paths pass an empty slice, so a user cannot obtain contract assumptions by
/// spelling a linker-shaped name.
pub fn run_surf_decls_properties_with_contract_decls(
    all_decls: &[Decl],
    entry_decls: &[Decl],
    module_decls: &[Decl],
    trusted_contract_decls: &[Decl],
    options: &PropertyRunOptions,
) -> Result<PropertyRunResult, String> {
    let properties = collect_surf_properties(entry_decls, options.only.as_deref());
    let mut out = Vec::new();
    for property in &properties {
        // chelis#436: the discharged proposition travels with the record,
        // rendered through the canonical Surf formatter so a consumer displays
        // exactly what was discharged rather than re-parsing it from source.
        // A guarded property's proposition is its full `where`-guarded form,
        // not its bare body (MED-1): the prover discharges `pre => body`.
        let goal = chelis_surf::format::format_proposition(
            &property.params,
            &property.preconditions,
            &property.body,
        );
        out.push(
            prove_surf_property(
                all_decls,
                module_decls,
                trusted_contract_decls,
                property,
                options,
            )
            .with_goal(goal),
        );
    }
    Ok(PropertyRunResult::Ran(out))
}

/// Discover and run every user `@property` declaration in Deep `.dp`
/// source. Returns `Err` if the source does not parse.
pub fn run_deep_source_properties(
    source: &str,
    options: &PropertyRunOptions,
) -> Result<PropertyRunResult, String> {
    let exprs =
        chelis_deep::parser::parse_and_stamp_file(source).map_err(|e| format!("parse: {e}"))?;
    let properties = discover_deep_properties(&exprs, options.only.as_deref())?;
    let mut out = Vec::new();
    for property in &properties {
        // chelis#436: the discharged proposition travels with the record,
        // rendered through the canonical Deep printer (flat) with
        // lowering/producer metadata stripped so a consumer sees the bare
        // proposition. A guarded property's proposition is the implication
        // `pre => body`, not the bare body (MED-1): the prover discharges
        // `(/\ preconditions) => body`.
        let goal = chelis_deep::printer::print_expr_flat(&chelis_deep::ast::strip_metadata(
            &deep_proposition(&property.preconditions, &property.body),
        ));
        out.push(prove_deep_property(&exprs, property, options).with_goal(goal));
    }
    Ok(PropertyRunResult::Ran(out))
}

// ===========================================================================
// Surf property model + discovery
// ===========================================================================

#[derive(Debug, Clone)]
struct Property {
    name: String,
    params: Vec<Param>,
    preconditions: Vec<Expr>,
    body: Expr,
    samples: Option<usize>,
    seed: Option<u64>,
    contracts: Vec<String>,
}

fn flatten_module_decls(decls: &[Decl]) -> Vec<Decl> {
    let mut out = Vec::new();
    for decl in decls {
        match decl {
            Decl::Module { decls, .. } => out.extend(flatten_module_decls(decls)),
            other => out.push(other.clone()),
        }
    }
    out
}

/// Discover every `@property` declaration in flattened Surf decls (no
/// hardcoded name; the `only` filter narrows by name pattern). This is the
/// SAME discovery the CLI uses (it was relocated here so both surfaces share
/// it, U4).
fn collect_surf_properties(decls: &[Decl], only: Option<&str>) -> Vec<Property> {
    decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::Property {
                name,
                params,
                preconditions,
                body,
                options,
                ..
            } if matches_filter(name, only) => Some(Property {
                name: name.clone(),
                params: params.clone(),
                preconditions: preconditions.clone(),
                body: body.clone(),
                samples: property_samples(options),
                seed: property_seed(options),
                contracts: property_contracts(options),
            }),
            _ => None,
        })
        .collect()
}

fn property_samples(options: &[PropertyOption]) -> Option<usize> {
    options.iter().find_map(|option| match option {
        PropertyOption::Samples(Expr::Lit(Literal::Int(value), _), _) => {
            (*value >= 0).then_some(*value as usize)
        }
        _ => None,
    })
}

fn property_seed(options: &[PropertyOption]) -> Option<u64> {
    options.iter().find_map(|option| match option {
        PropertyOption::Seed(Expr::Lit(Literal::Int(value), _), _) => {
            (*value >= 0).then_some(*value as u64)
        }
        _ => None,
    })
}

fn property_contracts(options: &[PropertyOption]) -> Vec<String> {
    options
        .iter()
        .filter_map(|option| match option {
            PropertyOption::Contract(id, _) => Some(id.clone()),
            _ => None,
        })
        .collect()
}

// ===========================================================================
// Surf property running (Tier B -> Tier C, with injection)
// ===========================================================================

fn prove_surf_property(
    decls: &[Decl],
    module_decls: &[Decl],
    trusted_contract_decls: &[Decl],
    property: &Property,
    options: &PropertyRunOptions,
) -> PropertyOutcome {
    let seed = options.effective_seed(property.seed);
    let contract_assumptions = match contract_assumptions(property) {
        Ok(records) => records,
        Err(reason) => {
            return PropertyOutcome::new(
                property.name.clone(),
                PropertyStatus::Unsupported,
                PropertyTier::None,
                0,
                seed,
                None,
                Some(reason),
                false,
                Vec::new(),
            );
        }
    };

    // chelis#978: induction is a fail-closed lane. An explicit request always
    // enters it. Default auto enters it before ordinary Tier B when the checked
    // AST says the property reaches a recursive model. Both accepted plans and
    // unsupported recursive structures are terminal: neither can trail into a
    // finite fuzz sample (or recursively overflow the concrete evaluator).
    let auto_recursive =
        options.tier == "auto" && property_reaches_recursive_model(decls, property);
    if options.tier == "induction-only" || auto_recursive {
        let deep = chelis_surf::desugar::desugar_program(decls);
        if let Err(infer) = chelis_types::check_typed_program(&deep) {
            return PropertyOutcome::new(
                property.name.clone(),
                PropertyStatus::Error,
                PropertyTier::Induction,
                0,
                seed,
                None,
                Some(format!(
                    "induction requires a type-checked compiler AST: {}",
                    infer
                        .errors
                        .iter()
                        .map(|error| error.message.as_str())
                        .collect::<Vec<_>>()
                        .join("; ")
                )),
                false,
                Vec::new(),
            );
        }
        let mut outcome = try_surf_induction(decls, property, options, seed);
        outcome.append_assumptions(contract_assumptions);
        return outcome;
    }

    // Assumption injection (RFC D-INJECT): a property with an
    // invariant-carrying opaque binder is verified ONLY over
    // invariant-satisfying binder values; the injection path owns it.
    if injection::property_has_opaque_invariant_binder(module_decls, &property.params) {
        let mut outcome = injection::prove_with_injection(
            module_decls,
            &property.name,
            &property.params,
            &property.preconditions,
            &property.body,
            options,
        );
        outcome.append_assumptions(contract_assumptions);
        return outcome;
    }

    // Tier B: attempt SMT proof when --tier auto / smt-only.
    if options.tier == "auto" || options.tier == "smt-only" {
        if let Some(mut outcome) =
            try_surf_tier_b(decls, trusted_contract_decls, property, options, seed)
        {
            outcome.append_assumptions(contract_assumptions);
            return outcome;
        }
        if options.tier == "smt-only" {
            // smt-only: a property that does not lower to Tier B is
            // unsupported (no fuzz fallback). This matches the obligation
            // engine's smt-only handling and the CLI's exit semantics.
            let mut outcome = PropertyOutcome::new(
                property.name.clone(),
                PropertyStatus::Unsupported,
                PropertyTier::Smt,
                0,
                seed,
                None,
                Some("property does not lower to Tier B (smt-only)".to_string()),
                false,
                Vec::new(),
            );
            outcome.append_assumptions(contract_assumptions);
            return outcome;
        }
    }

    // Tier C: fuzz.
    let mut outcome = prove_surf_property_fuzz(decls, property, options, seed);
    outcome.append_assumptions(contract_assumptions);
    outcome
}

/// Conservatively decide whether a property's compiler AST reaches a recursive
/// function. Auto uses this only as a lane-selection guard: a positive result
/// still has to pass the exact induction classifier, while a negative result
/// preserves the established Tier-B-then-C ordering for ordinary properties.
fn property_reaches_recursive_model(decls: &[Decl], property: &Property) -> bool {
    let function_names: std::collections::BTreeSet<String> = decls
        .iter()
        .filter_map(|decl| match decl {
            Decl::FunDef { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    let mut graph: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
    for decl in decls {
        let Decl::FunDef {
            name, params, body, ..
        } = decl
        else {
            continue;
        };
        let param_names = params.iter().map(|param| param.name.as_str()).collect();
        let mut refs = std::collections::BTreeSet::new();
        collect_expr_refs(body, &param_names, &function_names, &mut refs);
        graph.insert(name.clone(), refs);
    }

    let property_params = property
        .params
        .iter()
        .map(|param| param.name.as_str())
        .collect();
    let mut roots = std::collections::BTreeSet::new();
    collect_expr_refs(
        &property.body,
        &property_params,
        &function_names,
        &mut roots,
    );
    for precondition in &property.preconditions {
        collect_expr_refs(precondition, &property_params, &function_names, &mut roots);
    }

    roots.iter().any(|root| {
        let mut reachable = BTreeSet::new();
        collect_reachable_functions(root, &graph, &mut reachable);
        reachable.iter().any(|candidate| {
            function_reaches_itself(candidate, candidate, &graph, &mut BTreeSet::new())
        })
    })
}

fn collect_reachable_functions(
    current: &str,
    graph: &BTreeMap<String, std::collections::BTreeSet<String>>,
    reachable: &mut BTreeSet<String>,
) {
    if !reachable.insert(current.to_string()) {
        return;
    }
    if let Some(successors) = graph.get(current) {
        for successor in successors {
            collect_reachable_functions(successor, graph, reachable);
        }
    }
}

fn function_reaches_itself(
    origin: &str,
    current: &str,
    graph: &BTreeMap<String, std::collections::BTreeSet<String>>,
    visited: &mut BTreeSet<String>,
) -> bool {
    if !visited.insert(current.to_string()) {
        return false;
    }
    graph.get(current).is_some_and(|successors| {
        successors.iter().any(|successor| {
            successor == origin || function_reaches_itself(origin, successor, graph, visited)
        })
    })
}

fn contract_assumptions(property: &Property) -> Result<Vec<AssumptionRecord>, String> {
    let contracts = expanded_contracts(property);
    if contracts.is_empty() {
        return Ok(Vec::new());
    }
    let registry = match BeaconContractProver::from_env() {
        Some(prover) => standard_contract_registry_with_prover(&prover),
        None => standard_contract_registry(),
    };
    let probe = registry.probe_consumer(
        &property.name,
        CompositeVerdict::Proven,
        contracts.iter().map(String::as_str),
    );
    if let Some(missing) = probe
        .assumptions
        .iter()
        .find(|record| record.discharge.is_none())
    {
        return Err(format!("unknown contract `{}`", missing.name));
    }
    Ok(probe.assumptions)
}

fn expanded_contracts(property: &Property) -> Vec<String> {
    let mut contracts = property.contracts.clone();
    if contracts.iter().any(|id| id == NORMAL_CDF_REFLECTION)
        && !contracts.iter().any(|id| id == NORMAL_CDF_RANGE)
    {
        contracts.push(NORMAL_CDF_RANGE.to_string());
    }
    contracts.sort();
    contracts.dedup();
    contracts
}

/// chelis#434 envelope lane: try to discharge a transcendental-bearing SMT goal
/// through the certified special-function envelopes.
///
/// Pipeline: `NormalCdfToErf` (rewrite `normal_cdf(x)` to `½(1+erf(x/√2))`) then
/// `AbstractSubterm` (replace each `erf`/`exp`/`log`/`sqrt` with a fresh variable
/// bounded by its certified envelope over the argument's sound interval). If the
/// residual has NO transcendentals left (fully abstracted) and is inlineable, it
/// is solved over reals.
///
/// SOUNDNESS: the abstract-subterm transform is a sound OVER-approximation (the
/// fresh variable ranges over a certified superset of the true transcendental
/// value), so a residual **proof** entails the original goal. But an abstract
/// **counterexample** may be SPURIOUS (a point in the over-approximation that no
/// real input reaches — the coupled-subterm gap of chelis#637). Therefore this
/// lane returns a green outcome ONLY on `Proved`; on `Disproved`, `Timeout`, or
/// `Unknown` it returns `None` and the caller's existing honest paths run (the
/// goal is never marked failed/disproved off an abstract counterexample). A
/// `Proved` residual carries the `SpecialFunctionCertified` qualifier, so it
/// projects to `proven_modulo_certified_envelope` (never plain `proven` /
/// `proven_modulo_real_arithmetic`).
fn try_envelope_lane(
    property_name: &str,
    smt_prop: &crate::tier_b::SmtProperty,
    options: &PropertyRunOptions,
    seed: u64,
) -> Option<PropertyOutcome> {
    use crate::transformation::Transformation;
    use crate::transformations::abstract_subterm::AbstractSubterm;
    use crate::transformations::normal_cdf_erf::NormalCdfToErf;

    // Pre-pass (normal_cdf -> erf), then abstract-subterm. Each is a pure
    // Goal -> Vec<Goal>; both yield exactly one goal here (no goal-splitting).
    let goal = crate::discharge::Goal::smt(smt_prop.clone());
    let lowered = NormalCdfToErf.apply(&goal);
    let [lowered] = lowered.as_slice() else {
        return None;
    };
    let abstracted = AbstractSubterm::new().apply(lowered);
    let [residual] = abstracted.as_slice() else {
        return None;
    };
    let crate::discharge::GoalShape::Smt(ref res_prop) = residual.shape else {
        return None;
    };
    // The transform must have removed EVERY transcendental (residual inlineable)
    // AND actually changed the goal (something was abstracted); else there is
    // nothing the envelope lane can add over the base path — decline.
    if res_prop.postcondition == smt_prop.postcondition
        || !matches!(
            crate::classify_inlineability(&res_prop.postcondition),
            crate::Inlineability::Inlineable
        )
    {
        return None;
    }

    // Solve the residual over reals. Only a genuine PROOF is accepted.
    let discharge = crate::engine_registry::DischargeRegistry::with_builtin_engines()
        .dispatch(residual, options.smt_timeout_ms);
    let soundness = discharge.soundness();
    let mut qualifiers = discharge.qualifier_set().clone();
    if !matches!(discharge.into_result(), crate::tier_b::TierBResult::Proved) {
        // Disproved (possibly spurious under the over-approximation), Timeout, or
        // Unknown: decline. The caller falls through to the honest base paths.
        return None;
    }

    // Proved: the certified envelope backed the discharge. Add the
    // SpecialFunctionCertified qualifier so the badge is the distinct honest
    // `proven_modulo_certified_envelope` (chelis#434 milestone 3).
    qualifiers.insert(crate::discharge::Qualifier::SpecialFunctionCertified);
    let base_discharge = Some((soundness, qualifiers));

    // Non-vacuity of the ORIGINAL property (its preconditions must be
    // satisfiable), mirroring the base SMT path.
    let non_vacuity = smt_non_vacuity_record(smt_prop, options.smt_timeout_ms);
    let reason = match non_vacuity.status {
        NonVacuityStatus::Established => None,
        NonVacuityStatus::Invalid | NonVacuityStatus::Unsupported => non_vacuity.reason.clone(),
    };
    let assumptions = property_assumption_records(
        property_name,
        smt_prop,
        AssumptionDischarge::new(
            DischargeMethod::Smt,
            serde_json::json!({
                "status": "proved",
                "property": property_name,
                "arith_model": "real",
                "lane": "certified_envelope",
            }),
        ),
        non_vacuity,
    );
    let status = if reason.is_some() {
        PropertyStatus::Unsupported
    } else {
        PropertyStatus::Passed
    };
    Some(PropertyOutcome::with_base_discharge(
        property_name.to_string(),
        status,
        PropertyTier::Smt,
        0,
        seed,
        None,
        reason,
        false,
        assumptions,
        base_discharge,
    ))
}

fn induction_unsupported(
    property: &Property,
    seed: u64,
    reason: impl Into<String>,
) -> PropertyOutcome {
    PropertyOutcome::new(
        property.name.clone(),
        PropertyStatus::Unsupported,
        PropertyTier::Induction,
        0,
        seed,
        None,
        Some(format!(
            "induction unsupported: {} (chelis#978)",
            reason.into()
        )),
        false,
        Vec::new(),
    )
}

/// A compiler-AST-derived induction plan. The accepted v1 shape is purposely
/// small: one `int*` induction binder, an explicit `n >= 0` domain, one direct
/// scalar model call in the proposition, and one exact `f(n - 1, unchanged...)`
/// recursive call behind `if n <= 0`. Everything else is covered-or-rejected.
struct SurfInductionPlan {
    variable: String,
    variables: Vec<(String, crate::solver::SmtSort)>,
    preconditions: Vec<crate::solver::SmtExpr>,
    proposition: crate::solver::SmtExpr,
    outer_call: crate::solver::SmtExpr,
    model_params: Vec<String>,
    model_return_sort: crate::solver::SmtSort,
    condition: crate::solver::SmtExpr,
    base_expr: crate::solver::SmtExpr,
    step_expr: crate::solver::SmtExpr,
    recursive_call: crate::solver::SmtExpr,
}

fn try_surf_induction(
    decls: &[Decl],
    property: &Property,
    options: &PropertyRunOptions,
    seed: u64,
) -> PropertyOutcome {
    if !property.contracts.is_empty() {
        return induction_unsupported(
            property,
            seed,
            "contract abstractions are not in the induction lane",
        );
    }
    let plan = match classify_surf_induction(decls, property) {
        Ok(plan) => plan,
        Err(reason) => return induction_unsupported(property, seed, reason),
    };
    let k_name = "__chelis_induction_k";
    let value_name = "__chelis_induction_value";
    let base_recursive_name = "__chelis_induction_base_recursive";
    if plan
        .variables
        .iter()
        .any(|(name, _)| name == k_name || name == value_name || name == base_recursive_name)
    {
        return induction_unsupported(property, seed, "reserved induction symbol collision");
    }

    use crate::solver::{ArithOp, SmtExpr};
    let zero = SmtExpr::IntLit(0);
    let k = SmtExpr::Var(k_name.to_string());
    let successor = SmtExpr::Arith(
        ArithOp::Add,
        Box::new(k.clone()),
        Box::new(SmtExpr::IntLit(1)),
    );
    let induction_value = SmtExpr::Var(value_name.to_string());
    let base_recursive_value = SmtExpr::Var(base_recursive_name.to_string());

    let base_var_subst = BTreeMap::from([(plan.variable.clone(), zero.clone())]);
    let base_outer = substitute_smt_vars(&plan.outer_call, &base_var_subst);
    let mut base_model_subst = BTreeMap::new();
    let outer_args = smt_apply_args(&plan.outer_call).expect("classified call");
    for (param, arg) in plan.model_params.iter().zip(outer_args) {
        base_model_subst.insert(param.clone(), substitute_smt_vars(arg, &base_var_subst));
    }
    let base_recursive_call = substitute_smt_vars(&plan.recursive_call, &base_model_subst);
    let base_step = replace_smt_exact(
        &substitute_smt_vars(&plan.step_expr, &base_model_subst),
        &base_recursive_call,
        &base_recursive_value,
    );
    let base_value = SmtExpr::Ite(
        Box::new(substitute_smt_vars(&plan.condition, &base_model_subst)),
        Box::new(substitute_smt_vars(&plan.base_expr, &base_model_subst)),
        Box::new(base_step),
    );
    let base_post = replace_smt_exact(
        &substitute_smt_vars(&plan.proposition, &base_var_subst),
        &base_outer,
        &base_value,
    );
    let base_pre: Vec<_> = plan
        .preconditions
        .iter()
        .map(|pre| substitute_smt_vars(pre, &base_var_subst))
        .collect();
    let base_goal = crate::tier_b::SmtProperty {
        variables: {
            let mut variables: Vec<_> = plan
                .variables
                .iter()
                .filter(|(name, _)| name != &plan.variable)
                .cloned()
                .collect();
            variables.push((base_recursive_name.to_string(), plan.model_return_sort));
            variables
        },
        preconditions: base_pre,
        postcondition: base_post,
    };

    let step_var_subst = BTreeMap::from([(plan.variable.clone(), k.clone())]);
    let succ_var_subst = BTreeMap::from([(plan.variable.clone(), successor.clone())]);
    let ih_outer = substitute_smt_vars(&plan.outer_call, &step_var_subst);
    let ih = replace_smt_exact(
        &substitute_smt_vars(&plan.proposition, &step_var_subst),
        &ih_outer,
        &induction_value,
    );
    let successor_outer = substitute_smt_vars(&plan.outer_call, &succ_var_subst);
    let mut step_model_subst = BTreeMap::new();
    for (param, arg) in plan.model_params.iter().zip(outer_args) {
        step_model_subst.insert(param.clone(), substitute_smt_vars(arg, &succ_var_subst));
    }
    let recursive_at_k = substitute_smt_vars(&plan.recursive_call, &step_model_subst);
    // The recursive call after one-step unfolding is syntactically
    // `f((k + 1) - 1, ...)`; normalize that integer successor/predecessor pair
    // and require it to be EXACTLY the IH call `f(k, ...)`. This comparison is
    // the ownership check that prevents applying the hypothesis to a nearby or
    // reconstructed subproblem.
    if normalize_successor_predecessor(&recursive_at_k) != ih_outer {
        return induction_unsupported(
            property,
            seed,
            "the unfolded recursive call is not the exact induction-hypothesis subproblem",
        );
    }
    let step_branch = replace_smt_exact(
        &substitute_smt_vars(&plan.step_expr, &step_model_subst),
        &recursive_at_k,
        &induction_value,
    );
    let step_value = SmtExpr::Ite(
        Box::new(substitute_smt_vars(&plan.condition, &step_model_subst)),
        Box::new(substitute_smt_vars(&plan.base_expr, &step_model_subst)),
        Box::new(step_branch),
    );
    let conclusion = replace_smt_exact(
        &substitute_smt_vars(&plan.proposition, &succ_var_subst),
        &successor_outer,
        &step_value,
    );
    let mut step_pre: Vec<_> = plan
        .preconditions
        .iter()
        .map(|pre| substitute_smt_vars(pre, &step_var_subst))
        .collect();
    step_pre.push(ih);
    let mut step_variables: Vec<_> = plan
        .variables
        .iter()
        .map(|(name, sort)| {
            if name == &plan.variable {
                (k_name.to_string(), *sort)
            } else {
                (name.clone(), *sort)
            }
        })
        .collect();
    step_variables.push((value_name.to_string(), plan.model_return_sort));
    let step_goal = crate::tier_b::SmtProperty {
        variables: step_variables,
        preconditions: step_pre,
        postcondition: conclusion,
    };

    if smt_apply_count(&base_goal.postcondition) != 0
        || base_goal
            .preconditions
            .iter()
            .any(|pre| smt_apply_count(pre) != 0)
        || smt_apply_count(&step_goal.postcondition) != 0
        || step_goal
            .preconditions
            .iter()
            .any(|pre| smt_apply_count(pre) != 0)
    {
        return induction_unsupported(
            property,
            seed,
            "an uninterpreted call remained after exact one-step unfolding",
        );
    }

    dispatch_induction_goals(
        property,
        &plan.variable,
        base_goal,
        step_goal,
        options,
        seed,
    )
}

fn dispatch_induction_goals(
    property: &Property,
    variable: &str,
    base_goal: crate::tier_b::SmtProperty,
    step_goal: crate::tier_b::SmtProperty,
    options: &PropertyRunOptions,
    seed: u64,
) -> PropertyOutcome {
    use crate::tier_b::TierBResult;
    let registry = crate::engine_registry::DischargeRegistry::with_builtin_engines();
    let base = registry.dispatch(
        &crate::discharge::Goal::smt(base_goal.clone()),
        options.smt_timeout_ms,
    );
    let base_soundness = base.soundness();
    let base_qualifiers = base.qualifier_set().clone();
    let base_discharge = Some((base_soundness, base_qualifiers.clone()));
    let base_result = base.into_result();
    let base_status = induction_result_status(&base_result);
    if !matches!(base_result, TierBResult::Proved) {
        let counterexample = match &base_result {
            TierBResult::Disproved(model) => Some(model.clone()),
            _ => None,
        };
        let reason = induction_terminal_reason("base", &base_result);
        let mut outcome = PropertyOutcome::with_base_discharge(
            property.name.clone(),
            if counterexample.is_some() {
                PropertyStatus::Failed
            } else {
                PropertyStatus::Unsupported
            },
            PropertyTier::Induction,
            0,
            seed,
            counterexample,
            reason,
            false,
            Vec::new(),
            base_discharge,
        );
        outcome.induction_evidence = Some(InductionEvidence {
            variable: variable.to_string(),
            base: InductionCaseEvidence {
                status: base_status,
                arith_model: "real".to_string(),
                goal: base_goal,
                soundness: base_soundness,
                qualifiers: base_qualifiers,
                non_vacuity: None,
            },
            step: InductionCaseEvidence {
                status: "not_run".to_string(),
                arith_model: "real".to_string(),
                goal: step_goal,
                soundness: crate::discharge::Soundness::Untrusted,
                qualifiers: QualifierSet::new(),
                non_vacuity: None,
            },
        });
        return outcome;
    }
    let base_nv = smt_non_vacuity_record(&base_goal, options.smt_timeout_ms);
    if base_nv.status != NonVacuityStatus::Established {
        let reason = base_nv
            .reason
            .clone()
            .unwrap_or_else(|| "base obligation is vacuous".to_string());
        let mut outcome = induction_unsupported(property, seed, reason);
        outcome.induction_evidence = Some(InductionEvidence {
            variable: variable.to_string(),
            base: InductionCaseEvidence {
                status: "proved".to_string(),
                arith_model: "real".to_string(),
                goal: base_goal,
                soundness: base_soundness,
                qualifiers: base_qualifiers,
                non_vacuity: Some(base_nv),
            },
            step: InductionCaseEvidence {
                status: "not_run".to_string(),
                arith_model: "real".to_string(),
                goal: step_goal,
                soundness: crate::discharge::Soundness::Untrusted,
                qualifiers: QualifierSet::new(),
                non_vacuity: None,
            },
        });
        return outcome;
    }

    let step = registry.dispatch(
        &crate::discharge::Goal::smt(step_goal.clone()),
        options.smt_timeout_ms,
    );
    let step_soundness = step.soundness();
    let step_qualifiers = step.qualifier_set().clone();
    let step_discharge = Some((step_soundness, step_qualifiers.clone()));
    let step_result = step.into_result();
    let step_status = induction_result_status(&step_result);
    if !matches!(step_result, TierBResult::Proved) {
        let counterexample = match &step_result {
            TierBResult::Disproved(model) => Some(model.clone()),
            _ => None,
        };
        let reason = induction_terminal_reason("step", &step_result);
        let mut outcome = PropertyOutcome::with_base_discharge(
            property.name.clone(),
            if counterexample.is_some() {
                PropertyStatus::Failed
            } else {
                PropertyStatus::Unsupported
            },
            PropertyTier::Induction,
            0,
            seed,
            counterexample,
            reason,
            false,
            Vec::new(),
            step_discharge,
        );
        outcome.induction_evidence = Some(InductionEvidence {
            variable: variable.to_string(),
            base: InductionCaseEvidence {
                status: "proved".to_string(),
                arith_model: "real".to_string(),
                goal: base_goal,
                soundness: base_soundness,
                qualifiers: base_qualifiers,
                non_vacuity: Some(base_nv),
            },
            step: InductionCaseEvidence {
                status: step_status,
                arith_model: "real".to_string(),
                goal: step_goal,
                soundness: step_soundness,
                qualifiers: step_qualifiers,
                non_vacuity: None,
            },
        });
        return outcome;
    }
    let step_nv = smt_non_vacuity_record(&step_goal, options.smt_timeout_ms);
    if step_nv.status != NonVacuityStatus::Established {
        let reason = step_nv
            .reason
            .clone()
            .unwrap_or_else(|| "step obligation is vacuous".to_string());
        let mut outcome = induction_unsupported(property, seed, reason);
        outcome.induction_evidence = Some(InductionEvidence {
            variable: variable.to_string(),
            base: InductionCaseEvidence {
                status: "proved".to_string(),
                arith_model: "real".to_string(),
                goal: base_goal,
                soundness: base_soundness,
                qualifiers: base_qualifiers,
                non_vacuity: Some(base_nv),
            },
            step: InductionCaseEvidence {
                status: "proved".to_string(),
                arith_model: "real".to_string(),
                goal: step_goal,
                soundness: step_soundness,
                qualifiers: step_qualifiers,
                non_vacuity: Some(step_nv),
            },
        });
        return outcome;
    }
    let mut outcome = PropertyOutcome::with_base_discharge(
        property.name.clone(),
        PropertyStatus::Passed,
        PropertyTier::Induction,
        0,
        seed,
        None,
        None,
        false,
        Vec::new(),
        step_discharge,
    );
    outcome.induction_evidence = Some(InductionEvidence {
        variable: variable.to_string(),
        base: InductionCaseEvidence {
            status: "proved".to_string(),
            arith_model: "real".to_string(),
            goal: base_goal,
            soundness: base_soundness,
            qualifiers: base_qualifiers,
            non_vacuity: Some(base_nv),
        },
        step: InductionCaseEvidence {
            status: "proved".to_string(),
            arith_model: "real".to_string(),
            goal: step_goal,
            soundness: step_soundness,
            qualifiers: step_qualifiers,
            non_vacuity: Some(step_nv),
        },
    });
    outcome
}

fn induction_result_status(result: &crate::tier_b::TierBResult) -> String {
    use crate::tier_b::TierBResult;
    match result {
        TierBResult::Proved => "proved",
        TierBResult::Disproved(_) => "disproved",
        TierBResult::Timeout => "timeout",
        TierBResult::Unknown => "unknown",
        TierBResult::Error(_) => "error",
    }
    .to_string()
}

fn induction_terminal_reason(case: &str, result: &crate::tier_b::TierBResult) -> Option<String> {
    use crate::tier_b::TierBResult;
    match result {
        TierBResult::Disproved(_) => None,
        TierBResult::Timeout => Some(format!("induction {case} obligation timed out")),
        TierBResult::Unknown => Some(format!("induction {case} obligation was unknown")),
        TierBResult::Error(reason) => Some(format!("induction {case} obligation error: {reason}")),
        TierBResult::Proved => None,
    }
}

fn classify_surf_induction(
    decls: &[Decl],
    property: &Property,
) -> Result<SurfInductionPlan, String> {
    use crate::solver::{ArithOp, CmpOp, SmtExpr, SmtSort};
    let variables: Vec<(String, SmtSort)> = property
        .params
        .iter()
        .map(|param| {
            let TypeExpr::Named(name, _) = param
                .ty
                .as_ref()
                .ok_or_else(|| "every binder needs an explicit scalar type".to_string())?
            else {
                return Err("only scalar binders are supported".to_string());
            };
            if !(matches!(name.as_str(), "f32" | "f64" | "bool")
                || crate::opaque::is_int_width(name))
            {
                return Err(format!(
                    "binder `{}` has unsupported type `{name}`",
                    param.name
                ));
            }
            Ok((param.name.clone(), crate::opaque::prim_to_smt_sort(name)))
        })
        .collect::<Result<_, _>>()?;

    for decl in decls {
        let Decl::FunDef {
            name,
            params,
            ret_ty,
            body,
            ..
        } = decl
        else {
            continue;
        };
        if params.is_empty() {
            continue;
        }
        let Some(TypeExpr::Named(first_ty, _)) = params[0].ty.as_ref() else {
            continue;
        };
        if !crate::opaque::is_int_width(first_ty) {
            continue;
        }
        let Some(TypeExpr::Named(ret_name, _)) = ret_ty.as_ref() else {
            continue;
        };
        let model_return_sort = match ret_name.as_str() {
            "f32" | "f64" => SmtSort::Real,
            integer if crate::opaque::is_int_width(integer) => {
                crate::opaque::prim_to_smt_sort(integer)
            }
            _ => continue,
        };
        let without_model: Vec<Decl> = decls
            .iter()
            .filter(|candidate| !matches!(candidate, Decl::FunDef { name: candidate_name, .. } if candidate_name == name))
            .cloned()
            .collect();
        let ctx = InlineCtx {
            decls: &without_model,
            depth: 0,
            max_depth: 3,
            call_stack: vec![],
            contracts: None,
            grad_diagnostic: None,
        };
        let Some(proposition) = surf_expr_to_smt(&property.body, &ctx) else {
            continue;
        };
        let mut outer_calls = Vec::new();
        collect_named_smt_applies(&proposition, name, &mut outer_calls);
        if outer_calls.len() != 1 || smt_apply_count(&proposition) != 1 {
            continue;
        }
        let outer_call = outer_calls.pop().expect("one call");
        let Some(outer_args) = smt_apply_args(&outer_call) else {
            continue;
        };
        if outer_args.len() != params.len() {
            continue;
        }
        let SmtExpr::Var(induction_var) = &outer_args[0] else {
            continue;
        };
        let Some((_, variable_sort)) = variables
            .iter()
            .find(|(candidate, _)| candidate == induction_var)
        else {
            continue;
        };
        if !matches!(variable_sort, SmtSort::Int) {
            continue;
        }
        if outer_args.iter().any(|arg| !matches!(arg, SmtExpr::Var(_))) {
            continue;
        }

        let preconditions: Vec<SmtExpr> = property
            .preconditions
            .iter()
            .map(|expr| {
                surf_expr_to_smt(expr, &ctx)
                    .ok_or_else(|| "a domain precondition does not lower to SMT".to_string())
            })
            .collect::<Result<_, _>>()?;
        let domain_guards = preconditions
            .iter()
            .filter(|pre| smt_contains_var(pre, induction_var))
            .count();
        if domain_guards != 1
            || !preconditions
                .iter()
                .any(|pre| is_zero_lower_bound(pre, induction_var))
        {
            continue;
        }

        let Some(model_body) = surf_arith(body, &ctx) else {
            continue;
        };
        let SmtExpr::Ite(condition, base_expr, step_expr) = model_body else {
            continue;
        };
        let model_n = &params[0].name;
        if !matches!(condition.as_ref(),
            SmtExpr::Cmp(CmpOp::Le, left, right)
                if matches!(left.as_ref(), SmtExpr::Var(var) if var == model_n)
                    && matches!(right.as_ref(), SmtExpr::IntLit(0)))
        {
            continue;
        }
        if smt_apply_count(&base_expr) != 0 || smt_apply_count(&step_expr) != 1 {
            continue;
        }
        let mut recursive_calls = Vec::new();
        collect_named_smt_applies(&step_expr, name, &mut recursive_calls);
        if recursive_calls.len() != 1 {
            continue;
        }
        let recursive_call = recursive_calls.pop().expect("one recursive call");
        let Some(recursive_args) = smt_apply_args(&recursive_call) else {
            continue;
        };
        if recursive_args.len() != params.len() {
            continue;
        }
        let exact_decrement = matches!(&recursive_args[0],
            SmtExpr::Arith(ArithOp::Sub, left, right)
                if matches!(left.as_ref(), SmtExpr::Var(var) if var == model_n)
                    && matches!(right.as_ref(), SmtExpr::IntLit(1)));
        let unchanged_tail = params
            .iter()
            .skip(1)
            .zip(recursive_args.iter().skip(1))
            .all(|(param, arg)| matches!(arg, SmtExpr::Var(var) if var == &param.name));
        if !exact_decrement || !unchanged_tail {
            continue;
        }

        return Ok(SurfInductionPlan {
            variable: induction_var.clone(),
            variables,
            preconditions,
            proposition,
            outer_call,
            model_params: params.iter().map(|param| param.name.clone()).collect(),
            model_return_sort,
            condition: *condition,
            base_expr: *base_expr,
            step_expr: *step_expr,
            recursive_call,
        });
    }
    Err("no compiler-AST model matched the exact f(0)/f(n-1) induction shape".to_string())
}

fn smt_apply_args(expr: &crate::solver::SmtExpr) -> Option<&[crate::solver::SmtExpr]> {
    match expr {
        crate::solver::SmtExpr::Apply(_, args) => Some(args),
        _ => None,
    }
}

fn smt_apply_count(expr: &crate::solver::SmtExpr) -> usize {
    use crate::solver::SmtExpr;
    match expr {
        SmtExpr::Apply(_, args) => 1 + args.iter().map(smt_apply_count).sum::<usize>(),
        SmtExpr::Arith(_, left, right) | SmtExpr::Cmp(_, left, right) => {
            smt_apply_count(left) + smt_apply_count(right)
        }
        SmtExpr::Bool(_, children) => children.iter().map(smt_apply_count).sum(),
        SmtExpr::Not(inner) | SmtExpr::Forall(_, inner) | SmtExpr::Exists(_, inner) => {
            smt_apply_count(inner)
        }
        SmtExpr::Ite(cond, yes, no) => {
            smt_apply_count(cond) + smt_apply_count(yes) + smt_apply_count(no)
        }
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => 0,
    }
}

fn collect_named_smt_applies(
    expr: &crate::solver::SmtExpr,
    name: &str,
    out: &mut Vec<crate::solver::SmtExpr>,
) {
    use crate::solver::SmtExpr;
    match expr {
        SmtExpr::Apply(candidate, args) => {
            if candidate == name {
                out.push(expr.clone());
            }
            for arg in args {
                collect_named_smt_applies(arg, name, out);
            }
        }
        SmtExpr::Arith(_, left, right) | SmtExpr::Cmp(_, left, right) => {
            collect_named_smt_applies(left, name, out);
            collect_named_smt_applies(right, name, out);
        }
        SmtExpr::Bool(_, children) => {
            for child in children {
                collect_named_smt_applies(child, name, out);
            }
        }
        SmtExpr::Not(inner) | SmtExpr::Forall(_, inner) | SmtExpr::Exists(_, inner) => {
            collect_named_smt_applies(inner, name, out)
        }
        SmtExpr::Ite(cond, yes, no) => {
            collect_named_smt_applies(cond, name, out);
            collect_named_smt_applies(yes, name, out);
            collect_named_smt_applies(no, name, out);
        }
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {}
    }
}

fn smt_contains_var(expr: &crate::solver::SmtExpr, name: &str) -> bool {
    use crate::solver::SmtExpr;
    match expr {
        SmtExpr::Var(candidate) => candidate == name,
        SmtExpr::Arith(_, left, right) | SmtExpr::Cmp(_, left, right) => {
            smt_contains_var(left, name) || smt_contains_var(right, name)
        }
        SmtExpr::Bool(_, children) | SmtExpr::Apply(_, children) => {
            children.iter().any(|child| smt_contains_var(child, name))
        }
        SmtExpr::Not(inner) => smt_contains_var(inner, name),
        SmtExpr::Forall(vars, inner) | SmtExpr::Exists(vars, inner) => {
            !vars.iter().any(|(var, _)| var == name) && smt_contains_var(inner, name)
        }
        SmtExpr::Ite(cond, yes, no) => {
            smt_contains_var(cond, name)
                || smt_contains_var(yes, name)
                || smt_contains_var(no, name)
        }
        SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => false,
    }
}

fn is_zero_lower_bound(expr: &crate::solver::SmtExpr, name: &str) -> bool {
    use crate::solver::{CmpOp, SmtExpr};
    matches!(expr,
        SmtExpr::Cmp(CmpOp::Ge, left, right)
            if matches!(left.as_ref(), SmtExpr::Var(var) if var == name)
                && matches!(right.as_ref(), SmtExpr::IntLit(0)))
        || matches!(expr,
            SmtExpr::Cmp(CmpOp::Le, left, right)
                if matches!(left.as_ref(), SmtExpr::IntLit(0))
                    && matches!(right.as_ref(), SmtExpr::Var(var) if var == name))
}

fn substitute_smt_vars(
    expr: &crate::solver::SmtExpr,
    substitutions: &BTreeMap<String, crate::solver::SmtExpr>,
) -> crate::solver::SmtExpr {
    use crate::solver::SmtExpr;
    match expr {
        SmtExpr::Var(name) => substitutions
            .get(name)
            .cloned()
            .unwrap_or_else(|| expr.clone()),
        SmtExpr::Arith(op, left, right) => SmtExpr::Arith(
            *op,
            Box::new(substitute_smt_vars(left, substitutions)),
            Box::new(substitute_smt_vars(right, substitutions)),
        ),
        SmtExpr::Cmp(op, left, right) => SmtExpr::Cmp(
            *op,
            Box::new(substitute_smt_vars(left, substitutions)),
            Box::new(substitute_smt_vars(right, substitutions)),
        ),
        SmtExpr::Bool(op, children) => SmtExpr::Bool(
            *op,
            children
                .iter()
                .map(|child| substitute_smt_vars(child, substitutions))
                .collect(),
        ),
        SmtExpr::Not(inner) => SmtExpr::Not(Box::new(substitute_smt_vars(inner, substitutions))),
        SmtExpr::Apply(name, args) => SmtExpr::Apply(
            name.clone(),
            args.iter()
                .map(|arg| substitute_smt_vars(arg, substitutions))
                .collect(),
        ),
        SmtExpr::Ite(cond, yes, no) => SmtExpr::Ite(
            Box::new(substitute_smt_vars(cond, substitutions)),
            Box::new(substitute_smt_vars(yes, substitutions)),
            Box::new(substitute_smt_vars(no, substitutions)),
        ),
        SmtExpr::Forall(vars, inner) | SmtExpr::Exists(vars, inner) => {
            let filtered: BTreeMap<_, _> = substitutions
                .iter()
                .filter(|(name, _)| !vars.iter().any(|(bound, _)| bound == *name))
                .map(|(name, value)| (name.clone(), value.clone()))
                .collect();
            let body = Box::new(substitute_smt_vars(inner, &filtered));
            if matches!(expr, SmtExpr::Forall(_, _)) {
                SmtExpr::Forall(vars.clone(), body)
            } else {
                SmtExpr::Exists(vars.clone(), body)
            }
        }
        SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => expr.clone(),
    }
}

fn replace_smt_exact(
    expr: &crate::solver::SmtExpr,
    target: &crate::solver::SmtExpr,
    replacement: &crate::solver::SmtExpr,
) -> crate::solver::SmtExpr {
    if expr == target {
        return replacement.clone();
    }
    use crate::solver::SmtExpr;
    match expr {
        SmtExpr::Arith(op, left, right) => SmtExpr::Arith(
            *op,
            Box::new(replace_smt_exact(left, target, replacement)),
            Box::new(replace_smt_exact(right, target, replacement)),
        ),
        SmtExpr::Cmp(op, left, right) => SmtExpr::Cmp(
            *op,
            Box::new(replace_smt_exact(left, target, replacement)),
            Box::new(replace_smt_exact(right, target, replacement)),
        ),
        SmtExpr::Bool(op, children) => SmtExpr::Bool(
            *op,
            children
                .iter()
                .map(|child| replace_smt_exact(child, target, replacement))
                .collect(),
        ),
        SmtExpr::Not(inner) => {
            SmtExpr::Not(Box::new(replace_smt_exact(inner, target, replacement)))
        }
        SmtExpr::Apply(name, args) => SmtExpr::Apply(
            name.clone(),
            args.iter()
                .map(|arg| replace_smt_exact(arg, target, replacement))
                .collect(),
        ),
        SmtExpr::Ite(cond, yes, no) => SmtExpr::Ite(
            Box::new(replace_smt_exact(cond, target, replacement)),
            Box::new(replace_smt_exact(yes, target, replacement)),
            Box::new(replace_smt_exact(no, target, replacement)),
        ),
        SmtExpr::Forall(vars, inner) => SmtExpr::Forall(
            vars.clone(),
            Box::new(replace_smt_exact(inner, target, replacement)),
        ),
        SmtExpr::Exists(vars, inner) => SmtExpr::Exists(
            vars.clone(),
            Box::new(replace_smt_exact(inner, target, replacement)),
        ),
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {
            expr.clone()
        }
    }
}

fn normalize_successor_predecessor(expr: &crate::solver::SmtExpr) -> crate::solver::SmtExpr {
    use crate::solver::{ArithOp, SmtExpr};
    let normalized = match expr {
        SmtExpr::Arith(op, left, right) => SmtExpr::Arith(
            *op,
            Box::new(normalize_successor_predecessor(left)),
            Box::new(normalize_successor_predecessor(right)),
        ),
        SmtExpr::Cmp(op, left, right) => SmtExpr::Cmp(
            *op,
            Box::new(normalize_successor_predecessor(left)),
            Box::new(normalize_successor_predecessor(right)),
        ),
        SmtExpr::Bool(op, children) => SmtExpr::Bool(
            *op,
            children
                .iter()
                .map(normalize_successor_predecessor)
                .collect(),
        ),
        SmtExpr::Not(inner) => SmtExpr::Not(Box::new(normalize_successor_predecessor(inner))),
        SmtExpr::Apply(name, args) => SmtExpr::Apply(
            name.clone(),
            args.iter().map(normalize_successor_predecessor).collect(),
        ),
        SmtExpr::Ite(cond, yes, no) => SmtExpr::Ite(
            Box::new(normalize_successor_predecessor(cond)),
            Box::new(normalize_successor_predecessor(yes)),
            Box::new(normalize_successor_predecessor(no)),
        ),
        SmtExpr::Forall(vars, inner) => SmtExpr::Forall(
            vars.clone(),
            Box::new(normalize_successor_predecessor(inner)),
        ),
        SmtExpr::Exists(vars, inner) => SmtExpr::Exists(
            vars.clone(),
            Box::new(normalize_successor_predecessor(inner)),
        ),
        SmtExpr::Var(_) | SmtExpr::RealLit(_) | SmtExpr::IntLit(_) | SmtExpr::BoolLit(_) => {
            expr.clone()
        }
    };
    match &normalized {
        SmtExpr::Arith(ArithOp::Sub, left, right)
            if matches!(right.as_ref(), SmtExpr::IntLit(1))
                && matches!(left.as_ref(),
                    SmtExpr::Arith(ArithOp::Add, _, addend)
                        if matches!(addend.as_ref(), SmtExpr::IntLit(1))) =>
        {
            let SmtExpr::Arith(_, base, _) = left.as_ref() else {
                unreachable!()
            };
            base.as_ref().clone()
        }
        _ => normalized,
    }
}

/// Try Tier B (SMT) for a surf property. Returns `Some(outcome)` for a
/// determinate SMT verdict (Proved => Passed, Disproved => Failed), or
/// `None` to fall through to Tier C (the property did not lower, or the
/// solver timed out / errored).
fn try_surf_tier_b(
    decls: &[Decl],
    trusted_contract_decls: &[Decl],
    property: &Property,
    options: &PropertyRunOptions,
    seed: u64,
) -> Option<PropertyOutcome> {
    let contracts = expanded_contracts(property);
    let contract_abstraction = RefCell::new(ContractAbstraction::for_contracts(
        &contracts,
        trusted_contract_decls,
    ));
    let grad_diagnostic = RefCell::new(None);
    let postcondition = surf_expr_to_smt(
        &property.body,
        &InlineCtx {
            decls,
            depth: 0,
            max_depth: 3,
            call_stack: vec![],
            contracts: Some(&contract_abstraction),
            grad_diagnostic: Some(&grad_diagnostic),
        },
    );
    let postcondition = match postcondition {
        Some(postcondition) => postcondition,
        None if options.tier == "smt-only" => {
            let reason = grad_diagnostic.into_inner()?;
            return Some(PropertyOutcome::new(
                property.name.clone(),
                PropertyStatus::Unsupported,
                PropertyTier::Smt,
                0,
                seed,
                None,
                Some(reason),
                false,
                Vec::new(),
            ));
        }
        None => return None,
    };
    let variables: Vec<(String, crate::solver::SmtSort)> = property
        .params
        .iter()
        .filter_map(|p| {
            let sort = match p.ty.as_ref()? {
                // Single-source int-width -> sort decision (F3): an @property
                // scalar param routes through the same prim_to_smt_sort the
                // field and producer-param paths use. A non-scalar param type
                // (tensor/ADT) is not SMT-amenable, so the property falls back
                // to Tier C.
                TypeExpr::Named(name, _)
                    if matches!(name.as_str(), "f32" | "f64" | "bool")
                        || crate::opaque::is_int_width(name) =>
                {
                    crate::opaque::prim_to_smt_sort(name)
                }
                _ => return None,
            };
            Some((p.name.clone(), sort))
        })
        .collect();
    if variables.len() != property.params.len() {
        return None;
    }
    let preconditions: Vec<crate::solver::SmtExpr> = property
        .preconditions
        .iter()
        .filter_map(|e| {
            surf_expr_to_smt(
                e,
                &InlineCtx {
                    decls,
                    depth: 0,
                    max_depth: 3,
                    call_stack: vec![],
                    contracts: Some(&contract_abstraction),
                    grad_diagnostic: None,
                },
            )
        })
        .collect();
    if preconditions.len() != property.preconditions.len() {
        return None;
    }
    let abstraction = contract_abstraction.into_inner();
    if abstraction.requires_normal_cdf() && !abstraction.used_normal_cdf() {
        return Some(PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Unsupported,
            PropertyTier::Smt,
            0,
            seed,
            None,
            Some(
                "contract abstraction did not bind any call to Std.Contracts.normal_cdf"
                    .to_string(),
            ),
            false,
            Vec::new(),
        ));
    }
    if abstraction.requires_quantile() && !abstraction.used_quantile() {
        return Some(PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Unsupported,
            PropertyTier::Smt,
            0,
            seed,
            None,
            Some(
                "contract abstraction did not bind any trusted linked call to Nautilus.Stats.quantile_vec"
                    .to_string(),
            ),
            false,
            Vec::new(),
        ));
    }
    if abstraction.has_unsupported_quantile_contract() {
        return Some(PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Unsupported,
            PropertyTier::Smt,
            0,
            seed,
            None,
            Some(
                "Nautilus.Stats.quantile_vec contract abstraction currently supports std.quantile.monotonicity only"
                    .to_string(),
            ),
            false,
            Vec::new(),
        ));
    }
    if abstraction.requires_quantile() && !abstraction.has_quantile_monotonicity_pair() {
        return Some(PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Unsupported,
            PropertyTier::Smt,
            0,
            seed,
            None,
            Some(
                "std.quantile.monotonicity requires two Nautilus.Stats.quantile_vec calls over the same compiler-bound dataset"
                    .to_string(),
            ),
            false,
            Vec::new(),
        ));
    }
    if abstraction.requires_reflection_pair() && !abstraction.has_reflection_pair() {
        return Some(PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Unsupported,
            PropertyTier::Smt,
            0,
            seed,
            None,
            Some("normal CDF reflection contract requires a syntactic normal_cdf(x) / normal_cdf(-x) call pair".to_string()),
            false,
            Vec::new(),
        ));
    }
    let mut variables = variables;
    variables.extend(abstraction.variables());
    let mut preconditions = preconditions;
    preconditions.extend(abstraction.preconditions());
    let smt_prop = crate::tier_b::SmtProperty {
        variables,
        preconditions,
        postcondition,
    };
    if !matches!(
        crate::classify_inlineability(&smt_prop.postcondition),
        crate::Inlineability::Inlineable
    ) {
        // chelis#434 ENVELOPE LANE: the goal carries transcendentals the base
        // lowering cannot inline. Try discharging through the certified
        // special-function envelopes (normal_cdf -> erf, then abstract-subterm)
        // BEFORE falling through to fuzz. Returns Some ONLY on a genuine proof of
        // the residual (-> proven_modulo_certified_envelope); on any non-proof it
        // declines and the existing honest paths (smt-only transcendental
        // `unsupported`, or auto fuzz) run unchanged.
        if let Some(outcome) = try_envelope_lane(&property.name, &smt_prop, options, seed) {
            return Some(outcome);
        }
        return None;
    }
    // Route the solve through the WI-9 discharge-engine registry. The registry
    // selects the SMT engine for this SMT goal by fitness. chelis#422: keep the
    // FULL `Discharge` -- its `(soundness, qualifiers)` -- instead of
    // `.into_result()`-discarding it. The per-lane engine choice is internal to
    // `with_builtin_engines`: under `--features smt` it is the cvc5 engine; in
    // the default build it is the solver-free solve_property engine; an
    // out-of-tree engine (Beacon) registered on top owns its goal shape. Both
    // in-tree lanes classify a proof as over the reals (`SoundApproximate` +
    // `RealArith`); threading that here is what makes the green base read
    // `proven_modulo_real_arithmetic` rather than a flattened `proven`, and a
    // Beacon `SoundApproximate` + `SoundOverApproximation` discharge read
    // `sound_approximate`.
    let discharge = crate::engine_registry::DischargeRegistry::with_builtin_engines().dispatch(
        &crate::discharge::Goal::smt(smt_prop.clone()),
        options.smt_timeout_ms,
    );
    let base_discharge = Some((discharge.soundness(), discharge.qualifier_set().clone()));
    let discharge_result = discharge.into_result();
    match discharge_result {
        crate::tier_b::TierBResult::Proved => {
            let non_vacuity = smt_non_vacuity_record(&smt_prop, options.smt_timeout_ms);
            let reason = match non_vacuity.status {
                NonVacuityStatus::Established => None,
                NonVacuityStatus::Invalid | NonVacuityStatus::Unsupported => {
                    non_vacuity.reason.clone()
                }
            };
            let assumptions = property_assumption_records(
                &property.name,
                &smt_prop,
                AssumptionDischarge::new(
                    DischargeMethod::Smt,
                    serde_json::json!({
                        "status": "proved",
                        "property": property.name,
                        "arith_model": "real",
                    }),
                ),
                non_vacuity,
            );
            let status = if reason.is_some() {
                PropertyStatus::Unsupported
            } else {
                PropertyStatus::Passed
            };
            Some(PropertyOutcome::with_base_discharge(
                property.name.clone(),
                status,
                PropertyTier::Smt,
                0,
                seed,
                None,
                reason,
                false,
                assumptions,
                base_discharge,
            ))
        }
        crate::tier_b::TierBResult::Disproved(model) => {
            // chelis#422 (symmetric): thread the discharge's `(soundness,
            // qualifiers)` here too. A disproof over the reals carries
            // `RealArith`, so the failure base reads
            // `disproved_modulo_real_arithmetic` -- the counterexample may be a
            // false counterexample at machine arithmetic -- rather than a
            // flattened definite `failed`.
            Some(PropertyOutcome::with_base_discharge(
                property.name.clone(),
                PropertyStatus::Failed,
                PropertyTier::Smt,
                0,
                seed,
                Some(model),
                None,
                false,
                Vec::new(),
                base_discharge,
            ))
        }
        crate::tier_b::TierBResult::Timeout => {
            if options.tier == "smt-only" {
                Some(PropertyOutcome::new(
                    property.name.clone(),
                    PropertyStatus::Unsupported,
                    PropertyTier::Smt,
                    0,
                    seed,
                    None,
                    Some("smt timeout".to_string()),
                    false,
                    Vec::new(),
                ))
            } else {
                None
            }
        }
        crate::tier_b::TierBResult::Unknown => {
            if options.tier == "smt-only" {
                Some(PropertyOutcome::new(
                    property.name.clone(),
                    PropertyStatus::Unsupported,
                    PropertyTier::Smt,
                    0,
                    seed,
                    None,
                    Some("smt unknown".to_string()),
                    false,
                    Vec::new(),
                ))
            } else {
                None
            }
        }
        crate::tier_b::TierBResult::Error(reason) => {
            if options.tier == "smt-only" {
                // smt-only is terminal: there is no Tier C to route to, so a
                // term that does not lower (a wrong-arity intrinsic, a
                // mixed-sort comparison, a transcendental cvc5 has no kind for
                // -- chelis#434) is an HONEST Unsupported, not a bug. Frame it
                // as a capability boundary ("does not lower to the SMT tier"),
                // not an internal "smt lowering error".
                Some(PropertyOutcome::new(
                    property.name.clone(),
                    PropertyStatus::Unsupported,
                    PropertyTier::Smt,
                    0,
                    seed,
                    None,
                    Some(format!("property does not lower to the SMT tier: {reason}")),
                    false,
                    Vec::new(),
                ))
            } else {
                None
            }
        }
    }
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

fn property_assumption_records(
    property_name: &str,
    smt_prop: &crate::tier_b::SmtProperty,
    discharge: AssumptionDischarge,
    non_vacuity: NonVacuityRecord,
) -> Vec<AssumptionRecord> {
    if smt_prop.preconditions.is_empty() {
        Vec::new()
    } else {
        // WI-8: stamp the prover-side discharge tier from the discharge method,
        // keyed to the property's precondition source identity.
        let name = format!("preconditions:{property_name}");
        let tier = crate::composition::DischargeTier::new(
            discharge.method.engine(),
            discharge.method,
            Some(name.clone()),
        );
        vec![
            AssumptionRecord::new(name, Some(discharge), Some(non_vacuity))
                .with_discharge_tier(tier),
        ]
    }
}

const CONSTRAINT_SAMPLING_METHOD: &str = "constraint_directed";
const REJECTION_SAMPLING_METHOD: &str = "uniform_rejection";

#[derive(Debug, Clone, Copy)]
struct ScalarBound {
    value: f64,
    strict: bool,
}

#[derive(Debug, Clone, Copy)]
struct ScalarDomain {
    lower: ScalarBound,
    upper: ScalarBound,
}

#[derive(Debug, Clone)]
struct OrderEdge {
    lower: String,
    upper: String,
    strict: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum FloatKind {
    F32,
    F64,
}

impl FloatKind {
    fn from_type_name(name: &str) -> Self {
        match name {
            "f32" => Self::F32,
            "f64" => Self::F64,
            _ => unreachable!("constraint plans only admit f32/f64 binders"),
        }
    }

    fn lower_value(self, value: f64, strict: bool) -> f64 {
        match self {
            Self::F64 => {
                if strict {
                    value.next_up()
                } else {
                    value
                }
            }
            Self::F32 => {
                let rounded = value as f32;
                if strict {
                    rounded.next_up() as f64
                } else {
                    rounded as f64
                }
            }
        }
    }

    fn upper_value(self, value: f64, strict: bool) -> f64 {
        match self {
            Self::F64 => {
                if strict {
                    value.next_down()
                } else {
                    value
                }
            }
            Self::F32 => {
                let rounded = value as f32;
                if strict {
                    rounded.next_down() as f64
                } else {
                    rounded as f64
                }
            }
        }
    }

    fn next_up(self, value: f64) -> f64 {
        match self {
            Self::F32 => (value as f32).next_up() as f64,
            Self::F64 => value.next_up(),
        }
    }

    fn next_down(self, value: f64) -> f64 {
        match self {
            Self::F32 => (value as f32).next_down() as f64,
            Self::F64 => value.next_down(),
        }
    }

    fn max_finite(self) -> f64 {
        match self {
            Self::F32 => f32::MAX as f64,
            Self::F64 => f64::MAX,
        }
    }

    fn round_into_domain(self, value: f64, lower: f64, upper: f64) -> Option<f64> {
        let mut rounded = match self {
            Self::F32 => (value as f32) as f64,
            Self::F64 => value,
        };
        if rounded < lower {
            rounded = self.next_up(rounded);
        } else if rounded > upper {
            rounded = self.next_down(rounded);
        }
        (rounded.is_finite() && lower <= rounded && rounded <= upper).then_some(rounded)
    }
}

#[derive(Debug, Clone)]
struct ConstraintSamplingPlan {
    domains: BTreeMap<String, ScalarDomain>,
    kinds: BTreeMap<String, FloatKind>,
    incoming: BTreeMap<String, Vec<(String, bool)>>,
    topo: Vec<String>,
    reserve_depth: BTreeMap<String, usize>,
}

impl ConstraintSamplingPlan {
    fn derive(
        variables: impl IntoIterator<Item = (String, FloatKind)>,
        preconditions: &[crate::solver::SmtExpr],
    ) -> Result<Self, String> {
        let kinds = variables.into_iter().collect::<BTreeMap<_, _>>();
        let names = kinds.keys().cloned().collect::<BTreeSet<_>>();
        let default = ScalarDomain {
            lower: ScalarBound {
                value: f64::NEG_INFINITY,
                strict: false,
            },
            upper: ScalarBound {
                value: f64::INFINITY,
                strict: false,
            },
        };
        let mut domains = names
            .iter()
            .cloned()
            .map(|name| (name, default))
            .collect::<BTreeMap<_, _>>();
        let mut edges = Vec::new();
        for precondition in preconditions {
            collect_scalar_constraints(precondition, &names, &mut domains, &mut edges)?;
        }

        // Propagate constant bounds through the order graph. This is enough
        // to turn `0.99 < alpha1 < alpha2 < 1.0` into two narrow domains
        // before any random draw occurs.
        for _ in 0..names.len().saturating_mul(2).max(1) {
            let mut changed = false;
            for edge in &edges {
                let lower = domains[&edge.lower];
                let upper = domains[&edge.upper];
                changed |= tighten_upper(
                    domains.get_mut(&edge.lower).expect("known lower"),
                    ScalarBound {
                        value: upper.upper.value,
                        strict: upper.upper.strict || edge.strict,
                    },
                );
                changed |= tighten_lower(
                    domains.get_mut(&edge.upper).expect("known upper"),
                    ScalarBound {
                        value: lower.lower.value,
                        strict: lower.lower.strict || edge.strict,
                    },
                );
            }
            if !changed {
                break;
            }
        }
        for (name, domain) in &domains {
            if domain.lower.value > domain.upper.value
                || (domain.lower.value == domain.upper.value
                    && (domain.lower.strict || domain.upper.strict))
            {
                return Err(format!(
                    "inconsistent scalar guards leave `{name}` with an empty interval"
                ));
            }
        }

        // A missing side is a generator choice, not a hidden semantic bound.
        // Keep the historical width only as a finite sampling window anchored
        // at the user's actual guard; never intersect an explicit domain with
        // the old uniform [-10, 10] range.
        for (name, domain) in &mut domains {
            let max_finite = kinds[name].max_finite();
            match (
                domain.lower.value.is_finite(),
                domain.upper.value.is_finite(),
            ) {
                (false, false) => {
                    domain.lower.value = -10.0;
                    domain.upper.value = 10.0;
                }
                (true, false) => {
                    let width = (domain.lower.value.abs() * 0.1).max(20.0);
                    domain.upper.value = (domain.lower.value + width).min(max_finite);
                }
                (false, true) => {
                    let width = (domain.upper.value.abs() * 0.1).max(20.0);
                    domain.lower.value = (domain.upper.value - width).max(-max_finite);
                }
                (true, true) => {}
            }
            if !domain.lower.value.is_finite() || !domain.upper.value.is_finite() {
                return Err("scalar guard bounds exceed the finite sampling range".to_string());
            }
        }

        let mut incoming_count = names
            .iter()
            .cloned()
            .map(|name| (name, 0usize))
            .collect::<BTreeMap<_, _>>();
        let mut outgoing = BTreeMap::<String, Vec<(String, bool)>>::new();
        let mut incoming = BTreeMap::<String, Vec<(String, bool)>>::new();
        for edge in &edges {
            outgoing
                .entry(edge.lower.clone())
                .or_default()
                .push((edge.upper.clone(), edge.strict));
            incoming
                .entry(edge.upper.clone())
                .or_default()
                .push((edge.lower.clone(), edge.strict));
            *incoming_count.get_mut(&edge.upper).expect("known variable") += 1;
        }
        let mut ready = incoming_count
            .iter()
            .filter_map(|(name, count)| (*count == 0).then_some(name.clone()))
            .collect::<BTreeSet<_>>();
        let mut topo = Vec::with_capacity(names.len());
        while let Some(name) = ready.pop_first() {
            topo.push(name.clone());
            for (successor, _) in outgoing.get(&name).into_iter().flatten() {
                let count = incoming_count.get_mut(successor).expect("known successor");
                *count -= 1;
                if *count == 0 {
                    ready.insert(successor.clone());
                }
            }
        }
        if topo.len() != names.len() {
            return Err(
                "inconsistent or cyclic scalar ordering guards are not sampleable".to_string(),
            );
        }

        let mut reserve_depth = names
            .iter()
            .cloned()
            .map(|name| (name, 0usize))
            .collect::<BTreeMap<_, _>>();
        for name in topo.iter().rev() {
            let depth = outgoing
                .get(name)
                .into_iter()
                .flatten()
                .map(|(successor, strict)| reserve_depth[successor] + usize::from(*strict))
                .max()
                .unwrap_or(0);
            reserve_depth.insert(name.clone(), depth);
        }
        Ok(Self {
            domains,
            kinds,
            incoming,
            topo,
            reserve_depth,
        })
    }

    fn sample(&self, rng: &mut Lcg) -> Result<BTreeMap<String, f64>, String> {
        let mut values = BTreeMap::new();
        for name in &self.topo {
            let domain = self.domains[name];
            let kind = self.kinds[name];
            let mut lower = kind.lower_value(domain.lower.value, domain.lower.strict);
            for (predecessor, strict) in self.incoming.get(name).into_iter().flatten() {
                let predecessor_value = values[predecessor];
                lower = lower.max(if *strict {
                    kind.next_up(predecessor_value)
                } else {
                    predecessor_value
                });
            }
            let mut upper = kind.upper_value(domain.upper.value, domain.upper.strict);
            for _ in 0..self.reserve_depth[name] {
                upper = kind.next_down(upper);
            }
            if !lower.is_finite() || !upper.is_finite() || lower > upper {
                return Err(format!(
                    "inconsistent scalar guards leave `{name}` with no representable sample"
                ));
            }
            let draw = if lower == upper {
                lower
            } else {
                rng.next_f64(lower, upper)
            };
            let value = kind.round_into_domain(draw, lower, upper).ok_or_else(|| {
                format!("inconsistent scalar guards leave `{name}` with no representable sample")
            })?;
            values.insert(name.clone(), value);
        }
        Ok(values)
    }
}

fn tighten_lower(domain: &mut ScalarDomain, candidate: ScalarBound) -> bool {
    if candidate.value > domain.lower.value
        || (candidate.value == domain.lower.value && candidate.strict && !domain.lower.strict)
    {
        domain.lower = candidate;
        true
    } else {
        false
    }
}

fn tighten_upper(domain: &mut ScalarDomain, candidate: ScalarBound) -> bool {
    if candidate.value < domain.upper.value
        || (candidate.value == domain.upper.value && candidate.strict && !domain.upper.strict)
    {
        domain.upper = candidate;
        true
    } else {
        false
    }
}

fn collect_scalar_constraints(
    expr: &crate::solver::SmtExpr,
    names: &BTreeSet<String>,
    domains: &mut BTreeMap<String, ScalarDomain>,
    edges: &mut Vec<OrderEdge>,
) -> Result<(), String> {
    use crate::solver::{BoolOp, CmpOp, SmtExpr};
    if let SmtExpr::Bool(BoolOp::And, terms) = expr {
        for term in terms {
            collect_scalar_constraints(term, names, domains, edges)?;
        }
        return Ok(());
    }
    let SmtExpr::Cmp(op @ (CmpOp::Lt | CmpOp::Le | CmpOp::Gt | CmpOp::Ge), left, right) = expr
    else {
        return Err(
            "unsupported scalar guard: expected a conjunction of <, <=, >, or >= comparisons"
                .to_string(),
        );
    };
    let (left, right, strict) = match op {
        CmpOp::Lt => (left.as_ref(), right.as_ref(), true),
        CmpOp::Le => (left.as_ref(), right.as_ref(), false),
        CmpOp::Gt => (right.as_ref(), left.as_ref(), true),
        CmpOp::Ge => (right.as_ref(), left.as_ref(), false),
        _ => unreachable!(),
    };
    match (left, right) {
        (literal, SmtExpr::Var(name))
            if names.contains(name) && scalar_guard_literal(literal).is_some() =>
        {
            tighten_lower(
                domains.get_mut(name).expect("known variable"),
                ScalarBound {
                    value: scalar_guard_literal(literal).expect("guarded above"),
                    strict,
                },
            );
        }
        (SmtExpr::Var(name), literal)
            if names.contains(name) && scalar_guard_literal(literal).is_some() =>
        {
            tighten_upper(
                domains.get_mut(name).expect("known variable"),
                ScalarBound {
                    value: scalar_guard_literal(literal).expect("guarded above"),
                    strict,
                },
            );
        }
        (SmtExpr::Var(lower), SmtExpr::Var(upper))
            if names.contains(lower) && names.contains(upper) =>
        {
            edges.push(OrderEdge {
                lower: lower.clone(),
                upper: upper.clone(),
                strict,
            });
        }
        _ => {
            return Err(
                "unsupported scalar guard: comparison operands must be scalar binders or literals"
                    .to_string(),
            );
        }
    }
    Ok(())
}

fn scalar_guard_literal(expr: &crate::solver::SmtExpr) -> Option<f64> {
    match expr {
        crate::solver::SmtExpr::RealLit(value) => Some(*value),
        crate::solver::SmtExpr::IntLit(value) => Some(*value as f64),
        crate::solver::SmtExpr::Arith(crate::solver::ArithOp::Neg, inner, _) => {
            scalar_guard_literal(inner).map(|value| -value)
        }
        _ => None,
    }
}

fn prove_surf_property_fuzz(
    decls: &[Decl],
    property: &Property,
    options: &PropertyRunOptions,
    seed: u64,
) -> PropertyOutcome {
    let samples_needed = if options.samples != 100 {
        options.samples
    } else {
        property.samples.unwrap_or(options.samples)
    };
    let max_attempts = options
        .max_attempts
        .unwrap_or_else(|| samples_needed.saturating_mul(100).max(samples_needed));
    let mut rng = Lcg::new(seed);

    if let Some(reason) = unsupported_property_params(&property.params) {
        return unsupported(&property.name, seed, reason);
    }

    let constraint_plan = match surf_constraint_sampling_plan(decls, property) {
        Ok(plan) => plan,
        Err(reason) => {
            return unsupported(&property.name, seed, reason).with_sampling(
                CONSTRAINT_SAMPLING_METHOD,
                0,
                0,
            );
        }
    };
    let sampling_method = if constraint_plan.is_some() {
        CONSTRAINT_SAMPLING_METHOD
    } else if property.preconditions.is_empty() {
        "uniform"
    } else {
        REJECTION_SAMPLING_METHOD
    };

    let mut accepted = 0usize;
    let mut attempts = 0usize;
    while accepted < samples_needed && attempts < max_attempts {
        attempts += 1;
        let sample = match &constraint_plan {
            Some(plan) => sample_property_with_constraints(property, plan, &mut rng),
            None => sample_property(property, &mut rng),
        };
        let sample = match sample {
            Ok(sample) => sample,
            Err(reason) => {
                return unsupported(&property.name, seed, reason).with_sampling(
                    sampling_method,
                    attempts,
                    accepted,
                );
            }
        };
        if !property.preconditions.is_empty() {
            match eval_surf_sample(decls, property, &sample, true) {
                Ok(false) => continue,
                Ok(true) => {}
                Err(err) => {
                    return error(&property.name, seed, err).with_sampling(
                        sampling_method,
                        attempts,
                        accepted,
                    );
                }
            }
        }
        accepted += 1;
        match eval_surf_sample(decls, property, &sample, false) {
            Ok(true) => {}
            Ok(false) => {
                let (shrunk, shrink_steps) = shrink_surf_counterexample(decls, property, sample);
                return PropertyOutcome::new(
                    property.name.clone(),
                    PropertyStatus::Failed,
                    PropertyTier::Fuzz,
                    accepted,
                    seed,
                    Some(counterexample_json(&shrunk)),
                    None,
                    false,
                    Vec::new(),
                )
                .with_shrink_steps(shrink_steps)
                .with_sampling(sampling_method, attempts, accepted);
            }
            Err(err) => {
                return error(&property.name, seed, err).with_sampling(
                    sampling_method,
                    attempts,
                    accepted,
                );
            }
        }
    }

    if accepted < samples_needed {
        return error(
            &property.name,
            seed,
            format!(
                "generator exhausted after {attempts} attempts before collecting {samples_needed} valid samples"
            ),
        )
        .with_sampling(sampling_method, attempts, accepted);
    }

    PropertyOutcome::new(
        property.name.clone(),
        PropertyStatus::Passed,
        PropertyTier::Fuzz,
        accepted,
        seed,
        None,
        None,
        false,
        fuzz_precondition_assumptions(
            &property.name,
            property.preconditions.len(),
            accepted,
            attempts,
            sampling_method,
            seed,
        ),
    )
    .with_sampling(sampling_method, attempts, accepted)
}

fn unsupported(name: &str, seed: u64, reason: String) -> PropertyOutcome {
    PropertyOutcome::new(
        name,
        PropertyStatus::Unsupported,
        PropertyTier::None,
        0,
        seed,
        None,
        Some(reason),
        false,
        Vec::new(),
    )
}

fn error(name: &str, seed: u64, reason: String) -> PropertyOutcome {
    PropertyOutcome::new(
        name,
        PropertyStatus::Error,
        PropertyTier::None,
        0,
        seed,
        None,
        Some(reason),
        false,
        Vec::new(),
    )
}

fn fuzz_precondition_assumptions(
    property_name: &str,
    precondition_count: usize,
    samples: usize,
    attempts: usize,
    sampling_method: &str,
    seed: u64,
) -> Vec<AssumptionRecord> {
    if precondition_count == 0 {
        return Vec::new();
    }
    let name = format!("preconditions:{property_name}");
    vec![
        AssumptionRecord::new(
            name.clone(),
            Some(AssumptionDischarge::new(
                DischargeMethod::Fuzz,
                serde_json::json!({
                    "status": "validated",
                    "property": property_name,
                    "samples": samples,
                    "attempted_samples": attempts,
                    "rejected_samples": attempts.saturating_sub(samples),
                    "sampling_method": sampling_method,
                    "seed": seed,
                    "tolerance": FUZZ_TOLERANCE,
                }),
            )),
            Some(NonVacuityRecord::established(serde_json::json!({
                "method": "fuzz",
                "result": "sat",
                "accepted_samples": samples,
                "attempted_samples": attempts,
                "rejected_samples": attempts.saturating_sub(samples),
                "sampling_method": sampling_method,
                "seed": seed,
            }))),
        )
        // WI-8: stamp the prover-side fuzz discharge tier, keyed to the
        // precondition source identity.
        .with_discharge_tier(crate::composition::DischargeTier::new(
            DischargeMethod::Fuzz.engine(),
            DischargeMethod::Fuzz,
            Some(name),
        )),
    ]
}

// ===========================================================================
// Sampling + evaluation (shared by surf and deep)
// ===========================================================================

#[derive(Debug, Clone)]
struct Sample {
    values: Vec<SampleValue>,
}

#[derive(Debug, Clone)]
struct SampleValue {
    name: String,
    surf_expr: Expr,
    deep_expr: DeepExpr,
    json: serde_json::Value,
    tensor_binding: Option<(String, TensorValue)>,
}

fn unsupported_property_params(params: &[Param]) -> Option<String> {
    for param in params {
        let ty = param.ty.as_ref()?;
        if let Some(reason) = unsupported_type(ty) {
            return Some(format!("{}: {reason}", param.name));
        }
    }
    None
}

fn unsupported_type(ty: &TypeExpr) -> Option<String> {
    match ty {
        // The supported scalar set: bool / f32 / f64 / string, plus EVERY
        // signed integer width recognized through the single-source
        // `is_int_width` (review 5), so the supported-type gate and the
        // sampler agree on which integer widths are admissible.
        TypeExpr::Named(name, _)
            if matches!(name.as_str(), "bool" | "f32" | "f64" | "string")
                || crate::opaque::is_int_width(name) =>
        {
            None
        }
        TypeExpr::Tensor(dims, precision, _) if matches!(precision.as_str(), "f32" | "f64") => {
            if dims.iter().all(
                |dim| matches!(dim, TypeExpr::Named(value, _) if value.parse::<usize>().is_ok()),
            ) {
                None
            } else {
                Some("symbolic tensor dimensions are not supported in L2 v1".to_string())
            }
        }
        TypeExpr::Tensor(_, precision, _) => Some(format!(
            "tensor element type `{precision}` is not supported in L2 v1"
        )),
        _ => Some("type is not supported in L2 v1".to_string()),
    }
}

fn sample_property(property: &Property, rng: &mut Lcg) -> Result<Sample, String> {
    let mut values = Vec::new();
    for param in &property.params {
        let ty = param
            .ty
            .as_ref()
            .ok_or_else(|| format!("{} is missing an explicit type", param.name))?;
        values.push(sample_value(&param.name, ty, rng)?);
    }
    Ok(Sample { values })
}

fn surf_constraint_sampling_plan(
    decls: &[Decl],
    property: &Property,
) -> Result<Option<ConstraintSamplingPlan>, String> {
    if property.preconditions.is_empty()
        || !property.params.iter().all(|param| {
            matches!(param.ty.as_ref(), Some(TypeExpr::Named(name, _)) if name == "f32" || name == "f64")
        })
    {
        return Ok(None);
    }
    let ctx = InlineCtx {
        decls,
        depth: 0,
        max_depth: 3,
        call_stack: vec![],
        contracts: None,
        grad_diagnostic: None,
    };
    let preconditions = property
        .preconditions
        .iter()
        .map(|precondition| {
            surf_expr_to_smt(precondition, &ctx).ok_or_else(|| {
                "unsupported scalar guard: guard does not lower to a scalar comparison".to_string()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    ConstraintSamplingPlan::derive(
        property.params.iter().map(|param| {
            let TypeExpr::Named(type_name, _) = param.ty.as_ref().expect("typed float binder")
            else {
                unreachable!("constraint plan requires scalar float binders")
            };
            (param.name.clone(), FloatKind::from_type_name(type_name))
        }),
        &preconditions,
    )
    .map(Some)
}

fn sample_property_with_constraints(
    property: &Property,
    plan: &ConstraintSamplingPlan,
    rng: &mut Lcg,
) -> Result<Sample, String> {
    let values = plan.sample(rng)?;
    let mut sampled = Vec::with_capacity(property.params.len());
    for param in &property.params {
        let TypeExpr::Named(type_name, _) = param.ty.as_ref().expect("plan requires typed params")
        else {
            unreachable!("plan requires scalar params")
        };
        let value = values[&param.name];
        sampled.push(directed_float_sample(&param.name, type_name, value));
    }
    Ok(Sample { values: sampled })
}

fn sample_value(name: &str, ty: &TypeExpr, rng: &mut Lcg) -> Result<SampleValue, String> {
    let sp = chelis_deep::Span::new(0, 0);
    match ty {
        TypeExpr::Named(type_name, _) if type_name == "bool" => {
            let value = rng.next_bool();
            Ok(scalar_sample(
                name,
                Expr::Lit(Literal::Bool(value), sp),
                deep_lit(deep_bool(value), "bool"),
                serde_json::json!(value),
            ))
        }
        // Every signed integer width, recognized through the single-source
        // `is_int_width` and sampled within the width's representable range
        // via the single-source `int_sample_bounds` (review 5): an int8
        // samples in [-128, 127], never an unrepresentable value.
        TypeExpr::Named(type_name, _) if crate::opaque::is_int_width(type_name) => {
            let (lo, hi) = crate::opaque::int_sample_bounds(type_name)
                .expect("is_int_width implies int_sample_bounds");
            let value = rng.next_i64(lo, hi);
            let lit = Expr::Lit(Literal::Int(value), sp);
            if type_name == "int32" {
                // int32 is the integer-literal default; no cast needed.
                Ok(scalar_sample(
                    name,
                    lit,
                    deep_lit(deep_int(value), "int32"),
                    serde_json::json!(value),
                ))
            } else {
                Ok(scalar_sample(
                    name,
                    cast_expr(lit, type_name),
                    deep_lit(deep_int(value), type_name),
                    serde_json::json!(value),
                ))
            }
        }
        TypeExpr::Named(type_name, _) if type_name == "f32" || type_name == "f64" => {
            let value = rng.next_f64(-10.0, 10.0);
            Ok(float_sample(name, type_name, value))
        }
        TypeExpr::Named(type_name, _) if type_name == "string" => {
            let value = format!("s{}", rng.next_u64() % 1000);
            Ok(scalar_sample(
                name,
                Expr::Lit(Literal::Str(value.clone()), sp),
                deep_lit(deep_string(&value), "string"),
                serde_json::json!(value),
            ))
        }
        TypeExpr::Tensor(dims, precision, _) => sample_tensor_value(name, dims, precision, rng),
        _ => Err("type is not supported in L2 v1".to_string()),
    }
}

fn scalar_sample(
    name: &str,
    surf_expr: Expr,
    deep_expr: DeepExpr,
    json: serde_json::Value,
) -> SampleValue {
    SampleValue {
        name: name.to_string(),
        surf_expr,
        deep_expr,
        json,
        tensor_binding: None,
    }
}

fn directed_float_sample(name: &str, type_name: &str, value: f64) -> SampleValue {
    let machine_value = if type_name == "f32" {
        (value as f32) as f64
    } else {
        value
    };
    float_sample(name, type_name, machine_value)
}

fn sample_tensor_value(
    name: &str,
    dims: &[TypeExpr],
    precision: &str,
    rng: &mut Lcg,
) -> Result<SampleValue, String> {
    let shape = dims
        .iter()
        .map(|dim| match dim {
            TypeExpr::Named(value, _) => value
                .parse::<usize>()
                .map_err(|_| "symbolic tensor dimensions are not supported in L2 v1".to_string()),
            _ => Err("symbolic tensor dimensions are not supported in L2 v1".to_string()),
        })
        .collect::<Result<Vec<_>, _>>()?;
    if shape.is_empty() || shape.len() > 2 {
        return Err(
            "only rank-1 and rank-2 fixed-shape tensors are supported in L2 v1".to_string(),
        );
    }
    let count = shape.iter().product::<usize>();
    let values = (0..count)
        .map(|_| rng.next_f64(-10.0, 10.0))
        .collect::<Vec<_>>();
    let surf_expr = tensor_surf_expr(&shape, precision, &values);
    let deep_expr = tensor_deep_expr(&shape, precision, &values);
    Ok(SampleValue {
        name: name.to_string(),
        surf_expr,
        deep_expr,
        json: serde_json::json!({ "shape": shape, "data": values }),
        tensor_binding: None,
    })
}

fn tensor_surf_expr(shape: &[usize], precision: &str, values: &[f64]) -> Expr {
    let sp = chelis_deep::Span::new(0, 0);
    let scalar = |value| {
        let lit = Expr::Lit(Literal::Float(value), sp);
        if precision == "f64" {
            cast_expr(lit, "f64")
        } else {
            lit
        }
    };
    if shape.len() == 1 {
        return Expr::Apply(
            Box::new(Expr::Var("to_tensor".to_string(), sp)),
            vec![Expr::List(values.iter().copied().map(scalar).collect(), sp)],
            sp,
        );
    }
    let cols = shape[1];
    let rows = values
        .chunks(cols)
        .map(|row| Expr::List(row.iter().copied().map(scalar).collect(), sp))
        .collect::<Vec<_>>();
    Expr::Apply(
        Box::new(Expr::Var("pad_sequences".to_string(), sp)),
        vec![Expr::List(rows, sp), scalar(0.0)],
        sp,
    )
}

fn tensor_deep_expr(shape: &[usize], precision: &str, values: &[f64]) -> DeepExpr {
    let scalar = |value| deep_numeric_tensor_scalar(value, precision);
    if shape.len() == 1 {
        return deep_node(
            "app",
            vec![
                deep_var("to_tensor"),
                deep_cons_list(values.iter().copied().map(scalar).collect()),
            ],
        );
    }
    let cols = shape[1];
    let rows = values
        .chunks(cols)
        .map(|row| deep_cons_list(row.iter().copied().map(scalar).collect()))
        .collect::<Vec<_>>();
    deep_node(
        "app",
        vec![
            deep_var("pad_sequences"),
            deep_cons_list(rows),
            deep_numeric_tensor_scalar(0.0, precision),
        ],
    )
}

fn deep_numeric_tensor_scalar(value: f64, precision: &str) -> DeepExpr {
    let lit = deep_lit(deep_float(value), "f32");
    if precision == "f64" {
        deep_node(
            "cast",
            vec![lit, deep_node("t-prim", vec![deep_symbol("f64")])],
        )
    } else {
        lit
    }
}

fn deep_cons_list(items: Vec<DeepExpr>) -> DeepExpr {
    items.into_iter().rev().fold(deep_var("Nil"), |tail, item| {
        deep_node("app", vec![deep_var("Cons"), item, tail])
    })
}

fn cast_expr(expr: Expr, ty: &str) -> Expr {
    Expr::Cast(Box::new(expr), ty.to_string(), chelis_deep::Span::new(0, 0))
}

fn eval_surf_sample(
    decls: &[Decl],
    property: &Property,
    sample: &Sample,
    precondition: bool,
) -> Result<bool, String> {
    let root = if precondition {
        "__chelis_property_pre"
    } else {
        "__chelis_property_probe"
    };
    let mut source_decls = decls.to_vec();
    for value in &sample.values {
        if let Some((binding_name, tensor)) = &value.tensor_binding {
            source_decls.push(Decl::Sig {
                name: binding_name.clone(),
                ty: property
                    .params
                    .iter()
                    .find(|param| param.name == value.name)
                    .and_then(|param| param.ty.clone())
                    .expect("tensor property params are typed"),
                effects: None,
                span: chelis_deep::Span::new(0, 0),
            });
            debug_assert_eq!(tensor.data.len(), tensor.shape.iter().product::<usize>());
        }
    }
    source_decls.push(Decl::LetDef {
        name: root.to_string(),
        ty: None,
        value: sample_block_expr(property, sample, precondition),
        span: chelis_deep::Span::new(0, 0),
    });
    let source = chelis_surf::format::format_program(&source_decls);
    eval_bool_with_bindings(SourceKind::Surf, source, root, sample_bindings(sample))
}

fn sample_block_expr(property: &Property, sample: &Sample, precondition: bool) -> Expr {
    let sp = chelis_deep::Span::new(0, 0);
    let bindings = sample
        .values
        .iter()
        .map(|value| LetBinding {
            pattern: LetPattern::Var(value.name.clone(), sp),
            ty: None,
            value: value.surf_expr.clone(),
        })
        .collect::<Vec<_>>();
    let body = if precondition {
        combine_preconditions(&property.preconditions)
    } else {
        Expr::Apply(
            Box::new(Expr::Var(property.name.clone(), sp)),
            property
                .params
                .iter()
                .map(|param| Expr::Var(param.name.clone(), sp))
                .collect(),
            sp,
        )
    };
    Expr::Block(bindings, Box::new(body), sp)
}

fn combine_preconditions(preconditions: &[Expr]) -> Expr {
    let sp = chelis_deep::Span::new(0, 0);
    preconditions
        .iter()
        .cloned()
        .reduce(|left, right| Expr::Binary(BinOp::And, Box::new(left), Box::new(right), sp))
        .unwrap_or(Expr::Lit(Literal::Bool(true), sp))
}

fn eval_bool_with_bindings(
    source_kind: SourceKind,
    source: String,
    root: &str,
    bindings: BTreeMap<String, TensorValue>,
) -> Result<bool, String> {
    let result = chelis_compiler_api::compiler::eval_selected(
        EvalRequest {
            source_kind,
            source,
            bindings,
        },
        &[root.to_string()],
    )
    .map_err(|err| {
        err.errors
            .iter()
            .map(|diag| diag.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    match result.roots.as_slice() {
        [root] => match &root.value {
            ExecutionValue::Bool { value } => Ok(*value),
            ExecutionValue::Tensor { value } if value.shape.is_empty() && value.data.len() == 1 => {
                Ok(value.data.element_as_f64_lossy(0) != 0.0)
            }
            other => Err(format!(
                "property root evaluated to non-bool value: {other:?}"
            )),
        },
        _ => Err("property evaluation did not return exactly one root".to_string()),
    }
}

fn sample_bindings(sample: &Sample) -> BTreeMap<String, TensorValue> {
    sample
        .values
        .iter()
        .filter_map(|value| value.tensor_binding.clone())
        .collect()
}

fn counterexample_json(sample: &Sample) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for value in &sample.values {
        map.insert(value.name.clone(), value.json.clone());
    }
    serde_json::Value::Object(map)
}

const MAX_SHRINK_STEPS: usize = 64;

fn shrink_surf_counterexample(
    decls: &[Decl],
    property: &Property,
    sample: Sample,
) -> (Sample, usize) {
    shrink_counterexample(sample, &property.params, |candidate| {
        sample_still_fails_surf(decls, property, candidate)
    })
}

fn shrink_deep_counterexample(
    exprs: &[DeepExpr],
    property: &DeepProperty,
    sample: Sample,
) -> (Sample, usize) {
    shrink_counterexample(sample, &property.params, |candidate| {
        sample_still_fails_deep(exprs, property, candidate)
    })
}

fn shrink_counterexample<F>(
    mut sample: Sample,
    params: &[Param],
    mut still_fails: F,
) -> (Sample, usize)
where
    F: FnMut(&Sample) -> bool,
{
    let mut steps = 0usize;
    while steps < MAX_SHRINK_STEPS {
        let mut changed = false;
        for index in 0..sample.values.len() {
            let Some(ty) = params
                .iter()
                .find(|param| param.name == sample.values[index].name)
                .and_then(|param| param.ty.as_ref())
            else {
                continue;
            };
            for candidate in shrink_candidates(&sample.values[index], ty) {
                if candidate.json == sample.values[index].json {
                    continue;
                }
                let mut trial = sample.clone();
                trial.values[index] = candidate;
                if still_fails(&trial) {
                    sample = trial;
                    steps += 1;
                    changed = true;
                    break;
                }
            }
            if changed || steps >= MAX_SHRINK_STEPS {
                break;
            }
        }
        if !changed {
            break;
        }
    }
    (sample, steps)
}

fn sample_still_fails_surf(decls: &[Decl], property: &Property, sample: &Sample) -> bool {
    if !property.preconditions.is_empty() {
        match eval_surf_sample(decls, property, sample, true) {
            Ok(true) => {}
            Ok(false) | Err(_) => return false,
        }
    }
    matches!(eval_surf_sample(decls, property, sample, false), Ok(false))
}

fn sample_still_fails_deep(exprs: &[DeepExpr], property: &DeepProperty, sample: &Sample) -> bool {
    if !property.preconditions.is_empty() {
        match eval_deep_sample(exprs, property, sample, true) {
            Ok(true) => {}
            Ok(false) | Err(_) => return false,
        }
    }
    matches!(eval_deep_sample(exprs, property, sample, false), Ok(false))
}

fn shrink_candidates(value: &SampleValue, ty: &TypeExpr) -> Vec<SampleValue> {
    match ty {
        TypeExpr::Named(type_name, _) if type_name == "bool" => value
            .json
            .as_bool()
            .and_then(|current| current.then(|| bool_sample(&value.name, false)))
            .into_iter()
            .collect(),
        TypeExpr::Named(type_name, _) if crate::opaque::is_int_width(type_name) => value
            .json
            .as_i64()
            .map(|current| {
                let mut candidates = Vec::new();
                push_unique_int_candidate(&mut candidates, &value.name, type_name, current, 0);
                push_unique_int_candidate(
                    &mut candidates,
                    &value.name,
                    type_name,
                    current,
                    current / 2,
                );
                push_unique_int_candidate(
                    &mut candidates,
                    &value.name,
                    type_name,
                    current,
                    current.signum(),
                );
                candidates
            })
            .unwrap_or_default(),
        TypeExpr::Named(type_name, _) if type_name == "f32" || type_name == "f64" => value
            .json
            .as_f64()
            .map(|current| {
                let mut candidates = Vec::new();
                push_unique_float_candidate(&mut candidates, &value.name, type_name, current, 0.0);
                push_unique_float_candidate(
                    &mut candidates,
                    &value.name,
                    type_name,
                    current,
                    current / 2.0,
                );
                candidates
            })
            .unwrap_or_default(),
        TypeExpr::Named(type_name, _) if type_name == "string" => value
            .json
            .as_str()
            .map(|current| {
                let mut candidates = Vec::new();
                push_unique_string_candidate(&mut candidates, &value.name, current, "");
                if !current.is_empty() {
                    push_unique_string_candidate(
                        &mut candidates,
                        &value.name,
                        current,
                        &current[..current.len() / 2],
                    );
                }
                candidates
            })
            .unwrap_or_default(),
        TypeExpr::Tensor(_, precision, _) => tensor_shrink_candidates(value, precision),
        _ => Vec::new(),
    }
}

fn push_unique_int_candidate(
    candidates: &mut Vec<SampleValue>,
    name: &str,
    type_name: &str,
    current: i64,
    candidate: i64,
) {
    if candidate != current
        && !candidates
            .iter()
            .any(|sample| sample.json.as_i64() == Some(candidate))
    {
        candidates.push(int_sample(name, type_name, candidate));
    }
}

fn push_unique_float_candidate(
    candidates: &mut Vec<SampleValue>,
    name: &str,
    type_name: &str,
    current: f64,
    candidate: f64,
) {
    if (candidate - current).abs() > f64::EPSILON
        && !candidates.iter().any(|sample| {
            sample
                .json
                .as_f64()
                .is_some_and(|prior| (prior - candidate).abs() <= f64::EPSILON)
        })
    {
        candidates.push(float_sample(name, type_name, candidate));
    }
}

fn push_unique_string_candidate(
    candidates: &mut Vec<SampleValue>,
    name: &str,
    current: &str,
    candidate: &str,
) {
    if candidate != current
        && !candidates
            .iter()
            .any(|sample| sample.json.as_str() == Some(candidate))
    {
        candidates.push(string_sample(name, candidate));
    }
}

fn bool_sample(name: &str, value: bool) -> SampleValue {
    scalar_sample(
        name,
        Expr::Lit(Literal::Bool(value), chelis_deep::Span::new(0, 0)),
        deep_lit(deep_bool(value), "bool"),
        serde_json::json!(value),
    )
}

fn int_sample(name: &str, type_name: &str, value: i64) -> SampleValue {
    let (lo, hi) =
        crate::opaque::int_sample_bounds(type_name).expect("int shrink only uses int widths");
    let value = value.clamp(lo, hi);
    let lit = Expr::Lit(Literal::Int(value), chelis_deep::Span::new(0, 0));
    let surf_expr = if type_name == "int32" {
        lit
    } else {
        cast_expr(lit, type_name)
    };
    scalar_sample(
        name,
        surf_expr,
        deep_lit(deep_int(value), type_name),
        serde_json::json!(value),
    )
}

fn float_sample(name: &str, type_name: &str, value: f64) -> SampleValue {
    let lit = Expr::Lit(Literal::Float(value), chelis_deep::Span::new(0, 0));
    let surf_expr = if type_name == "f64" {
        cast_expr(lit, "f64")
    } else {
        lit
    };
    scalar_sample(
        name,
        surf_expr,
        deep_lit(deep_float(value), type_name),
        serde_json::json!(value),
    )
}

fn string_sample(name: &str, value: &str) -> SampleValue {
    scalar_sample(
        name,
        Expr::Lit(
            Literal::Str(value.to_string()),
            chelis_deep::Span::new(0, 0),
        ),
        deep_lit(deep_string(value), "string"),
        serde_json::json!(value),
    )
}

fn tensor_shrink_candidates(value: &SampleValue, precision: &str) -> Vec<SampleValue> {
    let Some(shape) = value
        .json
        .get("shape")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.as_u64().map(|dim| dim as usize))
                .collect::<Vec<_>>()
        })
    else {
        return Vec::new();
    };
    let Some(data) = value
        .json
        .get("data")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(serde_json::Value::as_f64)
                .collect::<Vec<_>>()
        })
    else {
        return Vec::new();
    };
    if shape.iter().product::<usize>() != data.len() {
        return Vec::new();
    }
    let mut candidates = Vec::new();
    let zeros = vec![0.0; data.len()];
    if data.iter().any(|value| value.abs() > f64::EPSILON) {
        candidates.push(tensor_sample(&value.name, &shape, precision, &zeros));
    }
    let halves = data.iter().map(|value| value / 2.0).collect::<Vec<_>>();
    if halves
        .iter()
        .zip(&data)
        .any(|(candidate, current)| (candidate - current).abs() > f64::EPSILON)
    {
        candidates.push(tensor_sample(&value.name, &shape, precision, &halves));
    }
    candidates
}

fn tensor_sample(name: &str, shape: &[usize], precision: &str, values: &[f64]) -> SampleValue {
    SampleValue {
        name: name.to_string(),
        surf_expr: tensor_surf_expr(shape, precision, values),
        deep_expr: tensor_deep_expr(shape, precision, values),
        json: serde_json::json!({ "shape": shape, "data": values }),
        tensor_binding: None,
    }
}

// ===========================================================================
// Deep property model + discovery + running
// ===========================================================================

#[derive(Debug, Clone)]
struct DeepProperty {
    name: String,
    params: Vec<Param>,
    preconditions: Vec<DeepExpr>,
    body: DeepExpr,
    samples: Option<usize>,
    seed: Option<u64>,
}

fn discover_deep_properties(
    exprs: &[DeepExpr],
    only: Option<&str>,
) -> Result<Vec<DeepProperty>, String> {
    let mut out = Vec::new();
    for expr in exprs {
        discover_deep_properties_expr(expr, only, &mut out)?;
    }
    Ok(out)
}

fn discover_deep_properties_expr(
    expr: &DeepExpr,
    only: Option<&str>,
    out: &mut Vec<DeepProperty>,
) -> Result<(), String> {
    let Some((tag, meta, children)) = deep_node_parts(expr) else {
        return Ok(());
    };
    if tag == DeepTag::Def
        && let Some(name) = children.first().and_then(symbol_text)
    {
        // Classify the def by source kind the SAME way the CLI discoverer
        // does (F6): a `chelis_role: "property"` def with an absent or
        // invalid `property_source_kind` is an ERROR (not silently skipped,
        // which previously made tide report total:0 while the CLI errored).
        // Only a `user` property is run by the shared runner; a
        // `bridge:c-earchin` property is skipped here (the CLI's bridge path
        // owns it, with its span/requirement rendering).
        match deep_property_source_kind(meta, name)? {
            Some(DeepSourceKind::User) => {
                let fn_expr = children
                    .get(1)
                    .ok_or_else(|| format!("property `{name}` def is missing a fn body"))?;
                let fn_params = deep_fn_params(fn_expr)
                    .ok_or_else(|| format!("property `{name}` def body must be a callable `fn`"))?;
                let params = if let Some(params) = deep_property_params(meta) {
                    if !params_match(&params, &fn_params) {
                        return Err(format!(
                            "property `{name}` property_quantifiers must match fn parameters"
                        ));
                    }
                    params
                } else {
                    return Err(format!(
                        "property `{name}` metadata must include `property_quantifiers`"
                    ));
                };
                if matches_filter(name, only) {
                    out.push(DeepProperty {
                        name: name.to_string(),
                        params,
                        preconditions: deep_property_preconditions(meta).unwrap_or_default(),
                        body: deep_fn_body(fn_expr).cloned().ok_or_else(|| {
                            format!("property `{name}` def body must be a callable `fn`")
                        })?,
                        samples: deep_int_meta(meta, "property_samples"),
                        seed: deep_int_meta(meta, "property_seed").map(|value| value as u64),
                    });
                }
            }
            // A bridge:c-earchin property is not the shared runner's concern;
            // a non-property def has no role. Both are skipped.
            Some(DeepSourceKind::Bridge) | None => {}
        }
    }
    for child in children {
        discover_deep_properties_expr(child, only, out)?;
    }
    Ok(())
}

/// The source kind of a Deep `@property` def.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepSourceKind {
    User,
    Bridge,
}

/// Classify a Deep def's `chelis_role`/`property_source_kind` metadata,
/// matching the CLI discoverer's `property_source_kind` (F6): `None` for a
/// non-property def; `Err` for a `chelis_role: "property"` def with an
/// absent or invalid `property_source_kind` (a malformed property is an
/// error on BOTH surfaces, never silently skipped). The legacy
/// `c_earchin_role` witness without an explicit kind defaults to Bridge.
fn deep_property_source_kind(meta: &MetaMap, name: &str) -> Result<Option<DeepSourceKind>, String> {
    let has_chelis = meta
        .entries
        .iter()
        .any(|(k, v)| k == "chelis_role" && string_value(v) == Some("property"));
    let has_legacy = meta
        .entries
        .iter()
        .any(|(k, v)| k == "c_earchin_role" && string_value(v) == Some("property_witness"));
    if !has_chelis && !has_legacy {
        return Ok(None);
    }
    let Some(kind) = deep_meta_value(meta, "property_source_kind").and_then(string_value) else {
        if has_legacy {
            return Ok(Some(DeepSourceKind::Bridge));
        }
        return Err(format!(
            "property `{name}` metadata must include string `property_source_kind`"
        ));
    };
    match kind {
        "user" => Ok(Some(DeepSourceKind::User)),
        "bridge:c-earchin" => Ok(Some(DeepSourceKind::Bridge)),
        other => Err(format!(
            "property `{name}` metadata has invalid property_source_kind `{other}`"
        )),
    }
}

fn prove_deep_property(
    exprs: &[DeepExpr],
    property: &DeepProperty,
    options: &PropertyRunOptions,
) -> PropertyOutcome {
    let seed = options.effective_seed(property.seed);

    // chelis#978's production induction classifier consumes checked Surf AST.
    // Deep has no equivalent structural-recursion ownership record yet. An
    // explicit induction request is therefore terminal on Deep: never let the
    // generic tail below reinterpret it as fuzz-only and launder samples into
    // a pass.
    if options.tier == "induction-only" {
        return PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Unsupported,
            PropertyTier::Induction,
            0,
            seed,
            None,
            Some(
                "induction-only is unavailable for Deep properties: no compiler-AST structural recursion attribution (chelis#978)"
                    .to_string(),
            ),
            false,
            Vec::new(),
        );
    }

    if options.tier == "auto" || options.tier == "smt-only" {
        if let Some(outcome) = try_deep_tier_b(exprs, property, options, seed) {
            return outcome;
        }
        if options.tier == "smt-only" {
            return PropertyOutcome::new(
                property.name.clone(),
                PropertyStatus::Unsupported,
                PropertyTier::Smt,
                0,
                seed,
                None,
                Some("property does not lower to Tier B (smt-only)".to_string()),
                false,
                Vec::new(),
            );
        }
    }

    let samples_needed = if options.samples != 100 {
        options.samples
    } else {
        property.samples.unwrap_or(options.samples)
    };
    let max_attempts = options
        .max_attempts
        .unwrap_or_else(|| samples_needed.saturating_mul(100).max(samples_needed));
    let mut rng = Lcg::new(seed);

    if let Some(reason) = unsupported_property_params(&property.params) {
        return unsupported(&property.name, seed, reason);
    }

    let constraint_plan = match deep_constraint_sampling_plan(exprs, property) {
        Ok(plan) => plan,
        Err(reason) => {
            return unsupported(&property.name, seed, reason).with_sampling(
                CONSTRAINT_SAMPLING_METHOD,
                0,
                0,
            );
        }
    };
    let sampling_method = if constraint_plan.is_some() {
        CONSTRAINT_SAMPLING_METHOD
    } else if property.preconditions.is_empty() {
        "uniform"
    } else {
        REJECTION_SAMPLING_METHOD
    };

    let mut accepted = 0usize;
    let mut attempts = 0usize;
    while accepted < samples_needed && attempts < max_attempts {
        attempts += 1;
        let sample = match &constraint_plan {
            Some(plan) => sample_deep_property_with_constraints(property, plan, &mut rng),
            None => sample_deep_property(property, &mut rng),
        };
        let sample = match sample {
            Ok(sample) => sample,
            Err(reason) => {
                return unsupported(&property.name, seed, reason).with_sampling(
                    sampling_method,
                    attempts,
                    accepted,
                );
            }
        };
        if !property.preconditions.is_empty() {
            match eval_deep_sample(exprs, property, &sample, true) {
                Ok(false) => continue,
                Ok(true) => {}
                Err(err) => {
                    return error(&property.name, seed, err).with_sampling(
                        sampling_method,
                        attempts,
                        accepted,
                    );
                }
            }
        }
        accepted += 1;
        match eval_deep_sample(exprs, property, &sample, false) {
            Ok(true) => {}
            Ok(false) => {
                let (shrunk, shrink_steps) = shrink_deep_counterexample(exprs, property, sample);
                return PropertyOutcome::new(
                    property.name.clone(),
                    PropertyStatus::Failed,
                    PropertyTier::Fuzz,
                    accepted,
                    seed,
                    Some(counterexample_json(&shrunk)),
                    None,
                    false,
                    Vec::new(),
                )
                .with_shrink_steps(shrink_steps)
                .with_sampling(sampling_method, attempts, accepted);
            }
            Err(err) => {
                return error(&property.name, seed, err).with_sampling(
                    sampling_method,
                    attempts,
                    accepted,
                );
            }
        }
    }

    if accepted < samples_needed {
        return error(
            &property.name,
            seed,
            format!(
                "generator exhausted after {attempts} attempts before collecting {samples_needed} valid samples"
            ),
        )
        .with_sampling(sampling_method, attempts, accepted);
    }

    PropertyOutcome::new(
        property.name.clone(),
        PropertyStatus::Passed,
        PropertyTier::Fuzz,
        accepted,
        seed,
        None,
        None,
        false,
        fuzz_precondition_assumptions(
            &property.name,
            property.preconditions.len(),
            accepted,
            attempts,
            sampling_method,
            seed,
        ),
    )
    .with_sampling(sampling_method, attempts, accepted)
}

fn try_deep_tier_b(
    exprs: &[DeepExpr],
    property: &DeepProperty,
    options: &PropertyRunOptions,
    seed: u64,
) -> Option<PropertyOutcome> {
    let ctx = DeepInlineCtx {
        exprs,
        depth: 0,
        max_depth: 3,
        call_stack: vec![],
    };
    let postcondition = deep_expr_to_smt(&property.body, &ctx)?;
    let variables: Vec<(String, crate::solver::SmtSort)> = property
        .params
        .iter()
        .filter_map(|param| {
            let sort = match param.ty.as_ref()? {
                TypeExpr::Named(name, _)
                    if matches!(name.as_str(), "f32" | "f64" | "bool")
                        || crate::opaque::is_int_width(name) =>
                {
                    crate::opaque::prim_to_smt_sort(name)
                }
                _ => return None,
            };
            Some((param.name.clone(), sort))
        })
        .collect();
    if variables.len() != property.params.len() {
        return None;
    }
    let preconditions: Vec<crate::solver::SmtExpr> = property
        .preconditions
        .iter()
        .filter_map(|expr| deep_expr_to_smt(expr, &ctx))
        .collect();
    if preconditions.len() != property.preconditions.len() {
        return None;
    }
    let smt_prop = crate::tier_b::SmtProperty {
        variables,
        preconditions,
        postcondition,
    };
    if !matches!(
        crate::classify_inlineability(&smt_prop.postcondition),
        crate::Inlineability::Inlineable
    ) {
        // chelis#434 ENVELOPE LANE: the goal carries transcendentals the base
        // lowering cannot inline. Try discharging through the certified
        // special-function envelopes (normal_cdf -> erf, then abstract-subterm)
        // BEFORE falling through to fuzz. Returns Some ONLY on a genuine proof of
        // the residual (-> proven_modulo_certified_envelope); on any non-proof it
        // declines and the existing honest paths (smt-only transcendental
        // `unsupported`, or auto fuzz) run unchanged.
        if let Some(outcome) = try_envelope_lane(&property.name, &smt_prop, options, seed) {
            return Some(outcome);
        }
        return None;
    }
    let discharge = crate::engine_registry::DischargeRegistry::with_builtin_engines().dispatch(
        &crate::discharge::Goal::smt(smt_prop.clone()),
        options.smt_timeout_ms,
    );
    let base_discharge = Some((discharge.soundness(), discharge.qualifier_set().clone()));
    match discharge.into_result() {
        crate::tier_b::TierBResult::Proved => {
            let non_vacuity = smt_non_vacuity_record(&smt_prop, options.smt_timeout_ms);
            let reason = match non_vacuity.status {
                NonVacuityStatus::Established => None,
                NonVacuityStatus::Invalid | NonVacuityStatus::Unsupported => {
                    non_vacuity.reason.clone()
                }
            };
            let assumptions = property_assumption_records(
                &property.name,
                &smt_prop,
                AssumptionDischarge::new(
                    DischargeMethod::Smt,
                    serde_json::json!({
                        "status": "proved",
                        "property": property.name,
                        "arith_model": "real",
                    }),
                ),
                non_vacuity,
            );
            let status = if reason.is_some() {
                PropertyStatus::Unsupported
            } else {
                PropertyStatus::Passed
            };
            Some(PropertyOutcome::with_base_discharge(
                property.name.clone(),
                status,
                PropertyTier::Smt,
                0,
                seed,
                None,
                reason,
                false,
                assumptions,
                base_discharge,
            ))
        }
        crate::tier_b::TierBResult::Disproved(model) => Some(PropertyOutcome::with_base_discharge(
            property.name.clone(),
            PropertyStatus::Failed,
            PropertyTier::Smt,
            0,
            seed,
            Some(model),
            None,
            false,
            Vec::new(),
            base_discharge,
        )),
        crate::tier_b::TierBResult::Timeout => {
            if options.tier == "smt-only" {
                Some(PropertyOutcome::new(
                    property.name.clone(),
                    PropertyStatus::Unsupported,
                    PropertyTier::Smt,
                    0,
                    seed,
                    None,
                    Some("smt timeout".to_string()),
                    false,
                    Vec::new(),
                ))
            } else {
                None
            }
        }
        crate::tier_b::TierBResult::Unknown => {
            if options.tier == "smt-only" {
                Some(PropertyOutcome::new(
                    property.name.clone(),
                    PropertyStatus::Unsupported,
                    PropertyTier::Smt,
                    0,
                    seed,
                    None,
                    Some("smt unknown".to_string()),
                    false,
                    Vec::new(),
                ))
            } else {
                None
            }
        }
        crate::tier_b::TierBResult::Error(reason) => {
            if options.tier == "smt-only" {
                Some(PropertyOutcome::new(
                    property.name.clone(),
                    PropertyStatus::Unsupported,
                    PropertyTier::Smt,
                    0,
                    seed,
                    None,
                    Some(format!("property does not lower to the SMT tier: {reason}")),
                    false,
                    Vec::new(),
                ))
            } else {
                None
            }
        }
    }
}

fn sample_deep_property(property: &DeepProperty, rng: &mut Lcg) -> Result<Sample, String> {
    let mut values = Vec::new();
    for param in &property.params {
        let ty = param
            .ty
            .as_ref()
            .ok_or_else(|| format!("{} is missing an explicit type", param.name))?;
        values.push(sample_value(&param.name, ty, rng)?);
    }
    Ok(Sample { values })
}

fn deep_constraint_sampling_plan(
    exprs: &[DeepExpr],
    property: &DeepProperty,
) -> Result<Option<ConstraintSamplingPlan>, String> {
    if property.preconditions.is_empty()
        || !property.params.iter().all(|param| {
            matches!(param.ty.as_ref(), Some(TypeExpr::Named(name, _)) if name == "f32" || name == "f64")
        })
    {
        return Ok(None);
    }
    let ctx = DeepInlineCtx {
        exprs,
        depth: 0,
        max_depth: 3,
        call_stack: vec![],
    };
    let preconditions = property
        .preconditions
        .iter()
        .map(|precondition| {
            deep_expr_to_smt(precondition, &ctx).ok_or_else(|| {
                "unsupported scalar guard: guard does not lower to a scalar comparison".to_string()
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    ConstraintSamplingPlan::derive(
        property.params.iter().map(|param| {
            let TypeExpr::Named(type_name, _) = param.ty.as_ref().expect("typed float binder")
            else {
                unreachable!("constraint plan requires scalar float binders")
            };
            (param.name.clone(), FloatKind::from_type_name(type_name))
        }),
        &preconditions,
    )
    .map(Some)
}

fn sample_deep_property_with_constraints(
    property: &DeepProperty,
    plan: &ConstraintSamplingPlan,
    rng: &mut Lcg,
) -> Result<Sample, String> {
    let values = plan.sample(rng)?;
    let mut sampled = Vec::with_capacity(property.params.len());
    for param in &property.params {
        let TypeExpr::Named(type_name, _) = param.ty.as_ref().expect("plan requires typed params")
        else {
            unreachable!("plan requires scalar params")
        };
        sampled.push(directed_float_sample(
            &param.name,
            type_name,
            values[&param.name],
        ));
    }
    Ok(Sample { values: sampled })
}

fn eval_deep_sample(
    exprs: &[DeepExpr],
    property: &DeepProperty,
    sample: &Sample,
    precondition: bool,
) -> Result<bool, String> {
    let root = if precondition {
        "__chelis_property_pre"
    } else {
        "__chelis_property_probe"
    };
    let mut source_exprs = exprs.to_vec();
    for value in &sample.values {
        if let Some((binding_name, _)) = &value.tensor_binding
            && let Some(ty) = property
                .params
                .iter()
                .find(|param| param.name == value.name)
                .and_then(|param| param.ty.as_ref())
                .and_then(surf_type_to_deep)
        {
            source_exprs.push(deep_node("defsig", vec![deep_symbol(binding_name), ty]));
        }
    }
    source_exprs.push(deep_node(
        "def",
        vec![
            deep_symbol(root),
            deep_sample_block_expr(property, sample, precondition),
        ],
    ));
    let source = chelis_deep::printer::print_canonical(&source_exprs);
    eval_bool_with_bindings(SourceKind::Deep, source, root, sample_bindings(sample))
}

fn deep_sample_block_expr(
    property: &DeepProperty,
    sample: &Sample,
    precondition: bool,
) -> DeepExpr {
    let body = if precondition {
        combine_deep_preconditions(&property.preconditions)
    } else {
        property.body.clone()
    };
    if sample.values.is_empty() {
        return body;
    }
    let mut bind_children = Vec::new();
    for value in &sample.values {
        bind_children.push(deep_symbol(&value.name));
        bind_children.push(value.deep_expr.clone());
    }
    deep_node("let", vec![deep_node("bind", bind_children), body])
}

fn combine_deep_preconditions(preconditions: &[DeepExpr]) -> DeepExpr {
    preconditions
        .iter()
        .cloned()
        .reduce(|left, right| deep_node("app", vec![deep_var("and"), left, right]))
        .unwrap_or_else(|| deep_lit(deep_bool(true), "bool"))
}

/// The discharged proposition of a Deep property as a Deep term (chelis#436):
/// the bare `body` when unguarded, or the implication `(implies (/\ pre) body)`
/// when guarded, so the rendered `goal` is exactly what the prover discharged
/// and a guarded property never reads as an unconditional claim (MED-1).
fn deep_proposition(preconditions: &[DeepExpr], body: &DeepExpr) -> DeepExpr {
    if preconditions.is_empty() {
        return body.clone();
    }
    deep_node(
        "app",
        vec![
            deep_var("implies"),
            combine_deep_preconditions(preconditions),
            body.clone(),
        ],
    )
}

// ===========================================================================
// Deep metadata helpers
// ===========================================================================

/// Observe either stamped `Node` or transitional canonical `List` through one
/// consumer view. This does not normalize, clone, or reconstruct the tree: new
/// file ingress stays in the role-typed representation while legacy callers
/// remain readable until the carrier is deleted atomically.
fn deep_node_parts(expr: &DeepExpr) -> Option<(DeepTag, &MetaMap, &[DeepExpr])> {
    match expr {
        DeepExpr::Node(node, _) => Some((node.tag(), node.meta(), node.children_slice())),
        DeepExpr::List(list, _) => {
            let tag = list_tag_from_list(list)?;
            let meta = list.elements.get(1).and_then(meta_map)?;
            let children = list.elements.get(2..)?;
            Some((tag, meta, children))
        }
        _ => None,
    }
}

fn meta_map(expr: &DeepExpr) -> Option<&MetaMap> {
    match expr {
        DeepExpr::Map(map, _) => Some(map),
        _ => None,
    }
}

fn deep_meta_value<'a>(meta: &'a MetaMap, key: &str) -> Option<&'a DeepExpr> {
    meta.entries
        .iter()
        .find_map(|(entry_key, value)| (entry_key == key).then_some(value))
}

fn deep_int_meta(meta: &MetaMap, key: &str) -> Option<usize> {
    match deep_meta_value(meta, key).and_then(deep_int_value) {
        Some(value) if value >= 0 => Some(value as usize),
        _ => None,
    }
}

fn deep_int_value(expr: &DeepExpr) -> Option<i64> {
    match expr {
        DeepExpr::Atom(DeepAtom::Int(value), _) => Some(*value),
        _ => match deep_node_parts(expr) {
            Some((DeepTag::Lit, _, [DeepExpr::Atom(DeepAtom::Int(value), _)])) => Some(*value),
            _ => None,
        },
    }
}

fn deep_property_params(meta: &MetaMap) -> Option<Vec<Param>> {
    let (tag, _, children) = deep_node_parts(deep_meta_value(meta, "property_quantifiers")?)?;
    if tag != DeepTag::Params {
        return None;
    }
    let mut params = Vec::new();
    for child in children {
        if let Some(param) = deep_param(child) {
            params.push(param);
        }
    }
    Some(params)
}

fn deep_property_preconditions(meta: &MetaMap) -> Option<Vec<DeepExpr>> {
    let (tag, _, children) = deep_node_parts(deep_meta_value(meta, "property_preconditions")?)?;
    if tag != DeepTag::Tuple {
        return None;
    }
    Some(children.to_vec())
}

fn type_expr_from_deep(expr: &DeepExpr) -> Option<TypeExpr> {
    let (tag, _, children) = deep_node_parts(expr)?;
    let span = expr.span();
    match tag {
        DeepTag::TPrim => children
            .first()
            .and_then(symbol_text)
            .map(|name| TypeExpr::Named(name.to_string(), span)),
        DeepTag::TTensor => {
            let precision = children.last().and_then(|expr| {
                let (tag, _, prim_children) = deep_node_parts(expr)?;
                (tag == DeepTag::TPrim)
                    .then(|| prim_children.first().and_then(symbol_text))
                    .flatten()
            })?;
            let dims = children
                .iter()
                .take(children.len().saturating_sub(1))
                .map(|dim| {
                    let (tag, _, dim_children) = deep_node_parts(dim)?;
                    match tag {
                        DeepTag::DLit => dim_children.first().and_then(|value| match value {
                            DeepExpr::Atom(DeepAtom::Int(value), _) => {
                                Some(TypeExpr::Named(value.to_string(), dim.span()))
                            }
                            _ => None,
                        }),
                        DeepTag::DName => dim_children
                            .first()
                            .and_then(symbol_text)
                            .map(|name| TypeExpr::Named(name.to_string(), dim.span())),
                        _ => None,
                    }
                })
                .collect::<Option<Vec<_>>>()?;
            Some(TypeExpr::Tensor(dims, precision.to_string(), span))
        }
        _ => None,
    }
}

fn deep_fn_body(expr: &DeepExpr) -> Option<&DeepExpr> {
    let (tag, _, children) = deep_node_parts(expr)?;
    if tag != DeepTag::Fn {
        return None;
    }
    children.get(1)
}

fn deep_fn_params(expr: &DeepExpr) -> Option<Vec<Param>> {
    let (tag, _, children) = deep_node_parts(expr)?;
    if tag != DeepTag::Fn {
        return None;
    }
    let (params_tag, _, params) = deep_node_parts(children.first()?)?;
    if params_tag != DeepTag::Params {
        return None;
    }
    let mut out = Vec::new();
    for child in params {
        out.push(deep_param(child)?);
    }
    Some(out)
}

fn deep_param(expr: &DeepExpr) -> Option<Param> {
    if let DeepExpr::Atom(DeepAtom::Name(name), span) = expr {
        return Some(Param {
            name: name.clone(),
            ty: None,
            span: *span,
        });
    }
    let (elements, span) = match expr {
        DeepExpr::BareList(elements, span) => (elements.as_slice(), *span),
        DeepExpr::List(list, span) => (list.elements.as_slice(), *span),
        _ => return None,
    };
    let name = elements.first().and_then(symbol_text)?;
    let ty = elements
        .get(1)
        .and_then(meta_map)
        .and_then(|meta| deep_meta_value(meta, "type"))
        .and_then(type_expr_from_deep);
    Some(Param {
        name: name.to_string(),
        ty,
        span,
    })
}

fn params_match(left: &[Param], right: &[Param]) -> bool {
    left.len() == right.len()
        && left.iter().zip(right).all(|(left, right)| {
            left.name == right.name
                && left.ty.as_ref().and_then(surf_type_to_deep)
                    == right.ty.as_ref().and_then(surf_type_to_deep)
        })
}

fn surf_type_to_deep(ty: &TypeExpr) -> Option<DeepExpr> {
    match ty {
        TypeExpr::Named(name, _) => Some(deep_node("t-prim", vec![deep_symbol(name)])),
        TypeExpr::Tensor(dims, precision, _) => {
            let mut children = dims
                .iter()
                .map(|dim| match dim {
                    TypeExpr::Named(value, _) => value
                        .parse::<i64>()
                        .ok()
                        .map(|dim| deep_node("d-lit", vec![deep_int(dim)]))
                        .or_else(|| Some(deep_node("d-name", vec![deep_symbol(value)]))),
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            children.push(deep_node("t-prim", vec![deep_symbol(precision)]));
            Some(deep_node("t-tensor", children))
        }
        _ => None,
    }
}

// ===========================================================================
// Deep builders + small helpers
// ===========================================================================

fn deep_span() -> chelis_deep::Span {
    chelis_deep::Span::new(0, 0)
}
fn deep_symbol(value: &str) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Name(value.to_string()), deep_span())
}
fn deep_int(value: i64) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Int(value), deep_span())
}
fn deep_float(value: f64) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Float(value), deep_span())
}
fn deep_bool(value: bool) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Bool(value), deep_span())
}
fn deep_string(value: &str) -> DeepExpr {
    DeepExpr::Atom(DeepAtom::Str(value.to_string()), deep_span())
}
fn deep_map(entries: Vec<(String, DeepExpr)>) -> DeepExpr {
    DeepExpr::Map(MetaMap { entries }, deep_span())
}
fn deep_list(elements: Vec<DeepExpr>) -> DeepExpr {
    DeepExpr::List(DeepList { elements }, deep_span())
}
fn deep_node(tag: &str, children: Vec<DeepExpr>) -> DeepExpr {
    let mut elements = vec![deep_symbol(tag), deep_map(Vec::new())];
    elements.extend(children);
    deep_list(elements)
}
fn deep_node_meta(
    tag: &str,
    entries: Vec<(String, DeepExpr)>,
    children: Vec<DeepExpr>,
) -> DeepExpr {
    let mut elements = vec![deep_symbol(tag), deep_map(entries)];
    elements.extend(children);
    deep_list(elements)
}
fn deep_var(name: &str) -> DeepExpr {
    deep_node("var", vec![deep_symbol(name)])
}
fn deep_lit(value: DeepExpr, ty_name: &str) -> DeepExpr {
    deep_node_meta(
        "lit",
        vec![(
            "type".to_string(),
            deep_node("t-prim", vec![deep_symbol(ty_name)]),
        )],
        vec![value],
    )
}

fn list_tag_from_list(list: &DeepList) -> Option<DeepTag> {
    list.tag()
}
fn symbol_text(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(DeepAtom::Name(value), _) => Some(value),
        _ => None,
    }
}
fn string_value(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(DeepAtom::Str(value), _) => Some(value),
        _ => None,
    }
}

fn matches_filter(name: &str, only: Option<&str>) -> bool {
    let Some(pattern) = only else {
        return true;
    };
    if pattern == name {
        return true;
    }
    if let Some(prefix) = pattern.strip_suffix('*') {
        return name.starts_with(prefix);
    }
    name.contains(pattern)
}

#[derive(Debug, Clone)]
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
        // Convex interpolation avoids overflowing `max - min` for a valid
        // finite interval such as [-1e308, 1e308]. Each weighted endpoint is
        // finite and their mathematical sum remains inside [min, max].
        min * (1.0 - unit) + max * unit
    }
}

// =========================================================================
// Property dependency extraction (chelis#490)
// =========================================================================

/// Extract dependency edges from every `@property` in `decls`: for each
/// property, collect the set of module-level function/value names its body
/// references (excluding the property's own parameter names and built-in
/// operators).
pub fn property_dependency_edges(
    source: &str,
) -> Result<Vec<crate::artifact::PropertyDependency>, String> {
    let parsed = chelis_surf::parser::parse_str(source).map_err(|e| format!("parse: {e}"))?;
    let flat = flatten_module_decls(&parsed);
    let module_names: std::collections::BTreeSet<String> = flat
        .iter()
        .filter_map(|d| match d {
            Decl::FunDef { name, .. } | Decl::LetDef { name, .. } => Some(name.clone()),
            _ => None,
        })
        .collect();
    let properties = collect_surf_properties(&flat, None);
    let mut edges = Vec::new();
    for property in &properties {
        let param_names: std::collections::BTreeSet<&str> =
            property.params.iter().map(|p| p.name.as_str()).collect();
        let mut refs = std::collections::BTreeSet::new();
        collect_expr_refs(&property.body, &param_names, &module_names, &mut refs);
        for pre in &property.preconditions {
            collect_expr_refs(pre, &param_names, &module_names, &mut refs);
        }
        edges.push(crate::artifact::PropertyDependency {
            property: property.name.clone(),
            references: refs.into_iter().collect(),
        });
    }
    Ok(edges)
}

fn collect_expr_refs(
    expr: &Expr,
    params: &std::collections::BTreeSet<&str>,
    module_names: &std::collections::BTreeSet<String>,
    out: &mut std::collections::BTreeSet<String>,
) {
    match expr {
        Expr::Var(name, _) => {
            if !params.contains(name.as_str()) && module_names.contains(name) {
                out.insert(name.clone());
            }
        }
        Expr::Apply(callee, args, _) => {
            if let Expr::Var(name, _) = callee.as_ref() {
                if !params.contains(name.as_str()) && module_names.contains(name) {
                    out.insert(name.clone());
                }
            } else {
                collect_expr_refs(callee, params, module_names, out);
            }
            for arg in args {
                collect_expr_refs(arg, params, module_names, out);
            }
        }
        Expr::Binary(_, l, r, _) => {
            collect_expr_refs(l, params, module_names, out);
            collect_expr_refs(r, params, module_names, out);
        }
        Expr::Unary(_, e, _) => collect_expr_refs(e, params, module_names, out),
        Expr::If(c, t, f, _) => {
            collect_expr_refs(c, params, module_names, out);
            collect_expr_refs(t, params, module_names, out);
            collect_expr_refs(f, params, module_names, out);
        }
        Expr::Pipe(head, stages, _) => {
            collect_expr_refs(head, params, module_names, out);
            for s in stages {
                collect_expr_refs(s, params, module_names, out);
            }
        }
        Expr::Block(bindings, body, _) => {
            for b in bindings {
                collect_expr_refs(&b.value, params, module_names, out);
            }
            collect_expr_refs(body, params, module_names, out);
        }
        Expr::Lambda(_, body, _) => collect_expr_refs(body, params, module_names, out),
        Expr::Tuple(elems, _) | Expr::List(elems, _) => {
            for e in elems {
                collect_expr_refs(e, params, module_names, out);
            }
        }
        Expr::Access(e, _, _) | Expr::TupleGet(e, _, _) | Expr::Annotate(e, _, _) => {
            collect_expr_refs(e, params, module_names, out);
        }
        _ => {}
    }
}

#[cfg(test)]
mod tests;
