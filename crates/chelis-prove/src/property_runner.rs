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

use std::{cell::RefCell, collections::BTreeMap};

use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr, List as DeepList, MetaMap};
use chelis_surf::ast::{
    BinOp, Decl, Expr, LetBinding, LetPattern, Literal, Param, PropertyOption, TypeExpr,
};

mod smt_lower;
use smt_lower::{ContractAbstraction, InlineCtx, surf_expr_to_smt};

mod injection;
use crate::composition::{
    AssumptionDischarge, AssumptionRecord, CompositeVerdict, DischargeMethod, FUZZ_TOLERANCE,
    NonVacuityRecord, NonVacuityStatus, rollup_composite,
};
use crate::contracts::{NORMAL_CDF_RANGE, NORMAL_CDF_REFLECTION, standard_contract_registry};

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
    Fuzz,
    /// No tier ran (a sampling/declaration error).
    None,
}

impl PropertyTier {
    pub fn as_str(self) -> &'static str {
        match self {
            PropertyTier::Smt => "smt",
            PropertyTier::Fuzz => "fuzz",
            PropertyTier::None => "none",
        }
    }
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
    /// Weakest-link verdict after composing the proof and its assumptions.
    pub composite_verdict: CompositeVerdict,
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
        let base = base_verdict(&status, proof_tier, samples, counterexample.as_ref());
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
            composite_verdict,
        }
    }

    pub(super) fn with_shrink_steps(mut self, shrink_steps: usize) -> Self {
        self.shrink_steps = shrink_steps;
        self
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
            CompositeVerdict::Failed | CompositeVerdict::Invalid | CompositeVerdict::Unsupported
        ) {
            return false;
        }
        self.status == PropertyStatus::Passed
            && (self.proof_tier == PropertyTier::Smt || self.samples > 0)
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
            CompositeVerdict::Failed => return "failed",
            CompositeVerdict::Invalid | CompositeVerdict::Unsupported => return "unsupported",
            CompositeVerdict::Proven
            | CompositeVerdict::ProvenModuloFuzzValidatedContract
            | CompositeVerdict::ProvenModuloAssertedAxiom => {}
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
) -> CompositeVerdict {
    match status {
        PropertyStatus::Passed => match proof_tier {
            PropertyTier::Smt => CompositeVerdict::Proven,
            PropertyTier::Fuzz if samples > 0 => {
                CompositeVerdict::ProvenModuloFuzzValidatedContract
            }
            _ => CompositeVerdict::Unsupported,
        },
        PropertyStatus::Failed => {
            if counterexample.is_some() {
                CompositeVerdict::Failed
            } else {
                CompositeVerdict::Unsupported
            }
        }
        PropertyStatus::Unsupported | PropertyStatus::Error => CompositeVerdict::Unsupported,
    }
}

/// Options for a property run, mirroring the prove surface.
#[derive(Debug, Clone)]
pub struct PropertyRunOptions {
    pub seed: u64,
    pub samples: usize,
    pub smt_timeout_ms: u64,
    /// `"auto"` (Tier B then C), `"smt-only"`, `"fuzz-only"`.
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
/// for standard contracts. The CLI Reef path passes the bundled chelis-std
/// declarations here; unlinked source paths pass an empty slice, so a user
/// cannot obtain std contract assumptions by spelling a linker-shaped name.
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
        out.push(prove_surf_property(
            all_decls,
            module_decls,
            trusted_contract_decls,
            property,
            options,
        ));
    }
    Ok(PropertyRunResult::Ran(out))
}

/// Discover and run every user `@property` declaration in Deep `.dp`
/// source. Returns `Err` if the source does not parse.
pub fn run_deep_source_properties(
    source: &str,
    options: &PropertyRunOptions,
) -> Result<PropertyRunResult, String> {
    let exprs = chelis_deep::parser::parse_str(source).map_err(|e| format!("parse: {e}"))?;
    let properties = discover_deep_properties(&exprs, options.only.as_deref())?;
    let mut out = Vec::new();
    for property in &properties {
        out.push(prove_deep_property(&exprs, property, options));
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

fn contract_assumptions(property: &Property) -> Result<Vec<AssumptionRecord>, String> {
    let contracts = expanded_contracts(property);
    if contracts.is_empty() {
        return Ok(Vec::new());
    }
    let registry = standard_contract_registry();
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
    let postcondition = surf_expr_to_smt(
        &property.body,
        &InlineCtx {
            decls,
            depth: 0,
            max_depth: 3,
            call_stack: vec![],
            contracts: Some(&contract_abstraction),
        },
    )?;
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
        return None;
    }
    match crate::solve_property(&smt_prop, options.smt_timeout_ms) {
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
            Some(PropertyOutcome::new(
                property.name.clone(),
                status,
                PropertyTier::Smt,
                0,
                seed,
                None,
                reason,
                false,
                assumptions,
            ))
        }
        crate::tier_b::TierBResult::Disproved(model) => Some(PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Failed,
            PropertyTier::Smt,
            0,
            seed,
            Some(model),
            None,
            false,
            Vec::new(),
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
                    Some(format!("smt lowering error: {reason}")),
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
        vec![AssumptionRecord::new(
            format!("preconditions:{property_name}"),
            Some(discharge),
            Some(non_vacuity),
        )]
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

    let mut accepted = 0usize;
    let mut attempts = 0usize;
    while accepted < samples_needed && attempts < max_attempts {
        attempts += 1;
        let sample = match sample_property(property, &mut rng) {
            Ok(sample) => sample,
            Err(reason) => return unsupported(&property.name, seed, reason),
        };
        if !property.preconditions.is_empty() {
            match eval_surf_sample(decls, property, &sample, true) {
                Ok(false) => continue,
                Ok(true) => {}
                Err(err) => return error(&property.name, seed, err),
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
                .with_shrink_steps(shrink_steps);
            }
            Err(err) => return error(&property.name, seed, err),
        }
    }

    if accepted < samples_needed {
        return error(
            &property.name,
            seed,
            format!(
                "generator exhausted after {attempts} attempts before collecting {samples_needed} valid samples"
            ),
        );
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
        fuzz_precondition_assumptions(&property.name, property.preconditions.len(), accepted, seed),
    )
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
    seed: u64,
) -> Vec<AssumptionRecord> {
    if precondition_count == 0 {
        return Vec::new();
    }
    vec![AssumptionRecord::new(
        format!("preconditions:{property_name}"),
        Some(AssumptionDischarge::new(
            DischargeMethod::Fuzz,
            serde_json::json!({
                "status": "validated",
                "property": property_name,
                "samples": samples,
                "seed": seed,
                "tolerance": FUZZ_TOLERANCE,
            }),
        )),
        Some(NonVacuityRecord::established(serde_json::json!({
            "method": "fuzz",
            "result": "sat",
            "accepted_samples": samples,
            "seed": seed,
        }))),
    )]
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
            let lit = Expr::Lit(Literal::Float(value), sp);
            if type_name == "f64" {
                Ok(scalar_sample(
                    name,
                    cast_expr(lit, "f64"),
                    deep_lit(deep_float(value), "f64"),
                    serde_json::json!(value),
                ))
            } else {
                Ok(scalar_sample(
                    name,
                    lit,
                    deep_lit(deep_float(value), "f32"),
                    serde_json::json!(value),
                ))
            }
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
                Ok(value.data[0] != 0.0)
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
    let DeepExpr::List(list, _) = expr else {
        return Ok(());
    };
    if list_tag(expr) == Some("def")
        && let Some(name) = list.elements.get(2).and_then(symbol_text)
        && let Some(meta) = list.elements.get(1).and_then(meta_map)
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
                let fn_expr = list
                    .elements
                    .get(3)
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
    for child in &list.elements {
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
    // Honor the --tier contract on the deep path (F7). A Deep property body
    // is in Deep AST and has no Surf->SMT lowering path (the Tier B
    // surf_expr_to_smt lowering the surf path uses takes a Surf body), so
    // `smt-only` is Unsupported rather than a silent fuzz run; `fuzz-only`
    // and `auto` run the Tier C fuzz loop below.
    if options.tier == "smt-only" {
        return PropertyOutcome::new(
            property.name.clone(),
            PropertyStatus::Unsupported,
            PropertyTier::Smt,
            0,
            seed,
            None,
            Some("deep property has no Tier B (SMT) lowering path (smt-only)".to_string()),
            false,
            Vec::new(),
        );
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

    let mut accepted = 0usize;
    let mut attempts = 0usize;
    while accepted < samples_needed && attempts < max_attempts {
        attempts += 1;
        let sample = match sample_deep_property(property, &mut rng) {
            Ok(sample) => sample,
            Err(reason) => return unsupported(&property.name, seed, reason),
        };
        if !property.preconditions.is_empty() {
            match eval_deep_sample(exprs, property, &sample, true) {
                Ok(false) => continue,
                Ok(true) => {}
                Err(err) => return error(&property.name, seed, err),
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
                .with_shrink_steps(shrink_steps);
            }
            Err(err) => return error(&property.name, seed, err),
        }
    }

    if accepted < samples_needed {
        return error(
            &property.name,
            seed,
            format!(
                "generator exhausted after {attempts} attempts before collecting {samples_needed} valid samples"
            ),
        );
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
        fuzz_precondition_assumptions(&property.name, property.preconditions.len(), accepted, seed),
    )
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

// ===========================================================================
// Deep metadata helpers
// ===========================================================================

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
        DeepExpr::List(list, _) if list_tag_from_list(list) == Some("lit") => {
            match list.elements.get(2) {
                Some(DeepExpr::Atom(DeepAtom::Int(value), _)) => Some(*value),
                _ => None,
            }
        }
        _ => None,
    }
}

fn deep_property_params(meta: &MetaMap) -> Option<Vec<Param>> {
    let DeepExpr::List(list, _) = deep_meta_value(meta, "property_quantifiers")? else {
        return None;
    };
    if list_tag_from_list(list) != Some("params") {
        return None;
    }
    let mut params = Vec::new();
    for child in list.elements.iter().skip(2) {
        let DeepExpr::List(param_list, span) = child else {
            continue;
        };
        let Some(name) = param_list.elements.first().and_then(symbol_text) else {
            continue;
        };
        let ty = param_list
            .elements
            .get(1)
            .and_then(meta_map)
            .and_then(|meta| deep_meta_value(meta, "type"))
            .and_then(type_expr_from_deep);
        params.push(Param {
            name: name.to_string(),
            ty,
            span: *span,
        });
    }
    Some(params)
}

fn deep_property_preconditions(meta: &MetaMap) -> Option<Vec<DeepExpr>> {
    let DeepExpr::List(list, _) = deep_meta_value(meta, "property_preconditions")? else {
        return None;
    };
    if list_tag_from_list(list) != Some("tuple") {
        return None;
    }
    Some(list.elements.iter().skip(2).cloned().collect())
}

fn type_expr_from_deep(expr: &DeepExpr) -> Option<TypeExpr> {
    let DeepExpr::List(list, span) = expr else {
        return None;
    };
    match list_tag_from_list(list)? {
        "t-prim" => list
            .elements
            .get(2)
            .and_then(symbol_text)
            .map(|name| TypeExpr::Named(name.to_string(), *span)),
        "t-tensor" => {
            let children = list.elements.iter().skip(2).collect::<Vec<_>>();
            let precision = children.last().and_then(|expr| {
                let DeepExpr::List(prim, _) = expr else {
                    return None;
                };
                (list_tag_from_list(prim) == Some("t-prim"))
                    .then(|| prim.elements.get(2).and_then(symbol_text))
                    .flatten()
            })?;
            let dims = children
                .iter()
                .take(children.len().saturating_sub(1))
                .map(|dim| match dim {
                    DeepExpr::List(dim_list, dim_span)
                        if list_tag_from_list(dim_list) == Some("d-lit") =>
                    {
                        dim_list.elements.get(2).and_then(|value| match value {
                            DeepExpr::Atom(DeepAtom::Int(value), _) => {
                                Some(TypeExpr::Named(value.to_string(), *dim_span))
                            }
                            _ => None,
                        })
                    }
                    DeepExpr::List(dim_list, dim_span)
                        if list_tag_from_list(dim_list) == Some("d-name") =>
                    {
                        dim_list
                            .elements
                            .get(2)
                            .and_then(symbol_text)
                            .map(|name| TypeExpr::Named(name.to_string(), *dim_span))
                    }
                    _ => None,
                })
                .collect::<Option<Vec<_>>>()?;
            Some(TypeExpr::Tensor(dims, precision.to_string(), *span))
        }
        _ => None,
    }
}

fn deep_fn_body(expr: &DeepExpr) -> Option<&DeepExpr> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    if list_tag_from_list(list) != Some("fn") {
        return None;
    }
    list.elements.get(3)
}

fn deep_fn_params(expr: &DeepExpr) -> Option<Vec<Param>> {
    let DeepExpr::List(list, _) = expr else {
        return None;
    };
    if list_tag_from_list(list) != Some("fn") {
        return None;
    }
    let DeepExpr::List(params, _) = list.elements.get(2)? else {
        return None;
    };
    if list_tag_from_list(params) != Some("params") {
        return None;
    }
    let mut out = Vec::new();
    for child in params.elements.iter().skip(2) {
        let DeepExpr::List(param_list, span) = child else {
            return None;
        };
        let name = param_list.elements.first().and_then(symbol_text)?;
        let ty = param_list
            .elements
            .get(1)
            .and_then(meta_map)
            .and_then(|meta| deep_meta_value(meta, "type"))
            .and_then(type_expr_from_deep);
        out.push(Param {
            name: name.to_string(),
            ty,
            span: *span,
        });
    }
    Some(out)
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
    DeepExpr::Atom(DeepAtom::Symbol(value.to_string()), deep_span())
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

fn list_tag(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::List(list, _) => list_tag_from_list(list),
        _ => None,
    }
}
fn list_tag_from_list(list: &DeepList) -> Option<&str> {
    list.elements.first().and_then(symbol_text)
}
fn symbol_text(expr: &DeepExpr) -> Option<&str> {
    match expr {
        DeepExpr::Atom(DeepAtom::Symbol(value), _) => Some(value),
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
        min + (max - min) * unit
    }
}

#[cfg(test)]
mod tests;
