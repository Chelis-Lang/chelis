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

use std::collections::BTreeMap;

use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind, TensorValue};
use chelis_deep::ast::{Atom as DeepAtom, Expr as DeepExpr, List as DeepList, MetaMap};
use chelis_surf::ast::{
    BinOp, Decl, Expr, LetBinding, LetPattern, Literal, Param, PropertyOption, TypeExpr,
};

mod smt_lower;
use smt_lower::{InlineCtx, surf_expr_to_smt};

mod injection;

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
    pub reason: Option<String>,
    /// `true` when the property was verified through the assumption-injection
    /// path (an invariant-carrying opaque binder). Only affects rendering.
    pub injected: bool,
}

impl PropertyOutcome {
    /// Whether this outcome is a genuine pass. A pass is `Passed` with at
    /// least one sample (or an SMT proof, which carries `samples == 0` but
    /// `proof_tier == Smt`). A `Passed` with zero fuzz samples is NOT a
    /// genuine pass -- it is the vacuous/timeout sentinel and the fold must
    /// treat it as not-ok (U4).
    pub fn is_pass(&self) -> bool {
        self.status == PropertyStatus::Passed
            && (self.proof_tier == PropertyTier::Smt || self.samples > 0)
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
    let properties = collect_surf_properties(&flat, options.only.as_deref());
    let mut out = Vec::new();
    for property in &properties {
        out.push(prove_surf_property(&flat, &parsed, property, options));
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

// ===========================================================================
// Surf property running (Tier B -> Tier C, with injection)
// ===========================================================================

fn prove_surf_property(
    decls: &[Decl],
    module_decls: &[Decl],
    property: &Property,
    options: &PropertyRunOptions,
) -> PropertyOutcome {
    // Assumption injection (RFC D-INJECT): a property with an
    // invariant-carrying opaque binder is verified ONLY over
    // invariant-satisfying binder values; the injection path owns it.
    if injection::property_has_opaque_invariant_binder(module_decls, &property.params) {
        return injection::prove_with_injection(
            module_decls,
            &property.name,
            &property.params,
            &property.preconditions,
            &property.body,
            options,
        );
    }

    let seed = options.effective_seed(property.seed);

    // Tier B: attempt SMT proof when --tier auto / smt-only.
    if options.tier == "auto" || options.tier == "smt-only" {
        if let Some(outcome) = try_surf_tier_b(decls, property, options, seed) {
            return outcome;
        }
        if options.tier == "smt-only" {
            // smt-only: a property that does not lower to Tier B is
            // unsupported (no fuzz fallback). This matches the obligation
            // engine's smt-only handling and the CLI's exit semantics.
            return PropertyOutcome {
                name: property.name.clone(),
                status: PropertyStatus::Unsupported,
                proof_tier: PropertyTier::Smt,
                samples: 0,
                seed,
                counterexample: None,
                reason: Some("property does not lower to Tier B (smt-only)".to_string()),
                injected: false,
            };
        }
    }

    // Tier C: fuzz.
    prove_surf_property_fuzz(decls, property, options, seed)
}

/// Try Tier B (SMT) for a surf property. Returns `Some(outcome)` for a
/// determinate SMT verdict (Proved => Passed, Disproved => Failed), or
/// `None` to fall through to Tier C (the property did not lower, or the
/// solver timed out / errored).
fn try_surf_tier_b(
    decls: &[Decl],
    property: &Property,
    options: &PropertyRunOptions,
    seed: u64,
) -> Option<PropertyOutcome> {
    let postcondition = surf_expr_to_smt(
        &property.body,
        &InlineCtx {
            decls,
            depth: 0,
            max_depth: 3,
            call_stack: vec![],
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
                },
            )
        })
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
        return None;
    }
    match crate::solve_property(&smt_prop, options.smt_timeout_ms) {
        crate::tier_b::TierBResult::Proved => Some(PropertyOutcome {
            name: property.name.clone(),
            status: PropertyStatus::Passed,
            proof_tier: PropertyTier::Smt,
            samples: 0,
            seed,
            counterexample: None,
            reason: None,
            injected: false,
        }),
        crate::tier_b::TierBResult::Disproved(_model) => Some(PropertyOutcome {
            name: property.name.clone(),
            status: PropertyStatus::Failed,
            proof_tier: PropertyTier::Smt,
            samples: 0,
            seed,
            counterexample: None,
            reason: Some("smt counterexample".to_string()),
            injected: false,
        }),
        // Timeout / Unknown / Error: fall through to Tier C (auto) or be
        // handled as unsupported by the caller (smt-only).
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
                return PropertyOutcome {
                    name: property.name.clone(),
                    status: PropertyStatus::Failed,
                    proof_tier: PropertyTier::Fuzz,
                    samples: accepted,
                    seed,
                    counterexample: Some(counterexample_json(&sample)),
                    reason: None,
                    injected: false,
                };
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

    PropertyOutcome {
        name: property.name.clone(),
        status: PropertyStatus::Passed,
        proof_tier: PropertyTier::Fuzz,
        samples: accepted,
        seed,
        counterexample: None,
        reason: None,
        injected: false,
    }
}

fn unsupported(name: &str, seed: u64, reason: String) -> PropertyOutcome {
    PropertyOutcome {
        name: name.to_string(),
        status: PropertyStatus::Unsupported,
        proof_tier: PropertyTier::None,
        samples: 0,
        seed,
        counterexample: None,
        reason: Some(reason),
        injected: false,
    }
}

fn error(name: &str, seed: u64, reason: String) -> PropertyOutcome {
    PropertyOutcome {
        name: name.to_string(),
        status: PropertyStatus::Error,
        proof_tier: PropertyTier::None,
        samples: 0,
        seed,
        counterexample: None,
        reason: Some(reason),
        injected: false,
    }
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
        TypeExpr::Named(name, _)
            if matches!(
                name.as_str(),
                "bool" | "int8" | "int16" | "int32" | "int64" | "f32" | "f64" | "string"
            ) =>
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
        TypeExpr::Named(type_name, _)
            if matches!(type_name.as_str(), "int8" | "int16" | "int32" | "int64") =>
        {
            let value = rng.next_i64(-1000, 1000);
            let lit = Expr::Lit(Literal::Int(value), sp);
            if type_name == "int32" {
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
        && property_is_user(meta)
    {
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
                body: deep_fn_body(fn_expr)
                    .cloned()
                    .ok_or_else(|| format!("property `{name}` def body must be a callable `fn`"))?,
                samples: deep_int_meta(meta, "property_samples"),
                seed: deep_int_meta(meta, "property_seed").map(|value| value as u64),
            });
        }
    }
    for child in &list.elements {
        discover_deep_properties_expr(child, only, out)?;
    }
    Ok(())
}

/// Whether a def's metadata marks it a USER property (the
/// `chelis_role: "property"` with `property_source_kind: "user"`). The
/// bridge `c-earchin` properties are NOT user properties and are not run
/// here (they go through the CLI's bridge path).
fn property_is_user(meta: &MetaMap) -> bool {
    let role_property = meta
        .entries
        .iter()
        .any(|(k, v)| k == "chelis_role" && string_value(v) == Some("property"));
    if !role_property {
        return false;
    }
    matches!(
        deep_meta_value(meta, "property_source_kind").and_then(string_value),
        Some("user")
    )
}

fn prove_deep_property(
    exprs: &[DeepExpr],
    property: &DeepProperty,
    options: &PropertyRunOptions,
) -> PropertyOutcome {
    let seed = options.effective_seed(property.seed);
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
                return PropertyOutcome {
                    name: property.name.clone(),
                    status: PropertyStatus::Failed,
                    proof_tier: PropertyTier::Fuzz,
                    samples: accepted,
                    seed,
                    counterexample: Some(counterexample_json(&sample)),
                    reason: None,
                    injected: false,
                };
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

    PropertyOutcome {
        name: property.name.clone(),
        status: PropertyStatus::Passed,
        proof_tier: PropertyTier::Fuzz,
        samples: accepted,
        seed,
        counterexample: None,
        reason: None,
        injected: false,
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
