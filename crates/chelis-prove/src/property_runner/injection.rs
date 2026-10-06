//! Property verification for binders with opaque invariants.
//!
//! When a `@property` binder has an opaque type with an invariant, the
//! generator supplies values that satisfy that invariant. This lets the
//! property run over values admitted by the opaque type. Other binders use
//! their usual generation rules.
//!
//! The runner desugars the module and collects its opaque invariants. For
//! each accepted sample, it builds a Deep probe that binds the generated
//! opaque records and ordinary scalar or tensor values, then evaluates the
//! property body. It generates opaque values inside their defining module.
//!
//! This module is part of the optional `chelis-prove` dependency.

use chelis_deep::Span;
use chelis_deep::annotations::{MetadataKey as K, MetadataValue as M, TypeSyntax};
use chelis_deep::ast::{Atom, Expr, Metadata};
use chelis_deep::{DeepTag, ExprCarrier};
use chelis_surf::ast::{Decl, Param, TypeExpr};
use chelis_types::types::Prim;
use chelis_types::{ScalarValue, scalar_from_f64, scalar_from_i64};

use crate::composition::{
    AssumptionDischarge, AssumptionRecord, DischargeMethod, DischargeTier, FUZZ_TOLERANCE,
    NonVacuityRecord,
};
use crate::smt_names::fresh_root_name;

use super::{PropertyOutcome, PropertyRunOptions, PropertyStatus, PropertyTier};

/// Does this property have at least one binder whose type is an
/// invariant-carrying opaque type? If so, the injection path owns it.
pub(super) fn property_has_opaque_invariant_binder(
    decls: &[Decl],
    property_path: &[usize],
    params: &[Param],
) -> Result<bool, String> {
    validate_property_path(decls, property_path)?;
    let exprs = chelis_surf::desugar::desugar_program(decls).map_err(|error| error.to_string())?;
    let invariants = crate::opaque::collect_opaque_invariants(&exprs);
    Ok(params.iter().any(|p| {
        matches!(&p.ty, Some(TypeExpr::Named(name, _))
            if invariants.iter().any(|inv| &inv.type_name == name))
    }))
}

fn validate_property_path(decls: &[Decl], property_path: &[usize]) -> Result<(), String> {
    let Some((&selected_index, nested_path)) = property_path.split_first() else {
        return Err("property declaration path is empty".to_string());
    };
    let Some(decl) = decls.get(selected_index) else {
        return Err(format!(
            "property declaration path index {selected_index} is out of bounds"
        ));
    };
    if nested_path.is_empty() {
        if matches!(decl, Decl::Property { .. }) {
            return Ok(());
        }
        return Err("property declaration path does not select a property".to_string());
    }

    let Decl::Module { decls, .. } = decl else {
        return Err("property declaration path descends through a non-module".to_string());
    };
    validate_property_path(decls, nested_path)
}

/// Run a user property that has an opaque binder with an invariant.
/// Returns the property status.
pub(super) fn prove_with_injection(
    decls: &[Decl],
    property_name: &str,
    params: &[Param],
    preconditions: &[chelis_surf::ast::Expr],
    body: &chelis_surf::ast::Expr,
    options: &PropertyRunOptions,
) -> PropertyOutcome {
    let seed = options.injection_seed();
    let samples_needed = options.samples;

    let exprs = match chelis_surf::desugar::desugar_program(decls) {
        Ok(exprs) => exprs,
        Err(error) => {
            return outcome(
                property_name,
                PropertyStatus::Error,
                0,
                seed,
                None,
                Some(format!("injection desugar failed: {error}")),
                Vec::new(),
            );
        }
    };
    let invariants = crate::opaque::collect_opaque_invariants(&exprs);
    let consts = resolve_constants(&exprs, &invariants);
    let module_source = chelis_deep::printer::print_canonical(&exprs);

    // Classify each binder.
    let mut binders = Vec::new();
    for p in params {
        match classify_binder(p, &invariants) {
            Some(b) => binders.push(b),
            None => {
                return outcome(
                    property_name,
                    PropertyStatus::Unsupported,
                    0,
                    seed,
                    None,
                    Some(format!(
                        "binder `{}` has an unsupported type for injection",
                        p.name
                    )),
                    Vec::new(),
                );
            }
        }
    }

    // The desugared property body + preconditions (Deep).
    let bound_names = params
        .iter()
        .map(|parameter| parameter.name.clone())
        .collect::<Vec<_>>();
    let body_deep =
        match chelis_surf::desugar::desugar_expr_in_program_scope(decls, body, &bound_names) {
            Ok(body) => body,
            Err(error) => {
                return outcome(
                    property_name,
                    PropertyStatus::Error,
                    0,
                    seed,
                    None,
                    Some(format!("property body desugar failed: {error}")),
                    Vec::new(),
                );
            }
        };
    let pre_deep: Vec<Expr> = match preconditions
        .iter()
        .map(|precondition| {
            chelis_surf::desugar::desugar_expr_in_program_scope(decls, precondition, &bound_names)
        })
        .collect::<Result<_, _>>()
    {
        Ok(preconditions) => preconditions,
        Err(error) => {
            return outcome(
                property_name,
                PropertyStatus::Error,
                0,
                seed,
                None,
                Some(format!("property precondition desugar failed: {error}")),
                Vec::new(),
            );
        }
    };

    // The probe is declared in the module that defines the first opaque
    // binder's type, since the sampled binder values construct that type.
    let home_type = binders
        .iter()
        .find_map(|binder| match binder {
            Binder::Opaque { inv, .. } => Some(inv.type_name.clone()),
            _ => None,
        })
        .unwrap_or_default();

    let mut rng = crate::opaque::GenRng::new(seed);
    let mut accepted = 0usize;
    let gen_budget = samples_needed.saturating_mul(100).max(200);
    // Outer rejection-loop cap, mirroring the surf/deep fuzz loops
    // (property_runner.rs): a precondition that no invariant-valid binder can
    // satisfy must report generator exhaustion rather than spin forever.
    // `gen_budget` only bounds the inner per-binder generator, not this loop.
    let max_attempts = max_injection_attempts(samples_needed, options.max_attempts);
    let mut attempts = 0usize;

    while accepted < samples_needed && attempts < max_attempts {
        attempts += 1;
        // Sample each binder. Opaque binders are generated invariant-valid
        // (the injected assumption); on starvation, report. Each binding
        // carries its Deep value expr and a JSON repr for counterexamples.
        let mut bindings: Vec<(String, Expr, serde_json::Value)> = Vec::new();
        for b in &binders {
            match b {
                Binder::Scalar { name, prim } => {
                    let v = sample_scalar(prim, &mut rng);
                    bindings.push((name.clone(), scalar_lit(prim, v), scalar_json(v)));
                }
                Binder::Tensor {
                    name,
                    dims,
                    precision,
                } => {
                    let count: usize = dims.iter().product::<usize>().max(1);
                    let values = (0..count)
                        .map(|_| sample_scalar(precision, &mut rng))
                        .collect::<Vec<_>>();
                    bindings.push((
                        name.clone(),
                        crate::opaque::tensor_value_expr_typed(dims, precision, &values),
                        crate::opaque::scalar_values_json(&values),
                    ));
                }
                Binder::Opaque { name, inv } => {
                    let mut grng = crate::opaque::GenRng::new(rng.next_u64());
                    let producers = generation_producers(&exprs, &invariants, inv);
                    match crate::opaque::generate_binder(
                        inv,
                        &consts,
                        crate::opaque::GenModule {
                            exprs: &exprs,
                            source: &module_source,
                            runtime: options.runtime,
                        },
                        &producers,
                        &mut grng,
                        options.invariant_min_rate,
                        gen_budget,
                    ) {
                        Ok(generated) => {
                            let json = crate::opaque::generated_env_json(&generated.env);
                            bindings.push((name.clone(), generated.value_expr, json));
                        }
                        Err(crate::opaque::GenerationFailure::ProbeRejected(reason)) => {
                            return outcome_error(property_name, seed, reason);
                        }
                        Err(crate::opaque::GenerationFailure::Starved(diag)) => {
                            if options.invariant_min_rate == 0.0 {
                                return outcome(
                                    property_name,
                                    PropertyStatus::Error,
                                    0,
                                    seed,
                                    None,
                                    Some(format!(
                                        "generator exhausted for invariant binder `{}` of type `{}`",
                                        name, diag.type_name
                                    )),
                                    Vec::new(),
                                );
                            }
                            return outcome(
                                property_name,
                                PropertyStatus::Unsupported,
                                0,
                                seed,
                                None,
                                Some(diag.message()),
                                Vec::new(),
                            );
                        }
                    }
                }
            }
        }

        // Precondition filter (composed with the injected invariant by the
        // generator already restricting opaque binders).
        if !pre_deep.is_empty() {
            match eval_bool_in_module(&exprs, &home_type, &bindings, &conjoin(&pre_deep)) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(e) => return outcome_error(property_name, seed, e),
            }
        }

        accepted += 1;
        match eval_bool_in_module(&exprs, &home_type, &bindings, &body_deep) {
            Ok(true) => {}
            Ok(false) => {
                let cx = counterexample(&bindings);
                return outcome(
                    property_name,
                    PropertyStatus::Failed,
                    accepted,
                    seed,
                    Some(cx),
                    None,
                    injection_assumptions(property_name, &binders, accepted, seed),
                );
            }
            Err(e) => return outcome_error(property_name, seed, e),
        }
    }

    if accepted < samples_needed {
        return outcome_error(
            property_name,
            seed,
            exhaustion_reason(attempts, samples_needed),
        );
    }

    outcome(
        property_name,
        PropertyStatus::Passed,
        samples_needed,
        seed,
        None,
        None,
        injection_assumptions(property_name, &binders, samples_needed, seed),
    )
}

/// Outer rejection-loop attempt cap, computed exactly as the surf/deep fuzz
/// loops in `property_runner.rs`: honor an explicit `max_attempts` override,
/// otherwise derive `samples_needed * 100` (with a floor of `samples_needed`).
/// Bounding this loop is what prevents an unsatisfiable-precondition spin.
fn max_injection_attempts(samples_needed: usize, override_attempts: Option<usize>) -> usize {
    override_attempts.unwrap_or_else(|| samples_needed.saturating_mul(100).max(samples_needed))
}

/// The generator-exhaustion diagnostic reported when the rejection loop hits
/// its attempt cap before collecting `samples_needed` accepted samples. Same
/// wording as the surf/deep fuzz loops for a single exhaustion message shape.
fn exhaustion_reason(attempts: usize, samples_needed: usize) -> String {
    format!(
        "generator exhausted after {attempts} attempts before collecting {samples_needed} valid samples"
    )
}

/// Build an outcome for a property checked with an invariant-valid binder.
/// Every outcome from this path carries `injected: true`.
fn outcome(
    name: &str,
    status: PropertyStatus,
    samples: usize,
    seed: u64,
    counterexample: Option<serde_json::Value>,
    reason: Option<String>,
    assumptions: Vec<AssumptionRecord>,
) -> PropertyOutcome {
    let tier = if matches!(status, PropertyStatus::Passed | PropertyStatus::Failed) {
        PropertyTier::Fuzz
    } else {
        PropertyTier::None
    };
    PropertyOutcome::new(
        name,
        status,
        tier,
        samples,
        seed,
        counterexample,
        reason,
        true,
        assumptions,
    )
}

fn outcome_error(name: &str, seed: u64, reason: String) -> PropertyOutcome {
    outcome(
        name,
        PropertyStatus::Error,
        0,
        seed,
        None,
        Some(reason),
        Vec::new(),
    )
}

enum Binder {
    Scalar {
        name: String,
        prim: String,
    },
    Tensor {
        name: String,
        dims: Vec<usize>,
        precision: String,
    },
    Opaque {
        name: String,
        inv: crate::opaque::OpaqueInvariant,
    },
}

fn injection_assumptions(
    property_name: &str,
    binders: &[Binder],
    samples: usize,
    seed: u64,
) -> Vec<AssumptionRecord> {
    // A reef-linked program carries linker-format names; the record names
    // the source spellings so a package property's assumption matches the
    // same property's assumption as a bare file (chelis#2416).
    let property_name = chelis_types::demangle_ident(property_name);
    binders
        .iter()
        .filter_map(|binder| match binder {
            Binder::Opaque { name, inv } => Some({
                let type_name = chelis_types::demangle_ident(&inv.type_name);
                AssumptionRecord::new(
                    format!("invariant:{type_name}:binder:{name}"),
                    Some(AssumptionDischarge::new(
                        DischargeMethod::Fuzz,
                        serde_json::json!({
                            "status": "validated",
                            "property": property_name,
                            "binder": name,
                            "source_type": type_name,
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
                )
                .with_source(type_name.clone(), format!("binder:{name}"))
                // WI-8: stamp the prover-side discharge tier on the
                // binder-matched (injected) assumption. The injection path is
                // the fuzz sampler discharging the invariant of an opaque
                // binder, keyed to the binder's source identity.
                .with_discharge_tier(DischargeTier::new(
                    DischargeMethod::Fuzz.engine(),
                    DischargeMethod::Fuzz,
                    Some(format!("invariant:{type_name}:binder:{name}")),
                ))
            }),
            _ => None,
        })
        .collect()
}

fn classify_binder(p: &Param, invariants: &[crate::opaque::OpaqueInvariant]) -> Option<Binder> {
    match p.ty.as_ref()? {
        TypeExpr::Named(name, _) => {
            // A scalar binder: f32/f64/bool, or ANY signed integer width
            // recognized through the single-source `is_int_width` (review 5).
            if matches!(name.as_str(), "f32" | "f64" | "bool") || crate::opaque::is_int_width(name)
            {
                Some(Binder::Scalar {
                    name: p.name.clone(),
                    prim: name.clone(),
                })
            } else {
                invariants
                    .iter()
                    .find(|i| &i.type_name == name)
                    .map(|inv| Binder::Opaque {
                        name: p.name.clone(),
                        inv: inv.clone(),
                    })
            }
        }
        TypeExpr::Tensor(dims, precision, _)
            // A sampled key tensor would be a key literal (spec/04 §1.1), so
            // only the data element dtypes are sampled.
            if Prim::parse_name(precision).is_some_and(|prim| prim.is_data_element_dtype()) =>
        {
            let lit: Option<Vec<usize>> = dims
                .iter()
                .map(|d| match d {
                    TypeExpr::Named(v, _) => v.parse::<usize>().ok(),
                    _ => None,
                })
                .collect();
            lit.map(|dims| Binder::Tensor {
                name: p.name.clone(),
                dims,
                precision: precision.to_string(),
            })
        }
        _ => None,
    }
}

/// Evaluate a boolean Deep expr with the given binder value bindings,
/// inside the module that defines `home_type` (so constructing and
/// inspecting the binder's opaque values is legal), with the invariant
/// metadata stripped (so a `sum`-bearing invariant does not block IR lowering
/// of the bound module).
fn eval_bool_in_module(
    exprs: &[Expr],
    home_type: &str,
    bindings: &[(String, Expr, serde_json::Value)],
    body: &Expr,
) -> Result<bool, String> {
    use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
    // Bind all binders via a let-chain around the body.
    let mut wrapped = body.clone();
    for (name, value, _) in bindings.iter().rev() {
        wrapped = node(
            "let",
            vec![node("bind", vec![sym(name), value.clone()]), wrapped],
        );
    }
    let stripped: Vec<Expr> = exprs.iter().map(strip_invariant_meta).collect();
    let (program, probe) = inject_probe_into_defining_module(&stripped, home_type, wrapped);
    let source = chelis_deep::printer::print_canonical(&program);
    let result = chelis_compiler_api::compiler::eval_selected(
        EvalRequest {
            source_kind: SourceKind::Deep,
            source,
            bindings: Default::default(),
        },
        &[probe],
    )
    .map_err(|e| {
        e.errors
            .iter()
            .map(|d| d.message.clone())
            .collect::<Vec<_>>()
            .join("; ")
    })?;
    match result.roots.as_slice() {
        [root] => match &root.value {
            ExecutionValue::Bool { value } => Ok(*value),
            ExecutionValue::Tensor { value } if value.shape.is_empty() && value.data.len() == 1 => {
                Ok(value.data.element_f64_lossy(0) != 0.0)
            }
            other => Err(format!("property evaluated to non-bool: {other:?}")),
        },
        _ => Err("property did not return exactly one root".to_string()),
    }
}

fn resolve_constants(
    exprs: &[Expr],
    invariants: &[crate::opaque::OpaqueInvariant],
) -> crate::opaque::ConstEnv {
    use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
    let mut env = crate::opaque::ConstEnv::new();
    let mut referenced = Vec::new();
    for inv in invariants {
        for v in crate::predicate_free_vars(&inv.predicate) {
            if v != inv.binder && !referenced.contains(&v) {
                referenced.push(v);
            }
        }
    }
    if referenced.is_empty() {
        return env;
    }
    let stripped: Vec<Expr> = exprs.iter().map(strip_invariant_meta).collect();
    // A fresh root, so a module that defines the plain spelling cannot turn
    // every probe into a duplicate definition and leave the constant
    // silently unresolved (chelis#3267).
    let probe = fresh_root_name(&stripped, "__chelis_const_probe");
    for name in referenced {
        // Try `name()` then `name`.
        for body in [node("app", vec![var_node(&name)]), var_node(&name)] {
            let program = inject_first_module(&stripped, node("def", vec![sym(&probe), body]));
            let source = chelis_deep::printer::print_canonical(&program);
            if let Ok(result) = chelis_compiler_api::compiler::eval_selected(
                EvalRequest {
                    source_kind: SourceKind::Deep,
                    source,
                    bindings: Default::default(),
                },
                std::slice::from_ref(&probe),
            ) {
                let [root] = result.roots.as_slice() else {
                    continue;
                };
                let value = match &root.value {
                    ExecutionValue::Scalar { value } => {
                        crate::opaque::finite_exact_constant(value.get())
                    }
                    ExecutionValue::Tensor { value }
                        if value.shape.is_empty() && value.data.len() == 1 =>
                    {
                        crate::opaque::finite_exact_constant(value.data.scalar_at(0))
                    }
                    _ => None,
                };
                if let Some(value) = value {
                    env.insert(name.clone(), value);
                    break;
                }
            }
        }
    }
    env
}

/// Exported base producers of the opaque input type for constructor-based
/// generation (mirrors the obligation engine's resolver).
fn generation_producers(
    exprs: &[Expr],
    _invariants: &[crate::opaque::OpaqueInvariant],
    input_inv: &crate::opaque::OpaqueInvariant,
) -> Vec<crate::opaque::GenProducer> {
    crate::obligation_engine::generation_producers_for(exprs, input_inv)
}

fn counterexample(bindings: &[(String, Expr, serde_json::Value)]) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (name, _, json) in bindings {
        map.insert(name.clone(), json.clone());
    }
    serde_json::Value::Object(map)
}

fn conjoin(exprs: &[Expr]) -> Expr {
    exprs
        .iter()
        .cloned()
        .reduce(|a, b| node("app", vec![var_node("and"), a, b]))
        .unwrap_or_else(|| bool_lit(true))
}

// --- sampling ---

fn sample_scalar(prim: &str, rng: &mut crate::opaque::GenRng) -> ScalarValue {
    // Integer widths sample within the width's representable range via the
    // single-source `int_sample_bounds` (review 5).
    if let Some((lo, hi)) = crate::opaque::int_sample_bounds(prim) {
        let span = (hi - lo + 1) as u64;
        return scalar_from_i64(
            "prove-injection-sample",
            Prim::parse_name(prim).expect("integer width is a Prim"),
            lo + (rng.next_u64() % span) as i64,
        )
        .expect("integer sample bounds are representable");
    }
    match prim {
        "bool" => scalar_from_i64(
            "prove-injection-sample",
            Prim::Bool,
            (rng.next_u64() & 1) as i64,
        )
        .expect("boolean sample is exactly zero or one"),
        _ => scalar_from_f64(
            "prove-injection-sample",
            Prim::parse_name(prim).expect("injection scalar dtype is classified"),
            rng_f64(rng),
        )
        .expect("float sample is valid at its declared width"),
    }
}

fn scalar_json(value: ScalarValue) -> serde_json::Value {
    if let Some(value) = value.as_bool_exact() {
        serde_json::Value::from(value)
    } else if let Some(value) = value.as_i64_exact() {
        serde_json::Value::from(value)
    } else {
        serde_json::Value::from(value.as_f64_lossy())
    }
}

fn rng_f64(rng: &mut crate::opaque::GenRng) -> f64 {
    let unit = (rng.next_u64() >> 11) as f64 / ((1u64 << 53) as f64);
    -10.0 + 20.0 * unit
}

// --- Deep builders ---

fn span0() -> Span {
    Span::new(0, 0)
}
fn sym(s: &str) -> Expr {
    Expr::Atom(Atom::Name(s.to_string()), span0())
}
fn node(tag: &str, kids: Vec<Expr>) -> Expr {
    Expr::node(
        DeepTag::parse(tag).expect("vocabulary builder"),
        Metadata::default(),
        kids,
        span0(),
    )
}
fn var_node(name: &str) -> Expr {
    node("var", vec![sym(name)])
}
fn typed_lit(prim: &str, value: Expr) -> Expr {
    let mut entries = Metadata::default();
    entries.replace(M::Type(
        TypeSyntax::try_new(node("t-prim", vec![sym(prim)])).expect("primitive type"),
    ));
    Expr::node(DeepTag::Lit, entries, vec![value], span0())
}
fn scalar_lit(prim: &str, value: ScalarValue) -> Expr {
    debug_assert_eq!(value.prim().name(), prim);
    if let Some(value) = value.as_bool_exact() {
        typed_lit("bool", Expr::Atom(Atom::Bool(value), span0()))
    } else if let Some(value) = value.as_i64_exact() {
        typed_lit(prim, Expr::Atom(Atom::Int(value), span0()))
    } else {
        typed_lit(prim, Expr::Atom(Atom::Float(value.as_f64_lossy()), span0()))
    }
}
fn bool_lit(v: bool) -> Expr {
    typed_lit("bool", Expr::Atom(Atom::Bool(v), span0()))
}

fn list_tag(expr: &Expr) -> Option<DeepTag> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, _, _) => Some(tag),
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn child0_sym(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, _, children) => match children.first()? {
            Expr::Atom(Atom::Name(s), _) => Some(s.as_str()),
            _ => None,
        },
        ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn module_defines(expr: &Expr, type_name: &str) -> bool {
    if list_tag(expr) == Some(DeepTag::Deftype) && child0_sym(expr) == Some(type_name) {
        return true;
    }
    match expr.carrier() {
        ExprCarrier::DecodedNode(_, _, children) => children
            .iter()
            .any(|child| module_defines(child, type_name)),
        ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::MetadataExpression(_) => false,
        ExprCarrier::Atom(_) | ExprCarrier::MetadataMap(_) => false,
    }
}

fn inject_into_module(exprs: &[Expr], type_name: &str, def: Expr) -> Vec<Expr> {
    let mut out = Vec::with_capacity(exprs.len());
    let mut injected = false;
    for expr in exprs {
        if !injected
            && list_tag(expr) == Some(DeepTag::Module)
            && (type_name.is_empty() || module_defines(expr, type_name))
            && let Expr::Node(module, span) = expr
        {
            let mut children = module.children_slice().to_vec();
            children.push(def.clone());
            out.push(Expr::node(
                module.tag(),
                module.meta().clone(),
                children,
                *span,
            ));
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

/// Declare `body` as a probe `def` inside the module that defines
/// `type_name`, returning the program and the probe's binding name.
///
/// Module identity has two spellings, and the checker attributes a
/// declaration to a module through whichever one the program uses
/// (`chelis_types` `module_key_for_item`). A lexical program nests the
/// defining module's declarations in a `module` wrapper, so the probe joins
/// that wrapper. A reef-linked program has no wrappers: every declaration
/// carries its module in its linker-format name, so the probe takes the
/// linker-format name of the type's own module (chelis#2416). Deriving the
/// probe's name from the type's name keeps the two attributions equal by
/// construction. Either spelling is then made fresh against the program
/// (chelis#3267); the suffix extends the linker-format terminal, so the
/// fresh name stays in the type's module.
fn inject_probe_into_defining_module(
    exprs: &[Expr],
    type_name: &str,
    body: Expr,
) -> (Vec<Expr>, String) {
    match chelis_types::linked_binding_in_module_of(type_name, "chelis_prop_probe") {
        Some(linked) => {
            let probe = fresh_root_name(exprs, &linked);
            let mut program = exprs.to_vec();
            program.push(node("def", vec![sym(&probe), body]));
            (program, probe)
        }
        None => {
            let probe = fresh_root_name(exprs, "__chelis_prop_probe");
            let def = node("def", vec![sym(&probe), body]);
            (inject_into_module(exprs, type_name, def), probe)
        }
    }
}

fn inject_first_module(exprs: &[Expr], def: Expr) -> Vec<Expr> {
    inject_into_module(exprs, "", def)
}

fn strip_invariant_meta(expr: &Expr) -> Expr {
    match expr {
        Expr::Node(node, span) => {
            let mut metadata = node.meta().clone();
            if node.tag() == DeepTag::Deftype {
                metadata.remove(K::Invariant);
                metadata.remove(K::InvariantAmenability);
            }
            let children = node
                .children_slice()
                .iter()
                .map(strip_invariant_meta)
                .collect();
            Expr::node(node.tag(), metadata, children, *span)
        }
        // A structural list is walked as the untagged list it replaced was.
        Expr::BareList(elements, span) => {
            Expr::BareList(elements.iter().map(strip_invariant_meta).collect(), *span)
        }
        other => other.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn resolved_module_constants(source: &str) -> crate::opaque::ConstEnv {
        let declarations =
            chelis_surf::parser::parse_str(source).expect("parse the invariant module");
        let exprs = chelis_surf::desugar::desugar_program(&declarations)
            .expect("desugar the invariant module");
        let invariants = crate::opaque::collect_opaque_invariants(&exprs);
        resolve_constants(&exprs, &invariants)
    }

    #[test]
    fn referenced_module_float_constant_is_resolved_for_invariant_validation() {
        let source = "module Stats.Simplex
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.0001
@property generated forall(p: Simplex):
  true
";
        let constants = resolved_module_constants(source);

        assert!(
            (constants.get("eps").copied().unwrap_or_default() - 0.0001).abs() < 1e-8,
            "the predicate's `eps` constant resolves before sample validation: {constants:?}"
        );
    }

    #[test]
    fn chelis_3267_constant_resolves_beside_a_definition_of_the_probe_spelling() {
        let source = "module Stats.Simplex
@opaque
@invariant(p) sum(p.weights) >= 1.0 - eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> f32 = 0.0001
__chelis_const_probe = true
@property generated forall(p: Simplex):
  true
";
        let constants = resolved_module_constants(source);

        assert!(
            (constants.get("eps").copied().unwrap_or_default() - 0.0001).abs() < 1e-8,
            "a module definition of the probe's spelling must not leave `eps` unresolved: \
             {constants:?}"
        );
    }

    #[test]
    fn chelis_3267_property_probe_avoids_a_definition_of_its_spelling() {
        let body = bool_lit(true);
        // Lexical program: the probe joins the defining module's wrapper.
        let lexical = vec![node(
            "module",
            vec![
                sym("M"),
                node("def", vec![sym("__chelis_prop_probe"), bool_lit(false)]),
            ],
        )];
        let (program, probe) = inject_probe_into_defining_module(&lexical, "", body.clone());
        assert_eq!(probe, "__chelis_prop_probe_1");
        assert_eq!(program.len(), 1, "the probe joins the module wrapper");

        // Linked program: the probe keeps the linker-format name of the
        // type's module, suffixed past the module's own definition.
        let linked = vec![node(
            "def",
            vec![sym("pkg__m__M__chelis_prop_probe"), bool_lit(false)],
        )];
        let (program, probe) = inject_probe_into_defining_module(&linked, "Pkg__m__M__T", body);
        assert_eq!(probe, "pkg__m__M__chelis_prop_probe_1");
        assert_eq!(program.len(), 2);
    }

    #[test]
    fn referenced_module_rank_zero_tensor_constant_is_resolved_for_invariant_validation() {
        let source = "module Stats.Simplex
@opaque
@invariant(p) sum(p.weights) >= eps && sum(p.weights) <= 1.0 + eps
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def eps() -> tensor[f32] = scalar_to_tensor(0.0001f32)
@property generated forall(p: Simplex):
  true
";
        let constants = resolved_module_constants(source);

        assert!(
            (constants.get("eps").copied().unwrap_or_default() - 0.0001).abs() < 1e-8,
            "the rank-zero tensor `eps` constant resolves: {constants:?}"
        );
    }

    #[test]
    fn non_finite_module_float_constants_are_omitted_from_invariant_validation() {
        let source = "module Stats.Simplex
@opaque
@invariant(p) sum(p.weights) >= nan && sum(p.weights) >= positive_inf && sum(p.weights) >= negative_inf
type Simplex =
  | Simplex { weights: tensor[3, f32] }
def nan() -> f32 = 0.0f32 / 0.0f32
def positive_inf() -> f32 = 1.0f32 / 0.0f32
def negative_inf() -> f32 = -1.0f32 / 0.0f32
@property generated forall(p: Simplex):
  true
";
        let constants = resolved_module_constants(source);

        for name in ["nan", "positive_inf", "negative_inf"] {
            assert!(
                !constants.contains_key(name),
                "non-finite constant `{name}` is omitted: {constants:?}"
            );
        }
    }

    #[test]
    fn integer_constant_without_exact_float_representation_is_omitted() {
        let source = "module Stats.IntegerBound
@opaque
@invariant(p) p.value >= bound
type LargeInt =
  | LargeInt { value: i64 }
def bound() -> i64 = 9007199254740993i64
";
        let constants = resolved_module_constants(source);
        assert!(
            !constants.contains_key("bound"),
            "a rounded integer must not enter the float constant environment: {constants:?}"
        );
    }

    fn parsed_module_property(
        source: &str,
        property_name: &str,
    ) -> (Vec<Decl>, Vec<usize>, Vec<Param>) {
        let parsed =
            chelis_surf::parser::parse_str(source).expect("parse injection routing fixture");
        let module_decls = parsed
            .iter()
            .find_map(|decl| match decl {
                Decl::Module { decls, .. } => Some(decls.clone()),
                _ => None,
            })
            .expect("fixture has a module");
        let (property_index, params) = module_decls
            .iter()
            .enumerate()
            .find_map(|decl| match decl {
                (index, Decl::Property { name, params, .. }) if name == property_name => {
                    Some((index, params.clone()))
                }
                _ => None,
            })
            .expect("fixture has the requested property");
        (module_decls, vec![property_index], params)
    }

    #[test]
    fn valid_plain_scalar_property_is_checked_without_selecting_injection() {
        let (decls, property_path, params) = parsed_module_property(
            "module M
@property nested_grad forall(x: f32):
  (grad(grad(fn (xx: f32) -> xx * xx, wrt=xx), wrt=xx)(x) >= 0.0)
",
            "nested_grad",
        );

        assert_eq!(
            property_has_opaque_invariant_binder(&decls, &property_path, &params),
            Ok(false),
            "a valid property without an opaque binder is not injection-owned"
        );
    }

    #[test]
    fn opaque_invariant_binder_still_selects_injection() {
        let (decls, property_path, params) = parsed_module_property(
            "module M
@opaque
@invariant(p) p.value >= 0.0
type Probability =
  | Probability { value: f32 }
@property bounded forall(p: Probability):
  (p.value >= 0.0)
",
            "bounded",
        );

        assert_eq!(
            property_has_opaque_invariant_binder(&decls, &property_path, &params),
            Ok(true),
            "an invariant-carrying opaque binder must remain injection-owned"
        );
    }

    #[test]
    fn module_search_does_not_recurse_into_unknown_forms() {
        let span = span0();
        let deftype = node("deftype", vec![sym("Token"), Expr::BareList(vec![], span)]);
        assert!(module_defines(&deftype, "Token"));
        let unknown = Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: "future-wrapper".into(),
            meta: Metadata::default(),
            children: vec![deftype],
            span,
        }));

        assert!(!module_defines(&unknown, "Token"));
    }

    #[test]
    fn injection_does_not_discover_types_through_metadata_wrapper() {
        let span = span0();
        let wrapped = Expr::MetaExpr(
            chelis_deep::MetaExpr {
                metadata: Metadata::default(),
                expr: Box::new(node(
                    "deftype",
                    vec![sym("Token"), Expr::BareList(vec![], span)],
                )),
            },
            span,
        );
        let module = node("module", vec![sym("M"), wrapped]);
        let marker = sym("marker");

        let injected = inject_into_module(&[module], "Token", marker.clone());
        assert_eq!(
            injected.len(),
            2,
            "metadata wrappers had no type-discovery authority before this slice"
        );
        assert_eq!(injected[1], marker);
    }

    #[test]
    fn injection_attempt_cap_is_finite_and_honors_override() {
        // Without an override, the cap derives from samples_needed (the
        // surf/deep parity formula) and is a finite, positive bound -- this
        // is the value that now terminates the outer rejection loop.
        assert_eq!(max_injection_attempts(10, None), 1000);
        // A floor of samples_needed protects the zero-multiplier edge.
        assert_eq!(max_injection_attempts(0, None), 0);
        // An explicit override is honored verbatim (used by the loop below).
        assert_eq!(max_injection_attempts(10, Some(7)), 7);
    }

    /// Faithful model of the outer rejection loop in `prove_with_injection`:
    /// every iteration generates a (conceptually valid) binder, the
    /// precondition rejects it (`Ok(false)` -> `continue` without advancing
    /// `accepted`), and the body never runs. The `never_accepts` flag stands
    /// in for "no invariant-valid binder can satisfy the precondition." The
    /// return mirrors the production control flow: a determinate
    /// generator-exhausted reason when the cap is hit, else `Ok`.
    fn run_rejection_loop(
        samples_needed: usize,
        max_attempts: Option<usize>,
        never_accepts: bool,
    ) -> Result<usize, String> {
        let cap = max_injection_attempts(samples_needed, max_attempts);
        let mut accepted = 0usize;
        let mut attempts = 0usize;
        while accepted < samples_needed && attempts < cap {
            attempts += 1;
            if never_accepts {
                // precondition evaluated Ok(false): skip without advancing.
                continue;
            }
            accepted += 1;
        }
        if accepted < samples_needed {
            return Err(exhaustion_reason(attempts, samples_needed));
        }
        Ok(accepted)
    }

    #[test]
    fn unsatisfiable_precondition_terminates_with_exhaustion_not_hang() {
        // Regression for the no-cap spin: a precondition no valid binder can
        // satisfy used to loop forever (each iteration `continue`d without
        // incrementing `accepted`). With the cap, the loop terminates at the
        // (tiny) attempt bound and returns the determinate generator-exhausted
        // outcome the surf/deep loops return. Without the `attempts < cap`
        // bound this call would never return.
        let result = run_rejection_loop(5, Some(3), /* never_accepts */ true);
        let reason = result.expect_err("unsatisfiable precondition must report exhaustion");
        assert_eq!(
            reason,
            "generator exhausted after 3 attempts before collecting 5 valid samples"
        );
    }

    #[test]
    fn satisfiable_precondition_path_collects_all_samples() {
        // The normal path is unchanged: when every sample is accepted the
        // loop collects exactly `samples_needed` and does not report
        // exhaustion (it never reaches the cap).
        let result = run_rejection_loop(5, Some(1000), /* never_accepts */ false);
        assert_eq!(result.expect("accepting path succeeds"), 5);
    }

    #[test]
    fn injected_int64_scalar_sample_is_not_widened_through_f64() {
        let mut rng = crate::opaque::GenRng::new(9);
        let value = sample_scalar("i64", &mut rng);

        assert_eq!(value.prim(), chelis_types::types::Prim::Int64);
        assert!(value.as_i64_exact().is_some());
        assert!(value.as_bool_exact().is_none());
    }
}
