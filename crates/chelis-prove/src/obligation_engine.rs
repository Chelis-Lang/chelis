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
    pub counterexample: Option<serde_json::Value>,
    pub reason: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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

/// Run all derived producer obligations of a desugared Deep program.
/// `sigs` is the checker-inferred def-name -> type map (from
/// `chelis_types::check_typed_program`). Returns one outcome per
/// obligation plus one `Error` outcome per collection error.
pub fn run_module_obligations(
    exprs: &[Expr],
    sigs: &BTreeMap<String, Type>,
    options: &ObligationRunOptions,
) -> Vec<ObligationOutcome> {
    let invariants = crate::opaque::collect_opaque_invariants(exprs);
    if invariants.is_empty() {
        return Vec::new();
    }
    let consts = resolve_module_constants(exprs, &invariants);
    let collection = crate::obligations::collect_obligations(exprs, &invariants, sigs);

    let mut out = Vec::new();
    for err in &collection.errors {
        out.push(ObligationOutcome {
            name: String::new(),
            meta: ObligationMeta {
                obligation_kind: "invariant_producer".to_string(),
                source_type: String::new(),
                producer: String::new(),
            },
            status: ObligationStatus::Error,
            proof_tier: ObligationTier::None,
            samples: 0,
            seed: options.seed,
            counterexample: None,
            reason: Some(err.to_string()),
        });
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
        out.push(run_one(exprs, inv, &invariants, ob, sigs, &consts, options));
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
            match crate::tier_b::solve_property(&lowered.property, options.smt_timeout_ms) {
                TierBResult::Proved => {
                    return outcome(
                        ob,
                        ObligationStatus::Passed,
                        ObligationTier::Smt,
                        0,
                        options.seed,
                        None,
                        None,
                    );
                }
                TierBResult::Disproved(model) => {
                    return outcome(
                        ob,
                        ObligationStatus::Failed,
                        ObligationTier::Smt,
                        0,
                        options.seed,
                        Some(model),
                        None,
                    );
                }
                TierBResult::Timeout | TierBResult::Unknown => {
                    if options.tier == "smt-only" {
                        return outcome(
                            ob,
                            ObligationStatus::Unsupported,
                            ObligationTier::Smt,
                            0,
                            options.seed,
                            None,
                            Some("smt timeout/unknown".to_string()),
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
        return match eval_obligation_body(exprs, inv, ob, &[]) {
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
                let cx = arg_values
                    .iter()
                    .map(|a| (a.name.clone(), a.json.clone()))
                    .collect::<serde_json::Map<_, _>>();
                return outcome(
                    ob,
                    ObligationStatus::Failed,
                    ObligationTier::Fuzz,
                    n + 1,
                    seed,
                    Some(serde_json::Value::Object(cx)),
                    None,
                );
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
enum ArgKind {
    Scalar(String),
    Tensor(Vec<usize>),
    Opaque(OpaqueInvariant),
}

/// A sampled argument value: the Deep value expr passed to the producer
/// call, a JSON repr for counterexamples, and the param name.
struct ArgValue {
    expr: Expr,
    json: serde_json::Value,
    name: String,
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

/// Public entry: the exported base producers of `input_inv`'s type usable
/// for constructor-based generation, computing the inferred signatures
/// from the program. Used by the user-property injection path (D-INJECT)
/// so it shares the obligation engine's producer-resolution rules.
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
                    match lit {
                        Some(d) => kinds.push(GenParamKind::Tensor {
                            dims: d,
                            precision: "f32".to_string(),
                        }),
                        None => {
                            raw_ok = false;
                            break;
                        }
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
    ObligationOutcome {
        name: ob.name.clone(),
        meta: ob.meta.clone(),
        status,
        proof_tier: tier,
        samples,
        seed,
        counterexample,
        reason,
    }
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
    format!("{p:?}").to_lowercase()
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
    let source = chelis_deep::printer::print_canonical(exprs);
    for name in referenced {
        if let Some(value) = eval_scalar_const(&source, &name) {
            env.insert(name, value);
        }
    }
    env
}

fn eval_scalar_const(source: &str, name: &str) -> Option<f64> {
    // An in-module constant may be a value binding (`(var name)`) or a
    // zero-argument constant function (`(app (var name))`, the desugaring
    // of `def eps() -> f32 = 0.01`). The probe def must live INSIDE the
    // same module as the constant, so reparse and inject into the module
    // wrapper. Try the call form first, then the bare-var form.
    // Strip the invariant metadata so a `sum`-bearing predicate does not
    // block IR lowering of the module when we evaluate the constant.
    let exprs: Vec<Expr> = chelis_deep::parser::parse_str(source)
        .ok()?
        .iter()
        .map(strip_invariant_meta)
        .collect();
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

fn eval_obligation_body(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    args: &[(String, f64)],
) -> Result<bool, String> {
    let probe = "__chelis_obligation_probe";
    // The synthetic defs reference the opaque type's fields (via the
    // typed predicate) and call the producer, so they MUST live inside the
    // defining module — otherwise the opacity checker rejects the field
    // access as out-of-module. Insert them into the module wrapper.
    let inv_def = deep_node(
        "def",
        vec![deep_sym("__chelis_inv_holds"), typed_predicate(inv)],
    );
    let probe_def = deep_node("def", vec![deep_sym(probe), obligation_body_expr(ob, args)]);
    let program = inject_into_defining_module(exprs, &inv.type_name, vec![inv_def, probe_def]);
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
        [root] => match &root.value {
            ExecutionValue::Bool { value } => Ok(*value),
            ExecutionValue::Tensor { value } if value.shape.is_empty() && value.data.len() == 1 => {
                Ok(value.data[0] != 0.0)
            }
            other => Err(format!("obligation evaluated to non-bool: {other:?}")),
        },
        _ => Err("obligation did not return exactly one root".to_string()),
    }
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

/// Rebuild the invariant predicate fn with its binder typed as the
/// opaque ADT: `(fn {} (params {} (<binder> {type: (t-adt {} <Type>)}))
/// <body>)`. Inside the defining module the field access `<binder>.field`
/// is then resolvable by the checker (field access on the opaque type is
/// legal in-module), so the synthesized `__chelis_inv_holds` def does not
/// trip the opacity gate during obligation evaluation.
fn typed_predicate(inv: &OpaqueInvariant) -> Expr {
    let body = node_children(&inv.predicate)
        .get(1)
        .cloned()
        .unwrap_or_else(|| deep_bool_lit(true));
    let typed_binder = {
        let mut entries = MetaMap::default();
        entries.entries.push((
            "type".to_string(),
            deep_node("t-adt", vec![deep_sym(&inv.type_name)]),
        ));
        Expr::List(
            List {
                elements: vec![deep_sym(&inv.binder), Expr::Map(entries, Span::new(0, 0))],
            },
            Span::new(0, 0),
        )
    };
    let params = deep_node("params", vec![typed_binder]);
    deep_node("fn", vec![params, body])
}

fn obligation_body_expr(ob: &ObligationProperty, args: &[(String, f64)]) -> Expr {
    let call = if ob.is_constant {
        deep_var(&ob.producer)
    } else {
        let mut app = vec![deep_var(&ob.producer)];
        for (_, v) in args {
            app.push(deep_float_lit(*v));
        }
        deep_node("app", app)
    };
    position_body(&ob.position, call)
}

/// Evaluate the obligation over richer argument VALUES (opaque records,
/// tensors, scalars) by EVALUATING THE PRODUCER and validating each
/// produced value's representation against the invariant via
/// `concrete_eval` (not by evaluating the predicate through the host
/// runtime — that cannot lower `sum`/constant-bearing tensor invariants).
/// This makes the update-shaped and tensor-input obligations supported,
/// and matches the generator's validation so proposal and acceptance
/// agree.
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

    // Two validation strategies. For a scalar-only-field invariant the
    // predicate evaluates fine through the host runtime (the W3 path,
    // which also handles every produced position incl. tuples). A
    // tensor-field invariant (`sum`/constant-bearing) cannot lower through
    // the runtime, so we EVALUATE THE PRODUCER and validate each produced
    // value's representation via `concrete_eval` (Direct / Option-of-Direct
    // in V1).
    let all_scalar = inv
        .fields
        .iter()
        .all(|(_, f)| matches!(f, crate::opaque::FieldType::Scalar(_)));
    if all_scalar {
        return eval_obligation_predicate(exprs, inv, ob, call);
    }
    let predicate = crate::opaque::lower_predicate_flattened(inv, &inv.binder, consts)
        .ok_or_else(|| "invariant predicate does not lower for validation".to_string())?;
    let module_source = chelis_deep::printer::print_canonical(exprs);
    validate_position(&module_source, inv, &ob.position, call, &predicate)
}

/// The W3 predicate-eval path: build `position_body(__chelis_inv_holds,
/// producer(args))` and evaluate it through the runtime. Works for
/// scalar-field invariants and every produced position.
fn eval_obligation_predicate(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    call: Expr,
) -> Result<bool, String> {
    let probe = "__chelis_obligation_probe";
    let inv_def = deep_node(
        "def",
        vec![deep_sym("__chelis_inv_holds"), typed_predicate(inv)],
    );
    let probe_def = deep_node(
        "def",
        vec![deep_sym(probe), position_body(&ob.position, call)],
    );
    let program = inject_into_defining_module(exprs, &inv.type_name, vec![inv_def, probe_def]);
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
        [root] => match &root.value {
            ExecutionValue::Bool { value } => Ok(*value),
            ExecutionValue::Tensor { value } if value.shape.is_empty() && value.data.len() == 1 => {
                Ok(value.data[0] != 0.0)
            }
            other => Err(format!("obligation evaluated to non-bool: {other:?}")),
        },
        _ => Err("obligation did not return exactly one root".to_string()),
    }
}

/// Validate every produced occurrence of the opaque type at `position`
/// inside `value_expr` against the invariant. Returns Ok(true) when every
/// occurrence satisfies the invariant (None-wrapped failures pass
/// vacuously), Ok(false) on a violation, Err on an evaluation error.
fn validate_position(
    module_source: &str,
    inv: &OpaqueInvariant,
    position: &ProducedPosition,
    value_expr: Expr,
    predicate: &crate::solver::SmtExpr,
) -> Result<bool, String> {
    match position {
        ProducedPosition::Direct => {
            // Read the record's representation fields and validate.
            match read_record_env(module_source, inv, value_expr)? {
                Some(env) => Ok(validate_with_predicate(&env, predicate)),
                // A None-projected value (NaN sentinel) cannot occur for a
                // Direct position; treat unreadable as an error upstream.
                None => Ok(true),
            }
        }
        ProducedPosition::InsideOption(inner) => {
            // Project the Some-branch value; a None result passes
            // vacuously (read_record_env returns None on the NaN sentinel).
            let some_value = opt_some_projection(value_expr, inner);
            match read_record_env(module_source, inv, some_value)? {
                Some(env) => Ok(validate_with_predicate(&env, predicate)),
                None => Ok(true), // None result: vacuously satisfied.
            }
        }
        ProducedPosition::TupleComponents(comps) => {
            for (idx, inner) in comps {
                let comp = deep_node(
                    "tuple-get",
                    vec![value_expr.clone(), deep_int_lit(*idx as i64)],
                );
                if !validate_position(module_source, inv, inner, comp, predicate)? {
                    return Ok(false);
                }
            }
            Ok(true)
        }
    }
}

/// Build the value to read for an `InsideOption` position: `match v {
/// Some(__v) => <inner-of-__v> | _ => <NaN sentinel> }`, where the inner
/// is the value at the inner position (Direct => `__v`).
fn opt_some_projection(value_expr: Expr, inner: &ProducedPosition) -> Expr {
    let inner_value = match inner {
        ProducedPosition::Direct => deep_var("__v"),
        // Nested Option/tuple: project further by recursion at read time.
        _ => deep_var("__v"),
    };
    deep_node(
        "match",
        vec![
            value_expr,
            deep_node(
                "arm",
                vec![
                    deep_node(
                        "pat-ctor",
                        vec![
                            deep_sym("Some"),
                            deep_node("pat-var", vec![deep_sym("__v")]),
                        ],
                    ),
                    deep_bare_list(vec![]),
                    inner_value,
                ],
            ),
            deep_node(
                "arm",
                vec![
                    deep_node("pat-wild", vec![]),
                    deep_bare_list(vec![]),
                    deep_var("__chelis_none_sentinel"),
                ],
            ),
        ],
    )
}

/// Read an opaque record value's representation fields into a flattened
/// env keyed by `<binder>.<field>[.<i>]`. Returns `None` when the value is
/// a None sentinel (NaN-projected). The probe is injected into the
/// defining module with the invariant metadata stripped (so the
/// shape-sensitive predicate does not block IR lowering) and a
/// `__chelis_none_sentinel` def supplying a NaN record.
fn read_record_env(
    module_source: &str,
    inv: &OpaqueInvariant,
    value_expr: Expr,
) -> Result<Option<BTreeMap<String, f64>>, String> {
    let mut env = BTreeMap::new();
    for (fname, fty) in &inv.fields {
        let field_path = format!("{}.{}", inv.binder, fname);
        match read_one_field(
            module_source,
            inv,
            &value_expr,
            fname,
            fty,
            &field_path,
            &mut env,
        )? {
            true => {}
            false => return Ok(None), // None sentinel detected.
        }
    }
    Ok(Some(env))
}

#[allow(clippy::too_many_arguments)]
fn read_one_field(
    module_source: &str,
    inv: &OpaqueInvariant,
    value_expr: &Expr,
    _field: &str,
    fty: &crate::opaque::FieldType,
    field_path: &str,
    env: &mut BTreeMap<String, f64>,
) -> Result<bool, String> {
    let probe = "__chelis_read_probe";
    // Bind the value and project the field. A None-sentinel record yields
    // NaN, which we detect.
    let body = let_block(
        "__r",
        value_expr.clone(),
        access_node(deep_var("__r"), _field),
    );
    let probe_def = node_def(probe, body);
    // A None sentinel: a record whose fields are NaN.
    let sentinel_def = node_def("__chelis_none_sentinel", none_sentinel_record(inv));
    let program =
        inject_into_module_stripped(module_source, &inv.type_name, vec![sentinel_def, probe_def])?;
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
    let root = match result.roots.as_slice() {
        [r] => r,
        _ => return Err("field read did not return one root".to_string()),
    };
    let ExecutionValue::Tensor { value } = &root.value else {
        return Err("field read returned a non-tensor value".to_string());
    };
    if value.data.iter().any(|v| v.is_nan()) {
        return Ok(false); // None sentinel.
    }
    match fty {
        crate::opaque::FieldType::Tensor { dims, .. } => {
            let count = dims.iter().product::<usize>().max(1);
            if value.data.len() != count {
                return Err("field read shape mismatch".to_string());
            }
            for (i, v) in value.data.iter().enumerate() {
                env.insert(format!("{field_path}.{i}"), *v);
            }
        }
        _ => {
            let v = *value
                .data
                .first()
                .ok_or_else(|| "empty scalar field read".to_string())?;
            env.insert(field_path.to_string(), v);
        }
    }
    Ok(true)
}

fn validate_with_predicate(
    env: &BTreeMap<String, f64>,
    predicate: &crate::solver::SmtExpr,
) -> bool {
    let hash: std::collections::HashMap<String, f64> =
        env.iter().map(|(k, v)| (k.clone(), *v)).collect();
    crate::concrete_eval::eval_bool(predicate, &hash)
}

/// A None-sentinel record: every field NaN-filled (0.0/0.0) so the reader
/// detects a None result.
fn none_sentinel_record(inv: &OpaqueInvariant) -> Expr {
    let mut children = vec![deep_sym(&inv.ctor_name)];
    for (fname, fty) in &inv.fields {
        let nan = deep_node(
            "app",
            vec![deep_var("div"), deep_float_lit(0.0), deep_float_lit(0.0)],
        );
        let value = match fty {
            crate::opaque::FieldType::Tensor { dims, .. } => {
                let count = dims.iter().product::<usize>().max(1);
                let elems: Vec<Expr> = (0..count).map(|_| nan.clone()).collect();
                deep_node("app", vec![deep_var("to_tensor"), deep_cons_list(elems)])
            }
            _ => nan,
        };
        children.push(deep_node("kv", vec![deep_sym(fname), value]));
    }
    deep_node("record", children)
}

fn deep_cons_list(items: Vec<Expr>) -> Expr {
    items.into_iter().rev().fold(deep_var("Nil"), |tail, item| {
        deep_node("app", vec![deep_var("Cons"), item, tail])
    })
}

fn access_node(target: Expr, field: &str) -> Expr {
    deep_node("access", vec![target, deep_sym(field)])
}

fn let_block(bind: &str, value: Expr, body: Expr) -> Expr {
    deep_node(
        "let",
        vec![deep_node("bind", vec![deep_sym(bind), value]), body],
    )
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

/// Build a typed scalar literal Deep expr for a producer argument.
fn scalar_lit(prim: &str, v: f64) -> Expr {
    match prim {
        "int32" | "int64" => deep_int_lit(v as i64),
        "bool" => deep_bool_lit(v != 0.0),
        _ => deep_float_lit(v),
    }
}

fn position_body(position: &ProducedPosition, value: Expr) -> Expr {
    match position {
        ProducedPosition::Direct => deep_node("app", vec![deep_var("__chelis_inv_holds"), value]),
        ProducedPosition::InsideOption(inner) => {
            let inner_body = position_body(inner, deep_var("__v"));
            deep_node(
                "match",
                vec![
                    value,
                    deep_node(
                        "arm",
                        vec![
                            deep_node(
                                "pat-ctor",
                                vec![
                                    deep_sym("Some"),
                                    deep_node("pat-var", vec![deep_sym("__v")]),
                                ],
                            ),
                            deep_bare_list(vec![]),
                            inner_body,
                        ],
                    ),
                    deep_node(
                        "arm",
                        vec![
                            deep_node("pat-wild", vec![]),
                            deep_bare_list(vec![]),
                            deep_bool_lit(true),
                        ],
                    ),
                ],
            )
        }
        ProducedPosition::TupleComponents(comps) => {
            let mut conj = deep_bool_lit(true);
            for (idx, inner) in comps {
                let comp = deep_node("tuple-get", vec![value.clone(), deep_int_lit(*idx as i64)]);
                let inner_body = position_body(inner, comp);
                conj = deep_node("app", vec![deep_var("and"), conj, inner_body]);
            }
            conj
        }
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
    match kind {
        "int32" | "int64" => rng.next_i64(-1000, 1000) as f64,
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
fn deep_bare_list(children: Vec<Expr>) -> Expr {
    Expr::List(List { elements: children }, Span::new(0, 0))
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
fn deep_int_lit(v: i64) -> Expr {
    deep_typed_lit("int32", Expr::Atom(Atom::Int(v), Span::new(0, 0)))
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
