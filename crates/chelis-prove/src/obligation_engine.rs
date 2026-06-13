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
}

impl Default for ObligationRunOptions {
    fn default() -> Self {
        Self {
            seed: 0,
            samples: 100,
            smt_timeout_ms: 5000,
            tier: "auto".to_string(),
            only: None,
        }
    }
}

/// Run obligations directly from module SOURCE (Surf `.ch` text). This is
/// the entry the chelis-tide MCP tool calls: it desugars, runs the
/// checker for inferred return types, then runs the same engine the CLI
/// uses (RFC D-PARITY). Returns `Err` if the source does not parse.
pub fn run_surf_source_obligations(
    source: &str,
    options: &ObligationRunOptions,
) -> Result<Vec<ObligationOutcome>, String> {
    let decls = chelis_surf::parser::parse_str(source).map_err(|e| format!("parse: {e}"))?;
    let exprs = chelis_surf::desugar::desugar_program(&decls);
    let sigs: BTreeMap<String, Type> = match chelis_types::check_typed_program(&exprs) {
        Ok(checked) => checked
            .signature_inference()
            .functions
            .iter()
            .map(|(n, s)| (n.clone(), s.checked_signature.clone()))
            .collect(),
        // Type errors: no obligations (the check surface reports them).
        Err(_) => BTreeMap::new(),
    };
    Ok(run_module_obligations(&exprs, &sigs, options))
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
        out.push(run_one(exprs, inv, ob, sigs, &consts, options));
    }
    out
}

fn run_one(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    sigs: &BTreeMap<String, Type>,
    consts: &ConstEnv,
    options: &ObligationRunOptions,
) -> ObligationOutcome {
    // Tier B.
    if options.tier == "auto" || options.tier == "smt-only" {
        let pparams = producer_param_types(sigs, &ob.producer);
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
    run_tier_c(exprs, inv, ob, sigs, options)
}

fn run_tier_c(
    exprs: &[Expr],
    inv: &OpaqueInvariant,
    ob: &ObligationProperty,
    sigs: &BTreeMap<String, Type>,
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
    let mut kinds = Vec::new();
    for t in arg_types {
        match t {
            Type::Prim(p) => kinds.push(prim_name(p)),
            _ => {
                return outcome(
                    ob,
                    ObligationStatus::Unsupported,
                    ObligationTier::Fuzz,
                    0,
                    seed,
                    None,
                    Some("producer has a non-scalar parameter (Tier C V1)".to_string()),
                );
            }
        }
    }
    let names = producer_param_names(exprs, &ob.producer);
    if names.len() != kinds.len() {
        return outcome(
            ob,
            ObligationStatus::Unsupported,
            ObligationTier::Fuzz,
            0,
            seed,
            None,
            Some("producer parameter arity mismatch".to_string()),
        );
    }
    let mut rng = Lcg::new(seed);
    for n in 0..options.samples {
        let args: Vec<(String, f64)> = names
            .iter()
            .zip(&kinds)
            .map(|(name, kind)| (name.clone(), sample_scalar(kind, &mut rng)))
            .collect();
        match eval_obligation_body(exprs, inv, ob, &args) {
            Ok(true) => {}
            Ok(false) => {
                let cx = args
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::json!(v)))
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
) -> Vec<(String, ProducerParamType)> {
    match sigs.get(producer) {
        Some(Type::Fn(args, _)) => args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let pt = match a {
                    Type::Prim(p) => ProducerParamType::Scalar(prim_name(p)),
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
    let probe = "__chelis_const_probe";
    let deep_probe = format!("{source}\n(def {{}} {probe} (var {{}} {name}))\n");
    let result = chelis_compiler_api::compiler::eval_selected(
        EvalRequest {
            source_kind: SourceKind::Deep,
            source: deep_probe,
            bindings: Default::default(),
        },
        &[probe.to_string()],
    )
    .ok()?;
    match result.roots.as_slice() {
        [root] => match &root.value {
            ExecutionValue::Tensor { value } if value.shape.is_empty() && value.data.len() == 1 => {
                Some(value.data[0])
            }
            _ => None,
        },
        _ => None,
    }
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
