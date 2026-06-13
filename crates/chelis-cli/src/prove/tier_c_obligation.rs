//! Tier C (fuzz) fallback for derived producer obligations (RFC D-OBLIG,
//! D-STARVE).
//!
//! When an obligation does not lower to Tier B (residual match, non-scalar
//! producer params, over-cap tensor, or the `smt` feature is not built),
//! it is validated by sampling the PRODUCER's raw inputs and evaluating
//! the obligation body
//! `match producer(args) with { Some v => inv(v) | _ => true }` through the
//! interpreter. There is no opaque-binder sampling here: the obligation
//! constructs the opaque value via the producer itself, so only the
//! producer's scalar inputs are sampled. (Opaque-binder sampling /
//! tiered generation, D-STARVE, drives assumption injection on user
//! properties, handled separately.)
#![cfg(feature = "chelis-prove")]

use std::collections::BTreeMap;

use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_types::types::Type;

use super::obligation_run::emit_obligation_record;
use super::{ProveOptions, Status, Summary};

/// Run one obligation through Tier C. Samples the producer's scalar
/// inputs, evaluates the obligation body, and reports pass/fail.
pub(super) fn run_obligation_tier_c(
    exprs: &[Expr],
    inv: &chelis_prove::opaque::OpaqueInvariant,
    ob: &chelis_prove::obligations::ObligationProperty,
    sigs: &BTreeMap<String, Type>,
    _consts: &chelis_prove::opaque::ConstEnv,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
) -> Status {
    let seed = options.seed.unwrap_or(0);

    // Constant producers: evaluate inv(value) once (no inputs).
    if ob.is_constant {
        return run_constant_obligation(exprs, inv, ob, options, totals, seed);
    }

    // Determine the producer's scalar param types.
    let Some(Type::Fn(arg_types, _)) = sigs.get(&ob.producer) else {
        // No function signature: cannot sample. Unsupported.
        totals.unsupported += 1;
        emit_obligation_record(
            options,
            ob,
            "unsupported",
            "fuzz",
            0,
            seed,
            None,
            Some("producer has no callable signature to sample".to_string()),
        );
        return Status::Unsupported;
    };

    // Only scalar-input producers are sampled in V1 Tier C.
    let mut param_kinds = Vec::new();
    for t in arg_types {
        match t {
            Type::Prim(p) => param_kinds.push(prim_name(p)),
            _ => {
                totals.unsupported += 1;
                emit_obligation_record(
                    options,
                    ob,
                    "unsupported",
                    "fuzz",
                    0,
                    seed,
                    None,
                    Some("producer has a non-scalar parameter (Tier C V1)".to_string()),
                );
                return Status::Unsupported;
            }
        }
    }

    let samples_needed = options.samples.unwrap_or(100);
    let mut rng = Lcg::new(seed);
    let producer_param_names = producer_param_names(exprs, &ob.producer);
    if producer_param_names.len() != param_kinds.len() {
        totals.unsupported += 1;
        emit_obligation_record(
            options,
            ob,
            "unsupported",
            "fuzz",
            0,
            seed,
            None,
            Some("producer parameter arity mismatch".to_string()),
        );
        return Status::Unsupported;
    }

    for n in 0..samples_needed {
        let args: Vec<(String, f64)> = producer_param_names
            .iter()
            .zip(&param_kinds)
            .map(|(name, kind)| (name.clone(), sample_scalar(kind, &mut rng)))
            .collect();
        match eval_obligation_body(exprs, inv, ob, &args) {
            Ok(true) => {}
            Ok(false) => {
                totals.failed += 1;
                let cx = args
                    .iter()
                    .map(|(k, v)| (k.clone(), serde_json::json!(v)))
                    .collect::<serde_json::Map<_, _>>();
                emit_obligation_record(
                    options,
                    ob,
                    "failed",
                    "fuzz",
                    n + 1,
                    seed,
                    Some(serde_json::Value::Object(cx)),
                    None,
                );
                return Status::Failed;
            }
            Err(err) => {
                totals.errors += 1;
                emit_obligation_record(options, ob, "error", "fuzz", n, seed, None, Some(err));
                return Status::Error;
            }
        }
    }

    totals.passed += 1;
    emit_obligation_record(
        options,
        ob,
        "passed",
        "fuzz",
        samples_needed,
        seed,
        None,
        None,
    );
    Status::Passed
}

fn run_constant_obligation(
    exprs: &[Expr],
    inv: &chelis_prove::opaque::OpaqueInvariant,
    ob: &chelis_prove::obligations::ObligationProperty,
    options: &ProveOptions<'_>,
    totals: &mut Summary,
    seed: u64,
) -> Status {
    match eval_obligation_body(exprs, inv, ob, &[]) {
        Ok(true) => {
            totals.passed += 1;
            emit_obligation_record(options, ob, "passed", "fuzz", 1, seed, None, None);
            Status::Passed
        }
        Ok(false) => {
            totals.failed += 1;
            emit_obligation_record(options, ob, "failed", "fuzz", 1, seed, None, None);
            Status::Failed
        }
        Err(err) => {
            totals.errors += 1;
            emit_obligation_record(options, ob, "error", "fuzz", 0, seed, None, Some(err));
            Status::Error
        }
    }
}

/// Build a Deep program that binds the sampled producer args and
/// evaluates the obligation body, then evaluate it to a bool.
fn eval_obligation_body(
    exprs: &[Expr],
    inv: &chelis_prove::opaque::OpaqueInvariant,
    ob: &chelis_prove::obligations::ObligationProperty,
    args: &[(String, f64)],
) -> Result<bool, String> {
    let probe = "__chelis_obligation_probe";
    let mut program = exprs.to_vec();

    // Define the invariant predicate as a callable def `__inv_holds`.
    program.push(deep_node(
        "def",
        vec![deep_sym("__chelis_inv_holds"), inv.predicate.clone()],
    ));

    // Build the obligation body expression.
    let body = obligation_body_expr(ob, args);
    program.push(deep_node("def", vec![deep_sym(probe), body]));

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

/// Build the obligation body Deep expr: `inv(producer(args))` for Direct,
/// `match producer(args) with {Some v => inv(v) | _ => true}` for Option.
fn obligation_body_expr(
    ob: &chelis_prove::obligations::ObligationProperty,
    args: &[(String, f64)],
) -> Expr {
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

fn position_body(position: &chelis_prove::obligations::ProducedPosition, value: Expr) -> Expr {
    use chelis_prove::obligations::ProducedPosition;
    match position {
        ProducedPosition::Direct => apply_inv(value),
        ProducedPosition::InsideOption(inner) => {
            // match value with { Some v => <inner over v> | None => true }
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
            // Conjoin inv over each carrying component via tuple-get.
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

fn apply_inv(value: Expr) -> Expr {
    deep_node("app", vec![deep_var("__chelis_inv_holds"), value])
}

fn producer_param_names(exprs: &[Expr], producer: &str) -> Vec<String> {
    fn find(exprs: &[Expr], producer: &str) -> Option<Vec<String>> {
        for expr in exprs {
            if list_tag(expr) == Some("def")
                && let Some(name) = children(expr).first().and_then(sym_text)
                && name == producer
                && let Some(fn_node) = children(expr).get(1)
                && list_tag(fn_node) == Some("fn")
                && let Some(params) = children(fn_node).first()
            {
                let mut out = Vec::new();
                for p in children(params) {
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

fn prim_name(p: &chelis_types::types::Prim) -> String {
    format!("{p:?}").to_lowercase()
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

fn deep_float_lit(v: f64) -> Expr {
    let mut entries = MetaMap::default();
    entries.entries.push((
        "type".to_string(),
        deep_node("t-prim", vec![deep_sym("f32")]),
    ));
    Expr::List(
        List {
            elements: vec![
                deep_sym("lit"),
                Expr::Map(entries, Span::new(0, 0)),
                Expr::Atom(Atom::Float(v), Span::new(0, 0)),
            ],
        },
        Span::new(0, 0),
    )
}

fn deep_int_lit(v: i64) -> Expr {
    let mut entries = MetaMap::default();
    entries.entries.push((
        "type".to_string(),
        deep_node("t-prim", vec![deep_sym("int32")]),
    ));
    Expr::List(
        List {
            elements: vec![
                deep_sym("lit"),
                Expr::Map(entries, Span::new(0, 0)),
                Expr::Atom(Atom::Int(v), Span::new(0, 0)),
            ],
        },
        Span::new(0, 0),
    )
}

fn deep_bool_lit(v: bool) -> Expr {
    let mut entries = MetaMap::default();
    entries.entries.push((
        "type".to_string(),
        deep_node("t-prim", vec![deep_sym("bool")]),
    ));
    Expr::List(
        List {
            elements: vec![
                deep_sym("lit"),
                Expr::Map(entries, Span::new(0, 0)),
                Expr::Atom(Atom::Bool(v), Span::new(0, 0)),
            ],
        },
        Span::new(0, 0),
    )
}

fn list_tag(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::List(l, _) => l.elements.first().and_then(sym_text),
        _ => None,
    }
}

fn children(expr: &Expr) -> &[Expr] {
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
