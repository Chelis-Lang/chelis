//! Assumption injection into user-property verification (RFC D-INJECT).
//!
//! When a `@property` binder's type is an invariant-carrying opaque type,
//! verification assumes the invariant of that binder: Tier C generates
//! only invariant-satisfying binder values (via the shared tiered
//! generator, RFC D-STARVE), so a property is checked only over the values
//! the invariant admits. Injection applies ONLY to invariant-carrying
//! opaque binders; non-opaque and invariant-free binders are unaffected
//! (a property that fails without injection still fails).
//!
//! This is the Tier C path. It desugars the module, collects the
//! opaque-invariant registry, and for each accepted sample builds a Deep
//! probe that binds the generated opaque records (and ordinary scalar/
//! tensor binders) and evaluates the property body. The binder values are
//! generated INSIDE the defining module so opaque construction is legal.
//!
//! Gated on the `chelis-prove` optional dependency (the obligation /
//! generation machinery lives there).

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_surf::ast::{Decl, Param, TypeExpr};

use super::{PropertyOutcome, PropertyRunOptions, PropertyStatus, PropertyTier};

/// Does this property have at least one binder whose type is an
/// invariant-carrying opaque type? If so, the injection path owns it.
pub(super) fn property_has_opaque_invariant_binder(decls: &[Decl], params: &[Param]) -> bool {
    let exprs = chelis_surf::desugar::desugar_program(decls);
    let invariants = crate::opaque::collect_opaque_invariants(&exprs);
    params.iter().any(|p| {
        matches!(&p.ty, Some(TypeExpr::Named(name, _))
            if invariants.iter().any(|inv| &inv.type_name == name))
    })
}

/// Run a user property that has an invariant-carrying opaque binder
/// through the injection-aware Tier C path. Returns the property status.
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

    let exprs = chelis_surf::desugar::desugar_program(decls);
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
                );
            }
        }
    }

    // The desugared property body + preconditions (Deep).
    let body_deep = chelis_surf::desugar::desugar_expr_only(body);
    let pre_deep: Vec<Expr> = preconditions
        .iter()
        .map(chelis_surf::desugar::desugar_expr_only)
        .collect();

    let mut rng = crate::opaque::GenRng::new(seed);
    let mut accepted = 0usize;
    let gen_budget = samples_needed.saturating_mul(100).max(200);

    while accepted < samples_needed {
        // Sample each binder. Opaque binders are generated invariant-valid
        // (the injected assumption); on starvation, report. Each binding
        // carries its Deep value expr and a JSON repr for counterexamples.
        let mut bindings: Vec<(String, Expr, serde_json::Value)> = Vec::new();
        for b in &binders {
            match b {
                Binder::Scalar { name, prim } => {
                    let v = sample_scalar(prim, &mut rng);
                    bindings.push((name.clone(), scalar_lit(prim, v), serde_json::json!(v)));
                }
                Binder::Tensor { name, dims } => {
                    let count: usize = dims.iter().product::<usize>().max(1);
                    let vals: Vec<f64> = (0..count).map(|_| rng_f64(&mut rng)).collect();
                    bindings.push((
                        name.clone(),
                        crate::opaque::tensor_value_expr_pub(dims, "f32", &vals),
                        serde_json::json!(vals),
                    ));
                }
                Binder::Opaque { name, inv } => {
                    let mut grng = crate::opaque::GenRng::new(rng.next_u64());
                    let producers = generation_producers(&exprs, &invariants, inv);
                    match crate::opaque::generate_binder(
                        inv,
                        &consts,
                        &module_source,
                        &producers,
                        &mut grng,
                        options.invariant_min_rate,
                        gen_budget,
                    ) {
                        Ok(generated) => {
                            let json = serde_json::to_value(&generated.env)
                                .unwrap_or(serde_json::json!(null));
                            bindings.push((name.clone(), generated.value_expr, json));
                        }
                        Err(diag) => {
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
                                );
                            }
                            return outcome(
                                property_name,
                                PropertyStatus::Unsupported,
                                0,
                                seed,
                                None,
                                Some(diag.message()),
                            );
                        }
                    }
                }
            }
        }

        // Precondition filter (composed with the injected invariant by the
        // generator already restricting opaque binders).
        if !pre_deep.is_empty() {
            match eval_bool_in_module(&exprs, &invariants, &bindings, &conjoin(&pre_deep)) {
                Ok(true) => {}
                Ok(false) => continue,
                Err(e) => return outcome_error(property_name, seed, e),
            }
        }

        accepted += 1;
        match eval_bool_in_module(&exprs, &invariants, &bindings, &body_deep) {
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
                );
            }
            Err(e) => return outcome_error(property_name, seed, e),
        }
    }

    outcome(
        property_name,
        PropertyStatus::Passed,
        samples_needed,
        seed,
        None,
        None,
    )
}

/// Build an injection-path [`PropertyOutcome`]. All injection outcomes are
/// Tier C (fuzz over generated invariant-valid binders) and carry
/// `injected: true`.
fn outcome(
    name: &str,
    status: PropertyStatus,
    samples: usize,
    seed: u64,
    counterexample: Option<serde_json::Value>,
    reason: Option<String>,
) -> PropertyOutcome {
    let tier = if matches!(status, PropertyStatus::Passed | PropertyStatus::Failed) {
        PropertyTier::Fuzz
    } else {
        PropertyTier::None
    };
    PropertyOutcome {
        name: name.to_string(),
        status,
        proof_tier: tier,
        samples,
        seed,
        counterexample,
        reason,
        injected: true,
    }
}

fn outcome_error(name: &str, seed: u64, reason: String) -> PropertyOutcome {
    outcome(name, PropertyStatus::Error, 0, seed, None, Some(reason))
}

enum Binder {
    Scalar {
        name: String,
        prim: String,
    },
    Tensor {
        name: String,
        dims: Vec<usize>,
    },
    Opaque {
        name: String,
        inv: crate::opaque::OpaqueInvariant,
    },
}

fn classify_binder(p: &Param, invariants: &[crate::opaque::OpaqueInvariant]) -> Option<Binder> {
    match p.ty.as_ref()? {
        TypeExpr::Named(name, _) => {
            if matches!(name.as_str(), "f32" | "f64" | "int32" | "int64" | "bool") {
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
        TypeExpr::Tensor(dims, precision, _) if matches!(precision.as_str(), "f32" | "f64") => {
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
            })
        }
        _ => None,
    }
}

/// Evaluate a boolean Deep expr with the given binder value bindings,
/// inside the defining module (so opaque construction/access is legal),
/// with the invariant metadata stripped (so a `sum`-bearing invariant does
/// not block IR lowering of the bound module).
fn eval_bool_in_module(
    exprs: &[Expr],
    invariants: &[crate::opaque::OpaqueInvariant],
    bindings: &[(String, Expr, serde_json::Value)],
    body: &Expr,
) -> Result<bool, String> {
    use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
    let probe = "__chelis_prop_probe";
    // Bind all binders via a let-chain around the body.
    let mut wrapped = body.clone();
    for (name, value, _) in bindings.iter().rev() {
        wrapped = node(
            "let",
            vec![node("bind", vec![sym(name), value.clone()]), wrapped],
        );
    }
    let probe_def = node("def", vec![sym(probe), wrapped]);
    // Inject into the module that defines the first opaque type (any will
    // do; binders are constructed via that module's ctors). If there are
    // none, append at top level.
    let type_name = invariants
        .first()
        .map(|i| i.type_name.as_str())
        .unwrap_or("");
    let stripped: Vec<Expr> = exprs.iter().map(strip_invariant_meta).collect();
    let program = inject_into_module(&stripped, type_name, probe_def);
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
    for name in referenced {
        // Try `name()` then `name`.
        for body in [node("app", vec![var_node(&name)]), var_node(&name)] {
            let probe = "__chelis_const_probe";
            let program = inject_first_module(&stripped, node("def", vec![sym(probe), body]));
            let source = chelis_deep::printer::print_canonical(&program);
            if let Ok(result) = chelis_compiler_api::compiler::eval_selected(
                EvalRequest {
                    source_kind: SourceKind::Deep,
                    source,
                    bindings: Default::default(),
                },
                &[probe.to_string()],
            ) && let [root] = result.roots.as_slice()
                && let ExecutionValue::Tensor { value } = &root.value
                && value.shape.is_empty()
                && value.data.len() == 1
            {
                env.insert(name.clone(), value.data[0]);
                break;
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

fn sample_scalar(prim: &str, rng: &mut crate::opaque::GenRng) -> f64 {
    match prim {
        "int32" | "int64" => (rng.next_u64() % 2001) as f64 - 1000.0,
        "bool" => (rng.next_u64() & 1) as f64,
        _ => rng_f64(rng),
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
    Expr::Atom(Atom::Symbol(s.to_string()), span0())
}
fn node(tag: &str, kids: Vec<Expr>) -> Expr {
    let mut elements = vec![sym(tag), Expr::Map(MetaMap::default(), span0())];
    elements.extend(kids);
    Expr::List(List { elements }, span0())
}
fn var_node(name: &str) -> Expr {
    node("var", vec![sym(name)])
}
fn typed_lit(prim: &str, value: Expr) -> Expr {
    let mut entries = MetaMap::default();
    entries
        .entries
        .push(("type".to_string(), node("t-prim", vec![sym(prim)])));
    Expr::List(
        List {
            elements: vec![sym("lit"), Expr::Map(entries, span0()), value],
        },
        span0(),
    )
}
fn scalar_lit(prim: &str, v: f64) -> Expr {
    match prim {
        "int32" => typed_lit("int32", Expr::Atom(Atom::Int(v as i64), span0())),
        "int64" => node(
            "cast",
            vec![
                typed_lit("int32", Expr::Atom(Atom::Int(v as i64), span0())),
                node("t-prim", vec![sym("int64")]),
            ],
        ),
        "bool" => typed_lit("bool", Expr::Atom(Atom::Bool(v != 0.0), span0())),
        "f64" => node(
            "cast",
            vec![
                typed_lit("f32", Expr::Atom(Atom::Float(v), span0())),
                node("t-prim", vec![sym("f64")]),
            ],
        ),
        _ => typed_lit("f32", Expr::Atom(Atom::Float(v), span0())),
    }
}
fn bool_lit(v: bool) -> Expr {
    typed_lit("bool", Expr::Atom(Atom::Bool(v), span0()))
}

fn list_tag(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::List(l, _) => match l.elements.first() {
            Some(Expr::Atom(Atom::Symbol(s), _)) => Some(s.as_str()),
            _ => None,
        },
        _ => None,
    }
}

fn child0_sym(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::List(l, _) if l.elements.len() >= 3 => match &l.elements[2] {
            Expr::Atom(Atom::Symbol(s), _) => Some(s.as_str()),
            _ => None,
        },
        _ => None,
    }
}

fn module_defines(expr: &Expr, type_name: &str) -> bool {
    if list_tag(expr) == Some("deftype") && child0_sym(expr) == Some(type_name) {
        return true;
    }
    if let Expr::List(l, _) = expr {
        return l.elements.iter().any(|c| module_defines(c, type_name));
    }
    false
}

fn inject_into_module(exprs: &[Expr], type_name: &str, def: Expr) -> Vec<Expr> {
    let mut out = Vec::with_capacity(exprs.len());
    let mut injected = false;
    for expr in exprs {
        if !injected
            && list_tag(expr) == Some("module")
            && (type_name.is_empty() || module_defines(expr, type_name))
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

fn inject_first_module(exprs: &[Expr], def: Expr) -> Vec<Expr> {
    inject_into_module(exprs, "", def)
}

fn strip_invariant_meta(expr: &Expr) -> Expr {
    match expr {
        Expr::List(list, span) => {
            let mut elements: Vec<Expr> = list.elements.iter().map(strip_invariant_meta).collect();
            if matches!(list.elements.first(), Some(Expr::Atom(Atom::Symbol(s), _)) if s == "deftype")
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
