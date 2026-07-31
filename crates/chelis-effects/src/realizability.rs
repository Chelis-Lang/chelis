//! Realizability inference for issue #912.
//!
//! Computes per-def lane assignment (Tensor vs Host) via transitive
//! fixed-point, in a separate lattice from algebraic effects. The inference
//! consults per-builtin declarations (`BUILTINS`), per-tag declarations
//! (`KNOWN_TAGS`), and a def-level precision check against backend capability.
//!
//! This module does NOT modify `enum Effect` or the mechanized `EffectRow`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use chelis_deep::ast::{Atom, Expr, List};
use chelis_types::known_tags::{LaneContribution, tag_lane_contribution};
use chelis_types::types::{Lane, Prim};
use chelis_types::{CheckedProgram, Realizability, builtin_decl};

// ─── Public types ────────────────────────────────────────────────────────────

/// Why a def routes to the host lane.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HostReason {
    HostOnlyBuiltin { name: String },
    ScalarTypedOp { builtin: String },
    PrecisionExceedsCapability { prim: Prim },
    StructuralForm { tag: String },
    TransitiveCaller { callee: String },
    UnrecognizedTag { tag: String },
}

/// Result of realizability inference.
#[derive(Debug, Clone)]
pub struct RealizabilityResult {
    pub lane_by_def: BTreeMap<String, Lane>,
    pub required_inputs_by_def: BTreeMap<String, BTreeSet<String>>,
    pub reasons_by_def: BTreeMap<String, Vec<HostReason>>,
}

// ─── Inference ───────────────────────────────────────────────────────────────

/// Infer per-def realizability for a checked program against a target's
/// tensor-capable prims.
pub fn infer_realizability(
    program: &CheckedProgram,
    target_prims: &[Prim],
) -> RealizabilityResult {
    let exprs = program.annotated_exprs();
    let type_env = program.type_env();

    // Collect top-level def names and their bodies.
    let mut def_bodies: HashMap<String, &Expr> = HashMap::new();
    let mut def_order: Vec<String> = Vec::new();
    for expr in exprs {
        if let Some((name, body)) = extract_def(expr) {
            def_bodies.insert(name.clone(), body);
            def_order.push(name);
        }
    }

    // Fixed-point: iterate until stable.
    let mut lane_by_def: HashMap<String, Lane> = HashMap::new();
    let mut reasons_by_def: HashMap<String, Vec<HostReason>> = HashMap::new();
    let mut required_inputs_by_def: HashMap<String, BTreeSet<String>> = HashMap::new();

    // Initialize all defs as Tensor.
    for name in &def_order {
        lane_by_def.insert(name.clone(), Lane::Tensor);
        reasons_by_def.insert(name.clone(), Vec::new());
        required_inputs_by_def.insert(name.clone(), BTreeSet::new());
    }

    // Def-level precision check: if the def's declared type has a prim
    // NOT in target_prims, it must route Host.
    let target_set: HashSet<Prim> = target_prims.iter().copied().collect();
    for name in &def_order {
        if let Some(ty_expr) = type_env.get(name) {
            for prim in extract_prims_from_type_expr(ty_expr) {
                if !target_set.contains(&prim) && prim != Prim::String {
                    lane_by_def.insert(name.clone(), Lane::Host);
                    reasons_by_def
                        .entry(name.clone())
                        .or_default()
                        .push(HostReason::PrecisionExceedsCapability { prim });
                    break;
                }
            }
        }
    }

    // Fixed-point iteration over call graph.
    let mut changed = true;
    let mut iteration = 0;
    while changed && iteration < 100 {
        changed = false;
        iteration += 1;

        for name in &def_order {
            if lane_by_def[name] == Lane::Host {
                continue; // Already Host, can't go back.
            }

            let body = match def_bodies.get(name) {
                Some(b) => *b,
                None => continue,
            };

            let mut reasons = Vec::new();
            let mut inputs = BTreeSet::new();
            let needs_host = expr_needs_host(
                body,
                &lane_by_def,
                &target_set,
                type_env,
                &mut reasons,
                &mut inputs,
            );

            if needs_host {
                lane_by_def.insert(name.clone(), Lane::Host);
                reasons_by_def.insert(name.clone(), reasons);
                changed = true;
            }
            // Accumulate inputs regardless of lane.
            required_inputs_by_def
                .entry(name.clone())
                .or_default()
                .extend(inputs);
        }
    }

    RealizabilityResult {
        lane_by_def: lane_by_def.into_iter().collect(),
        required_inputs_by_def: required_inputs_by_def.into_iter().collect(),
        reasons_by_def: reasons_by_def.into_iter().collect(),
    }
}

// ─── Expression walker ───────────────────────────────────────────────────────

fn expr_needs_host(
    expr: &Expr,
    lane_by_def: &HashMap<String, Lane>,
    target_prims: &HashSet<Prim>,
    type_env: &HashMap<String, Expr>,
    reasons: &mut Vec<HostReason>,
    inputs: &mut BTreeSet<String>,
) -> bool {
    match expr {
        Expr::Atom(Atom::Str(_), _) => {
            // String literals force host (the host runtime handles strings).
            true
        }
        Expr::Atom(_, _) | Expr::Map(_, _) => false,
        Expr::MetaExpr(meta, _) => expr_needs_host(
            &meta.expr,
            lane_by_def,
            target_prims,
            type_env,
            reasons,
            inputs,
        ),
        Expr::List(list, _) => {
            list_needs_host(list, lane_by_def, target_prims, type_env, reasons, inputs)
        }
    }
}

fn list_needs_host(
    list: &List,
    lane_by_def: &HashMap<String, Lane>,
    target_prims: &HashSet<Prim>,
    type_env: &HashMap<String, Expr>,
    reasons: &mut Vec<HostReason>,
    inputs: &mut BTreeSet<String>,
) -> bool {
    let tag = get_tag(list);
    let children = get_children(list);

    // Check tag against KNOWN_TAGS.
    if let Some(tag_str) = tag {
        match tag_lane_contribution(tag_str) {
            Some(LaneContribution::ForcesHost) => {
                reasons.push(HostReason::StructuralForm {
                    tag: tag_str.to_string(),
                });
                return true;
            }
            Some(LaneContribution::Propagates) => {
                // Fall through to check children.
            }
            None => {
                // Unknown tag → Host + reason.
                reasons.push(HostReason::UnrecognizedTag {
                    tag: tag_str.to_string(),
                });
                return true;
            }
        }
    }

    let tag_str = tag.unwrap_or("");

    // Uppercase var references (ADT constructors) → Host.
    if tag_str == "var" {
        if let Some(name) = children.first().and_then(symbol_name) {
            if name.chars().next().is_some_and(|c| c.is_uppercase()) {
                reasons.push(HostReason::StructuralForm {
                    tag: format!("var({})", name),
                });
                return true;
            }
            // Track free variables as potential inputs.
            if !lane_by_def.contains_key(name) && !is_builtin(name) {
                inputs.insert(name.to_string());
            }
        }
    }

    // App: check the callee builtin.
    if tag_str == "app" {
        if let Some(callee_name) = app_callee_name(list) {
            // Uppercase callee (ADT constructor call) → Host.
            if callee_name.chars().next().is_some_and(|c| c.is_uppercase()) {
                reasons.push(HostReason::StructuralForm {
                    tag: format!("app({})", callee_name),
                });
                return true;
            }

            // Check builtin realizability.
            if let Some(decl) = builtin_decl(&callee_name) {
                match decl.realizability {
                    Realizability::HostOnly => {
                        reasons.push(HostReason::HostOnlyBuiltin {
                            name: callee_name.to_string(),
                        });
                        return true;
                    }
                    Realizability::TensorAtTensorType => {
                        // Check if the application's type is scalar.
                        if app_type_is_scalar(list) {
                            reasons.push(HostReason::ScalarTypedOp {
                                builtin: callee_name.to_string(),
                            });
                            return true;
                        }
                    }
                    Realizability::Universal => {}
                }
            }

            // Check transitive callee lane.
            if let Some(Lane::Host) = lane_by_def.get(&callee_name) {
                reasons.push(HostReason::TransitiveCaller {
                    callee: callee_name.to_string(),
                });
                return true;
            }
        }
    }

    // Walk children (skip metadata map at element 1 if present).
    let meta_idx = if matches!(list.elements.get(1), Some(Expr::Map(_, _))) {
        Some(1usize)
    } else {
        None
    };

    for (idx, child) in list.elements.iter().enumerate() {
        if Some(idx) == meta_idx {
            continue;
        }
        if expr_needs_host(child, lane_by_def, target_prims, type_env, reasons, inputs) {
            return true;
        }
    }

    false
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

fn get_tag(list: &List) -> Option<&str> {
    list.elements.first().and_then(|e| match e {
        Expr::Atom(Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    })
}

fn get_children(list: &List) -> &[Expr] {
    if list.elements.len() > 2
        && matches!(list.elements.get(1), Some(Expr::Map(_, _)))
    {
        &list.elements[2..]
    } else if list.elements.len() > 1 {
        &list.elements[1..]
    } else {
        &[]
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn is_builtin(name: &str) -> bool {
    builtin_decl(name).is_some()
}

fn app_callee_name(list: &List) -> Option<String> {
    let children = get_children(list);
    let callee = children.first()?;
    match callee {
        Expr::List(callee_list, _) if get_tag(callee_list) == Some("var") => {
            get_children(callee_list)
                .first()
                .and_then(symbol_name)
                .map(|s| s.to_string())
        }
        _ => None,
    }
}

/// Check if an app expression's type metadata indicates a scalar (t-prim) type.
fn app_type_is_scalar(list: &List) -> bool {
    // The type annotation is in the metadata map at element 1,
    // under the "type" key.
    if let Some(Expr::Map(meta, _)) = list.elements.get(1) {
        for (key, value) in &meta.entries {
            if key == "type" {
                if let Expr::List(ty_list, _) = value {
                    return get_tag(ty_list) == Some("t-prim");
                }
            }
        }
    }
    false
}

/// Extract a def's name and body from a top-level expression.
fn extract_def(expr: &Expr) -> Option<(String, &Expr)> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("def") {
        return None;
    }
    let children = get_children(list);
    let name = children.first().and_then(symbol_name)?;
    let body = children.get(1)?;
    // Skip fn-typed defs (they're callable, not roots).
    if let Expr::List(body_list, _) = body {
        if get_tag(body_list) == Some("fn") {
            // Still register the def for transitive analysis, but the body
            // is the fn body, not the fn itself.
            let fn_children = get_children(body_list);
            let fn_body = fn_children.get(1).unwrap_or(body);
            return Some((name.to_string(), fn_body));
        }
    }
    Some((name.to_string(), body))
}

/// Extract all Prim values referenced in a type expression.
fn extract_prims_from_type_expr(expr: &Expr) -> Vec<Prim> {
    let mut prims = Vec::new();
    extract_prims_recursive(expr, &mut prims);
    prims
}

fn extract_prims_recursive(expr: &Expr, out: &mut Vec<Prim>) {
    match expr {
        Expr::Atom(Atom::Symbol(s), _) => {
            if let Some(prim) = Prim::parse_name(s) {
                out.push(prim);
            }
        }
        Expr::List(list, _) => {
            // Check for (t-prim {} <prim-name>) or (t-tensor {} dims... prim)
            for elem in &list.elements {
                extract_prims_recursive(elem, out);
            }
        }
        Expr::MetaExpr(meta, _) => {
            extract_prims_recursive(&meta.expr, out);
        }
        _ => {}
    }
}
