//! Realizability inference for issue #912.
//!
//! Computes per-def lane assignment (Tensor vs Host) via transitive
//! fixed-point, in a separate lattice from algebraic effects. The inference
//! consults per-builtin declarations (`BUILTINS`), the exhaustive typed
//! `DeepTag` lane disposition, and a def-level precision check against backend
//! capability.
//!
//! This module does NOT modify `enum Effect` or the mechanized `EffectRow`.

use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet};

use chelis_deep::{
    DeepTag,
    ast::{Atom, Expr, List},
};
use chelis_types::known_tags::{LaneContribution, deep_tag_lane_contribution};
use chelis_types::manifest::HostReason;
use chelis_types::types::{Lane, Prim};
use chelis_types::{CheckedProgram, Realizability, builtin_decl};

// ─── Public types ────────────────────────────────────────────────────────────

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
pub fn infer_realizability(program: &CheckedProgram, target_prims: &[Prim]) -> RealizabilityResult {
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
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        Expr::Node(node, span) => {
            let bridged = Expr::List(node.to_list(*span), *span);
            expr_needs_host(
                &bridged,
                lane_by_def,
                target_prims,
                type_env,
                reasons,
                inputs,
            )
        }
        // Fail-closed (#1086): a headless list or an unrecognized form has no
        // tag to classify, so it must not silently route Tensor. Force Host and
        // record a reason, matching the typed DeepTag disposition's raw-form
        // boundary and the #731/#908 exhaustive-disposition principle. These
        // forms do not survive
        // `chelis check` today (they score < 1 as UnknownForm), so this is
        // defensive alignment with the stated contract rather than a live
        // wrong-answer path — but a silent `false` here is exactly the
        // fail-open default those contracts exist to eliminate.
        Expr::BareList(_, _) => {
            reasons.push(HostReason::UnrecognizedTag {
                tag: "<bare-list>".to_string(),
            });
            true
        }
        Expr::UnknownForm(_) => {
            reasons.push(HostReason::UnrecognizedTag {
                tag: "<unknown-form>".to_string(),
            });
            true
        }
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

    let Some(tag) = tag else {
        // Fail-closed (#1086, #1080): an empty legacy List or a List whose head
        // is not a decoded DeepTag has no typed disposition. Do not re-decode a
        // raw Name here; ingress owns that boundary.
        reasons.push(HostReason::UnrecognizedTag {
            tag: list
                .unknown_tag_symbol()
                .unwrap_or("<untagged-list>")
                .to_string(),
        });
        return true;
    };

    match deep_tag_lane_contribution(tag) {
        LaneContribution::ForcesHost => {
            reasons.push(HostReason::StructuralForm {
                tag: tag.as_str().to_string(),
            });
            return true;
        }
        LaneContribution::Propagates => {
            // Fall through to check children.
        }
    }

    // Uppercase var references (ADT constructors) → Host.
    if tag == DeepTag::Var
        && let Some(name) = children.first().and_then(symbol_name)
    {
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

    // App: check the callee builtin.
    if tag == DeepTag::App
        && let Some(callee_name) = app_callee_name(list)
    {
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

fn get_tag(list: &List) -> Option<DeepTag> {
    list.tag()
}

fn get_children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 && matches!(list.elements.get(1), Some(Expr::Map(_, _))) {
        &list.elements[2..]
    } else if list.elements.len() > 1 {
        &list.elements[1..]
    } else {
        &[]
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(s), _) => Some(s.as_str()),
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
        Expr::List(callee_list, _) if get_tag(callee_list) == Some(DeepTag::Var) => {
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
            if key == "type"
                && let Expr::List(ty_list, _) = value
            {
                return get_tag(ty_list) == Some(DeepTag::TPrim);
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
    if get_tag(list) != Some(DeepTag::Def) {
        return None;
    }
    let children = get_children(list);
    let name = children.first().and_then(symbol_name)?;
    let body = children.get(1)?;
    // Skip fn-typed defs (they're callable, not roots).
    if let Expr::List(body_list, _) = body
        && get_tag(body_list) == Some(DeepTag::Fn)
    {
        // Still register the def for transitive analysis, but the body
        // is the fn body, not the fn itself.
        let fn_children = get_children(body_list);
        let fn_body = fn_children.get(1).unwrap_or(body);
        return Some((name.to_string(), fn_body));
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
        Expr::Atom(Atom::Name(s), _) => {
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

// ─── Manifest computation ────────────────────────────────────────────────────

use chelis_types::manifest::{RootEntry, RootManifest};

/// Compute the root manifest from a checked program and its realizability result.
/// This walks top-level non-fn defs, expands tuples/ADTs into dotted names,
/// and populates each entry from the realizability result keyed on originating def.
pub fn compute_root_manifest(
    program: &CheckedProgram,
    realizability: &RealizabilityResult,
) -> RootManifest {
    let exprs = program.annotated_exprs();
    let type_env = program.type_env();
    let mut entries = Vec::new();

    for expr in exprs {
        collect_manifest_entries(expr, type_env, realizability, &mut entries);
    }

    RootManifest { entries }
}

fn collect_manifest_entries(
    expr: &Expr,
    type_env: &HashMap<String, Expr>,
    realizability: &RealizabilityResult,
    out: &mut Vec<RootEntry>,
) {
    let Expr::List(list, _) = expr else { return };
    let tag = get_tag(list);

    if tag == Some(DeepTag::Module) {
        for child in get_children(list).iter().skip(1) {
            collect_manifest_entries(child, type_env, realizability, out);
        }
        return;
    }

    if tag != Some(DeepTag::Def) {
        return;
    }

    let children = get_children(list);
    let Some(name) = children.first().and_then(symbol_name) else {
        return;
    };
    let body = children.get(1);

    // Skip fn-typed defs (they're callable, not roots).
    if let Some(Expr::List(body_list, _)) = body
        && get_tag(body_list) == Some(DeepTag::Fn)
    {
        return;
    }

    let lane = realizability
        .lane_by_def
        .get(name)
        .copied()
        .unwrap_or(Lane::Host);
    let required_inputs = realizability
        .required_inputs_by_def
        .get(name)
        .cloned()
        .unwrap_or_default();
    let reasons: Vec<HostReason> = realizability
        .reasons_by_def
        .get(name)
        .cloned()
        .unwrap_or_default();

    let ty = type_env
        .get(name)
        .or_else(|| body.and_then(expr_type_metadata))
        .cloned()
        .unwrap_or_else(|| {
            Expr::Atom(
                Atom::Name("unknown".to_string()),
                chelis_deep::Span::new(0, 0),
            )
        });

    // TODO: expand tuples/ADTs into dotted names (Task 7 full implementation).
    // For now, emit a single entry per def.
    out.push(RootEntry {
        name: name.to_string(),
        def_name: name.to_string(),
        ty,
        lane,
        required_inputs,
        reasons,
    });
}

/// Extract type metadata from an expression's metadata map.
fn expr_type_metadata(expr: &Expr) -> Option<&Expr> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if let Some(Expr::Map(meta, _)) = list.elements.get(1) {
        for (key, value) in &meta.entries {
            if key == "type" {
                return Some(value);
            }
        }
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::ast::MetaMap;
    use chelis_types::types::Prim;

    fn check_program_from_source(source: &str) -> CheckedProgram {
        let decls = chelis_surf::parser::parse_str(source).expect("parse");
        let deep = chelis_macros::expand_program(
            &chelis_surf::desugar::desugar_program(&decls),
            &chelis_macros::ExpansionOptions::default(),
        )
        .expect("desugar")
        .into_exprs();
        let checked = chelis_types::check_ir_program(&deep).expect("typecheck");
        crate::check_program(&checked).expect("effects")
    }

    const C_PRIMS: &[Prim] = &[
        Prim::F32,
        Prim::Bool,
        Prim::Bf16,
        Prim::F16,
        Prim::Int32,
        Prim::Int64,
    ];
    const EVAL_PRIMS: &[Prim] = &[
        Prim::F32,
        Prim::F64,
        Prim::Bool,
        Prim::Bf16,
        Prim::F16,
        Prim::Int32,
        Prim::Int64,
    ];

    #[test]
    fn host_only_builtin_routes_host() {
        let checked =
            check_program_from_source("a: List[int32] = [cast(1, int32)]\nresult = concat(a, a)\n");
        let result = infer_realizability(&checked, C_PRIMS);
        assert_eq!(result.lane_by_def.get("result"), Some(&Lane::Host));
    }

    #[test]
    fn pure_tensor_def_routes_tensor() {
        let checked =
            check_program_from_source("x = add(to_tensor([1.0, 2.0]), to_tensor([3.0, 4.0]))\n");
        let result = infer_realizability(&checked, C_PRIMS);
        // to_tensor is HostOnly, so this routes Host
        assert_eq!(result.lane_by_def.get("x"), Some(&Lane::Host));
    }

    #[test]
    fn f64_def_routes_host_for_c_target() {
        let checked =
            check_program_from_source("def f(a: tensor[4, f64]) -> tensor[4, f64] = mul(a, a)\n");
        let result = infer_realizability(&checked, C_PRIMS);
        assert_eq!(result.lane_by_def.get("f"), Some(&Lane::Host));
    }

    #[test]
    fn f64_def_routes_tensor_for_eval_target() {
        let checked =
            check_program_from_source("def f(a: tensor[4, f64]) -> tensor[4, f64] = mul(a, a)\n");
        let result = infer_realizability(&checked, EVAL_PRIMS);
        assert_eq!(result.lane_by_def.get("f"), Some(&Lane::Tensor));
    }

    #[test]
    fn transitive_host_propagates() {
        let checked = check_program_from_source(
            "a: List[int32] = [cast(1, int32)]\nb = concat(a, a)\nc = len(b)\n",
        );
        let result = infer_realizability(&checked, C_PRIMS);
        // concat is HostOnly → b is Host
        assert_eq!(result.lane_by_def.get("b"), Some(&Lane::Host));
        // len is also HostOnly
        assert_eq!(result.lane_by_def.get("c"), Some(&Lane::Host));
    }

    // #1084: a genuine Tensor-lane positive under the C target. The pre-existing
    // `pure_tensor_def_routes_tensor` above asserts `Host` (its body uses the
    // HostOnly `to_tensor`), so before this test nothing showed any def reaching
    // the Tensor lane on the C target — the outcome the C DAG path exists to
    // serve. An f32 elementwise def is C-capable and non-HostOnly.
    #[test]
    fn f32_tensor_def_routes_tensor_for_c_target() {
        let checked =
            check_program_from_source("def f(a: tensor[4, f32]) -> tensor[4, f32] = mul(a, a)\n");
        let result = infer_realizability(&checked, C_PRIMS);
        assert_eq!(result.lane_by_def.get("f"), Some(&Lane::Tensor));
    }

    // #1084: first coverage of `compute_root_manifest`, which had none. A Host
    // value root must appear in the manifest with its lane recorded.
    #[test]
    fn compute_root_manifest_lists_value_roots_with_lanes() {
        let checked =
            check_program_from_source("a: List[int32] = [cast(1, int32)]\nresult = concat(a, a)\n");
        let realizability = infer_realizability(&checked, C_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        let names: Vec<&str> = manifest.entries.iter().map(|e| e.name.as_str()).collect();
        assert!(
            names.contains(&"result"),
            "manifest must list the `result` value root; got {names:?}"
        );
        let result_entry = manifest
            .entries
            .iter()
            .find(|e| e.name == "result")
            .expect("result entry");
        assert_eq!(
            result_entry.lane,
            Lane::Host,
            "the concat-fed `result` root routes Host"
        );
    }

    // #1086: a headless list must fail closed to Host with a recorded reason,
    // not silently route Tensor.
    #[test]
    fn bare_list_routes_host_fail_closed() {
        let expr = Expr::BareList(vec![], chelis_deep::Span::new(0, 0));
        let lane_by_def: HashMap<String, Lane> = HashMap::new();
        let target: HashSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: HashMap<String, Expr> = HashMap::new();
        let mut reasons = Vec::new();
        let mut inputs = BTreeSet::new();
        let needs_host = expr_needs_host(
            &expr,
            &lane_by_def,
            &target,
            &type_env,
            &mut reasons,
            &mut inputs,
        );
        assert!(needs_host, "BareList must route Host (fail-closed)");
        assert!(
            reasons
                .iter()
                .any(|r| matches!(r, HostReason::UnrecognizedTag { .. })),
            "BareList Host routing must record a reason, not be silent; got {reasons:?}"
        );
    }

    // #1086: an unrecognized form must fail closed to Host with a reason.
    #[test]
    fn unknown_form_routes_host_fail_closed() {
        use chelis_deep::ast::{MetaMap, UnknownFormData};
        let expr = Expr::UnknownForm(Box::new(UnknownFormData {
            head: "mystery".to_string(),
            meta: MetaMap::default(),
            children: vec![],
            span: chelis_deep::Span::new(0, 0),
        }));
        let lane_by_def: HashMap<String, Lane> = HashMap::new();
        let target: HashSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: HashMap<String, Expr> = HashMap::new();
        let mut reasons = Vec::new();
        let mut inputs = BTreeSet::new();
        let needs_host = expr_needs_host(
            &expr,
            &lane_by_def,
            &target,
            &type_env,
            &mut reasons,
            &mut inputs,
        );
        assert!(needs_host, "UnknownForm must route Host (fail-closed)");
        assert!(
            !reasons.is_empty(),
            "UnknownForm Host routing must record a reason, not be silent"
        );
    }

    // #1086 completeness: the same fail-open existed one level down in
    // `list_needs_host` — a list whose head is not a Deep tag (`get_tag` None)
    // fell through to Tensor with no reason. It must fail closed too.
    #[test]
    fn untagged_list_routes_host_fail_closed() {
        let list = List {
            elements: vec![Expr::Atom(Atom::Int(0), chelis_deep::Span::new(0, 0))],
        };
        let expr = Expr::List(list, chelis_deep::Span::new(0, 0));
        let lane_by_def: HashMap<String, Lane> = HashMap::new();
        let target: HashSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: HashMap<String, Expr> = HashMap::new();
        let mut reasons = Vec::new();
        let mut inputs = BTreeSet::new();
        let needs_host = expr_needs_host(
            &expr,
            &lane_by_def,
            &target,
            &type_env,
            &mut reasons,
            &mut inputs,
        );
        assert!(needs_host, "an untagged non-empty list must route Host");
        assert!(
            !reasons.is_empty(),
            "untagged-list Host routing must record a reason, not be silent"
        );
    }

    #[test]
    fn empty_legacy_list_routes_host_fail_closed() {
        let expr = Expr::List(List { elements: vec![] }, chelis_deep::Span::new(0, 0));
        let lane_by_def: HashMap<String, Lane> = HashMap::new();
        let target: HashSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: HashMap<String, Expr> = HashMap::new();
        let mut reasons = Vec::new();
        let mut inputs = BTreeSet::new();

        assert!(expr_needs_host(
            &expr,
            &lane_by_def,
            &target,
            &type_env,
            &mut reasons,
            &mut inputs,
        ));
        assert_eq!(
            reasons,
            [HostReason::UnrecognizedTag {
                tag: "<untagged-list>".to_string(),
            }]
        );
    }

    #[test]
    fn raw_name_that_spells_a_known_tag_routes_host_fail_closed() {
        let span = chelis_deep::Span::new(0, 0);
        let expr = Expr::List(
            List {
                elements: vec![Expr::Atom(Atom::Name("app".to_string()), span)],
            },
            span,
        );
        let lane_by_def: HashMap<String, Lane> = HashMap::new();
        let target: HashSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: HashMap<String, Expr> = HashMap::new();
        let mut reasons = Vec::new();
        let mut inputs = BTreeSet::new();

        assert!(expr_needs_host(
            &expr,
            &lane_by_def,
            &target,
            &type_env,
            &mut reasons,
            &mut inputs,
        ));
        assert_eq!(
            reasons,
            [HostReason::UnrecognizedTag {
                tag: "app".to_string(),
            }]
        );
    }

    // #1080 red-team finding: testing the disposition table alone did not
    // prove that the production realizability walker consulted it. Keep the
    // Host-forcing tag below a propagating parent so this test covers both the
    // stamped Node bridge and the recursive consumer connection.
    #[test]
    fn stamped_host_forcing_tag_routes_host_through_propagating_parent() {
        let span = chelis_deep::Span::new(0, 0);
        let record = Expr::node(
            DeepTag::Record,
            MetaMap::default(),
            vec![Expr::Atom(Atom::Name("R".to_string()), span)],
            span,
        );
        let expr = Expr::node(DeepTag::Block, MetaMap::default(), vec![record], span);
        let lane_by_def: HashMap<String, Lane> = HashMap::new();
        let target: HashSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: HashMap<String, Expr> = HashMap::new();
        let mut reasons = Vec::new();
        let mut inputs = BTreeSet::new();

        assert!(expr_needs_host(
            &expr,
            &lane_by_def,
            &target,
            &type_env,
            &mut reasons,
            &mut inputs,
        ));
        assert_eq!(
            reasons,
            [HostReason::StructuralForm {
                tag: "record".to_string(),
            }]
        );
        assert!(inputs.is_empty());
    }

    // Negative parity for the production connection above: propagating tags
    // must not invent a Host requirement or reason.
    #[test]
    fn stamped_propagating_tags_preserve_tensor_lane() {
        let span = chelis_deep::Span::new(0, 0);
        let literal = Expr::node(
            DeepTag::Lit,
            MetaMap::default(),
            vec![Expr::Atom(Atom::Int(1), span)],
            span,
        );
        let expr = Expr::node(DeepTag::Block, MetaMap::default(), vec![literal], span);
        let lane_by_def: HashMap<String, Lane> = HashMap::new();
        let target: HashSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: HashMap<String, Expr> = HashMap::new();
        let mut reasons = Vec::new();
        let mut inputs = BTreeSet::new();

        assert!(!expr_needs_host(
            &expr,
            &lane_by_def,
            &target,
            &type_env,
            &mut reasons,
            &mut inputs,
        ));
        assert!(reasons.is_empty());
        assert!(inputs.is_empty());
    }

    // #1084 / red-team Finding 1: the Host case alone cannot prove the manifest
    // reads the real lane, because the lookup falls back to `Lane::Host`. A pure
    // Tensor value root must be recorded as Tensor — this kills the mutation
    // where the lane lookup is broken and every entry collapses to the default.
    #[test]
    fn compute_root_manifest_records_tensor_lane_not_just_host_default() {
        let checked = check_program_from_source("x = cast(1, int32)\n");
        let realizability = infer_realizability(&checked, C_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        let x = manifest
            .entries
            .iter()
            .find(|e| e.name == "x")
            .expect("x entry");
        assert_eq!(
            x.lane,
            Lane::Tensor,
            "a pure Tensor value root must record Lane::Tensor, not the Host fallback"
        );
        assert_eq!(manifest.tensor_root_names(), vec!["x"]);
    }
}
