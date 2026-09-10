//! Realizability inference for issue #912.
//!
//! Computes per-def lane assignment (Tensor vs Host) via transitive
//! fixed-point, in a separate lattice from algebraic effects. The inference
//! consults per-builtin declarations (`BUILTINS`), the exhaustive typed
//! `DeepTag` lane disposition, and a def-level precision check against backend
//! capability.
//!
//! This module does NOT modify `enum Effect` or the mechanized `EffectRow`.

use chelis_unord::UnordSet;
use std::collections::{BTreeMap, BTreeSet};

use chelis_deep::{
    DeepTag,
    ast::{Atom, Expr, List, Metadata},
};
use chelis_types::known_tags::{LaneContribution, deep_tag_lane_contribution};
use chelis_types::manifest::{HostReason, RootPathStep};
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
    let mut def_bodies: BTreeMap<String, &Expr> = BTreeMap::new();
    let mut declared_types: BTreeMap<String, &Expr> = BTreeMap::new();
    let mut def_order: Vec<String> = Vec::new();
    let mut function_defs = BTreeSet::new();
    for expr in exprs {
        collect_top_level_defs(
            expr,
            type_env,
            &mut def_bodies,
            &mut declared_types,
            &mut def_order,
            &mut function_defs,
        );
    }

    // Fixed-point: iterate until stable.
    let mut lane_by_def: BTreeMap<String, Lane> = BTreeMap::new();
    let mut reasons_by_def: BTreeMap<String, Vec<HostReason>> = BTreeMap::new();
    let mut required_inputs_by_def: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();

    // Initialize all defs as Tensor.
    for name in &def_order {
        lane_by_def.insert(name.clone(), Lane::Tensor);
        reasons_by_def.insert(name.clone(), Vec::new());
        required_inputs_by_def.insert(name.clone(), BTreeSet::new());
    }

    // Top-level dependency edges are distinct from function parameters. A
    // self-referential tensor binding (`x: tensor[...] = x`) names a runtime
    // input; another root which references `x` inherits that input. Function
    // parameters are absent from this top-level set and therefore never leak
    // into a caller under their declaration-local names.
    let top_level_names = def_order.iter().cloned().collect::<BTreeSet<_>>();
    let parameter_names_by_def = program
        .signature_inference()
        .functions
        .iter()
        .map(|(name, function)| {
            (
                name.clone(),
                function
                    .params
                    .iter()
                    .map(|param| param.name.clone())
                    .collect::<BTreeSet<_>>(),
            )
        })
        .collect::<BTreeMap<_, _>>();
    let nullary_function_defs = program
        .signature_inference()
        .functions
        .iter()
        .filter(|(_, function)| function.params.is_empty())
        .map(|(name, _)| name.clone())
        .collect::<BTreeSet<_>>();
    let mut dependencies_by_def = BTreeMap::<String, BTreeSet<String>>::new();
    for name in &def_order {
        let mut dependencies = BTreeSet::new();
        let mut references_self = false;
        let initial_bound = parameter_names_by_def
            .get(name)
            .cloned()
            .unwrap_or_default();
        if let Some(body) = def_bodies.get(name) {
            collect_top_level_dependencies(
                body,
                name,
                &top_level_names,
                &initial_bound,
                &mut dependencies,
                &mut references_self,
            );
        }
        if references_self
            && !function_defs.contains(name)
            && declared_types
                .get(name)
                .copied()
                .is_some_and(type_expr_contains_tensor)
        {
            required_inputs_by_def
                .entry(name.clone())
                .or_default()
                .insert(name.clone());
        }
        dependencies_by_def.insert(name.clone(), dependencies);
    }

    // Def-level precision check: if the def's declared type has a prim
    // NOT in target_prims, it must route Host.
    let target_set: UnordSet<Prim> = target_prims.iter().copied().collect();
    for name in &def_order {
        // Nullary arrow-form defs are observation thunks, not DAG value
        // bindings. Their applied value is produced by the host evaluator;
        // assigning one to Tensor would promise a named DAG root that does
        // not exist and would turn the #947 surfacing path into [05-UNS-1].
        if nullary_function_defs.contains(name) {
            lane_by_def.insert(name.clone(), Lane::Host);
            reasons_by_def
                .entry(name.clone())
                .or_default()
                .push(HostReason::StructuralForm {
                    tag: "fn(nullary-root)".to_string(),
                });
        }
        if let Some(ty_expr) = declared_types.get(name).copied() {
            // The tensor lane's named-root ABI is tensor-valued. Scalar
            // functions and scalar value bindings are realized by the host
            // evaluator/emitter even when every operation inside them is a
            // `Universal` builtin. Checked application nodes do not all carry
            // result-type metadata, so deriving this from the declaration is
            // the structural authority; relying only on `app_type_is_scalar`
            // misroutes `def f(x: f32) -> f32 = atan(x)` and its callers.
            if type_expr_has_primitive_result(ty_expr) {
                lane_by_def.insert(name.clone(), Lane::Host);
                reasons_by_def
                    .entry(name.clone())
                    .or_default()
                    .push(HostReason::StructuralForm {
                        tag: "scalar-result".to_string(),
                    });
            }
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
            let body = match def_bodies.get(name) {
                Some(b) => *b,
                None => continue,
            };

            let mut reasons = Vec::new();
            let mut inputs = BTreeSet::new();
            let mut needs_host = expr_needs_host(
                body,
                &lane_by_def,
                &target_set,
                type_env,
                &mut reasons,
                &mut inputs,
            );
            for dependency in &dependencies_by_def[name] {
                if lane_by_def.get(dependency) == Some(&Lane::Host) {
                    reasons.push(HostReason::TransitiveCaller {
                        callee: dependency.clone(),
                    });
                    needs_host = true;
                }
            }

            if needs_host {
                if lane_by_def[name] != Lane::Host {
                    lane_by_def.insert(name.clone(), Lane::Host);
                    changed = true;
                }
                let recorded = reasons_by_def.entry(name.clone()).or_default();
                for reason in reasons {
                    if !recorded.contains(&reason) {
                        recorded.push(reason);
                    }
                }
            }
            // Accumulate inputs regardless of lane.
            required_inputs_by_def
                .entry(name.clone())
                .or_default()
                .extend(inputs);
        }
    }

    // Close runtime-input demands over top-level references. This is a second
    // monotone fixed point because lane routing and input liveness are
    // independent facts. It is deliberately over declaration dependencies,
    // not raw names in a function body, so a callee's parameter spelling can
    // never become a caller input.
    let mut inputs_changed = true;
    while inputs_changed {
        inputs_changed = false;
        for name in &def_order {
            let mut closed = required_inputs_by_def[name].clone();
            for dependency in &dependencies_by_def[name] {
                if let Some(inputs) = required_inputs_by_def.get(dependency) {
                    closed.extend(inputs.iter().cloned());
                }
            }
            if closed.len() != required_inputs_by_def[name].len() {
                required_inputs_by_def.insert(name.clone(), closed);
                inputs_changed = true;
            }
        }
    }

    RealizabilityResult {
        lane_by_def: lane_by_def.into_iter().collect(),
        required_inputs_by_def: required_inputs_by_def.into_iter().collect(),
        reasons_by_def: reasons_by_def.into_iter().collect(),
    }
}

/// Collect declaration facts through the checked program's optional module
/// wrapper. Surf package and compiled-context paths retain `(module ...)` at
/// this boundary, so a flat-only scan makes the manifest entries and their
/// realizability maps disagree.
fn collect_top_level_defs<'a>(
    expr: &'a Expr,
    type_env: &'a BTreeMap<String, Expr>,
    def_bodies: &mut BTreeMap<String, &'a Expr>,
    declared_types: &mut BTreeMap<String, &'a Expr>,
    def_order: &mut Vec<String>,
    function_defs: &mut BTreeSet<String>,
) {
    let Some((tag, children)) = tagged_children(expr) else {
        return;
    };

    if tag == DeepTag::Module {
        for child in children.iter().skip(1) {
            collect_top_level_defs(
                child,
                type_env,
                def_bodies,
                declared_types,
                def_order,
                function_defs,
            );
        }
        return;
    }

    let Some((name, body)) = extract_def(expr) else {
        return;
    };
    if children
        .get(1)
        .and_then(tagged_children)
        .is_some_and(|(body_tag, _)| body_tag == DeepTag::Fn)
    {
        function_defs.insert(name.clone());
    }
    if let Some(ty) = type_env
        .get(&name)
        .or_else(|| expr_type_metadata(expr))
        .or_else(|| expr_type_metadata(body))
    {
        declared_types.insert(name.clone(), ty);
    }
    def_bodies.insert(name.clone(), body);
    def_order.push(name);
}

// ─── Expression walker ───────────────────────────────────────────────────────

fn expr_needs_host(
    expr: &Expr,
    lane_by_def: &BTreeMap<String, Lane>,
    target_prims: &UnordSet<Prim>,
    type_env: &BTreeMap<String, Expr>,
    reasons: &mut Vec<HostReason>,
    inputs: &mut BTreeSet<String>,
) -> bool {
    match expr {
        Expr::Atom(Atom::Str(_), _) => {
            // String literals force host (the host runtime handles strings).
            true
        }
        Expr::Atom(_, _) | Expr::Map(_, _) => false,
        // chelis#1082: the stamped carrier is the production path. Observe
        // its decoded tag, metadata, and children directly; reconstructing a
        // legacy List here makes manifest correctness depend on #908 debt.
        Expr::Node(node, _) => tagged_needs_host(
            node.tag(),
            Some(node.meta()),
            node.children_slice(),
            &LaneWalkContext {
                lane_by_def,
                target_prims,
                type_env,
            },
            reasons,
            inputs,
        ),
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
    lane_by_def: &BTreeMap<String, Lane>,
    target_prims: &UnordSet<Prim>,
    type_env: &BTreeMap<String, Expr>,
    reasons: &mut Vec<HostReason>,
    inputs: &mut BTreeSet<String>,
) -> bool {
    let tag = get_tag(list);
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

    tagged_needs_host(
        tag,
        list_meta(list),
        get_children(list),
        &LaneWalkContext {
            lane_by_def,
            target_prims,
            type_env,
        },
        reasons,
        inputs,
    )
}

struct LaneWalkContext<'a> {
    lane_by_def: &'a BTreeMap<String, Lane>,
    target_prims: &'a UnordSet<Prim>,
    type_env: &'a BTreeMap<String, Expr>,
}

fn tagged_needs_host(
    tag: DeepTag,
    meta: Option<&Metadata>,
    children: &[Expr],
    context: &LaneWalkContext<'_>,
    reasons: &mut Vec<HostReason>,
    inputs: &mut BTreeSet<String>,
) -> bool {
    let mut needs_host = false;

    match deep_tag_lane_contribution(tag) {
        LaneContribution::ForcesHost => {
            reasons.push(HostReason::StructuralForm {
                tag: tag.as_str().to_string(),
            });
            needs_host = true;
        }
        LaneContribution::Propagates => {
            // Fall through to check children.
        }
    }

    // Uppercase var references (ADT constructors) → Host.
    if tag == DeepTag::Var
        && let Some(name) = children.first().and_then(symbol_name)
        && name.chars().next().is_some_and(|c| c.is_uppercase())
    {
        reasons.push(HostReason::StructuralForm {
            tag: format!("var({})", name),
        });
        needs_host = true;
    }
    // Runtime inputs are derived from top-level declaration dependencies
    // above, where function parameters are structurally out of scope.
    // Treating every non-builtin Var here as an input leaks parameter
    // spellings into the manifest.

    // App: check the callee builtin.
    if tag == DeepTag::App
        && let Some(callee_name) = app_callee_name(children)
    {
        // Uppercase callee (ADT constructor call) → Host.
        if callee_name.chars().next().is_some_and(|c| c.is_uppercase()) {
            reasons.push(HostReason::StructuralForm {
                tag: format!("app({})", callee_name),
            });
            needs_host = true;
        }

        // Check builtin realizability.
        if let Some(decl) = builtin_decl(&callee_name) {
            match decl.realizability {
                Realizability::HostOnly => {
                    reasons.push(HostReason::HostOnlyBuiltin {
                        name: callee_name.to_string(),
                    });
                    needs_host = true;
                }
                Realizability::TensorAtTensorType => {
                    // Check if the application's type is scalar.
                    if app_type_is_scalar(meta) {
                        reasons.push(HostReason::ScalarTypedOp {
                            builtin: callee_name.to_string(),
                        });
                        needs_host = true;
                    }
                }
                Realizability::Universal => {}
            }
        }

        // Check transitive callee lane.
        if let Some(Lane::Host) = context.lane_by_def.get(&callee_name) {
            reasons.push(HostReason::TransitiveCaller {
                callee: callee_name.to_string(),
            });
            needs_host = true;
        } else if builtin_decl(&callee_name).is_none()
            && !context.lane_by_def.contains_key(&callee_name)
            && context.type_env.contains_key(&callee_name)
        {
            // A checked declaration supplied by a compiled library context is
            // absent from this new-code fixed point. Its callable value is
            // realized by the library-aware host evaluator, not by the
            // new-code named DAG. Fail closed to that lane instead of
            // promising a Tensor root which lowering cannot produce.
            reasons.push(HostReason::TransitiveCaller {
                callee: callee_name.to_string(),
            });
            needs_host = true;
        }
    }

    // Walk every child even after a Host contribution. `required_inputs` is
    // an independent manifest fact and may not become prefix-dependent on the
    // first routing reason encountered.
    for child in children {
        if expr_needs_host(
            child,
            context.lane_by_def,
            context.target_prims,
            context.type_env,
            reasons,
            inputs,
        ) {
            needs_host = true;
        }
    }

    needs_host
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

fn list_meta(list: &List) -> Option<&Metadata> {
    match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => Some(meta),
        _ => None,
    }
}

fn tagged_children(expr: &Expr) -> Option<(DeepTag, &[Expr])> {
    match expr {
        Expr::Node(node, _) => Some((node.tag(), node.children_slice())),
        Expr::List(list, _) => list.tag().map(|tag| (tag, get_children(list))),
        _ => None,
    }
}

fn tagged_meta(expr: &Expr) -> Option<&Metadata> {
    match expr {
        Expr::Node(node, _) => Some(node.meta()),
        Expr::List(list, _) => list_meta(list),
        _ => None,
    }
}

fn collect_top_level_dependencies(
    expr: &Expr,
    current_def: &str,
    top_level_names: &BTreeSet<String>,
    initial_bound: &BTreeSet<String>,
    dependencies: &mut BTreeSet<String>,
    references_self: &mut bool,
) {
    let mut bound = vec![initial_bound.clone()];
    collect_top_level_dependencies_scoped(
        expr,
        current_def,
        top_level_names,
        &mut bound,
        dependencies,
        references_self,
    );
}

/// Return the candidate names referenced free by one callable body. This is
/// the callable-selection counterpart of the top-level dependency walk: the
/// abstract manifest excludes declaration-local parameters, while a concrete
/// selected call must recover exactly the live tensor parameters before it can
/// become an owed root.
pub fn referenced_runtime_inputs(body: &Expr, candidates: &BTreeSet<String>) -> BTreeSet<String> {
    let mut inputs = BTreeSet::new();
    let mut references_self = false;
    collect_top_level_dependencies(
        body,
        "",
        candidates,
        &BTreeSet::new(),
        &mut inputs,
        &mut references_self,
    );
    inputs
}

fn collect_top_level_dependencies_scoped(
    expr: &Expr,
    current_def: &str,
    top_level_names: &BTreeSet<String>,
    bound: &mut Vec<BTreeSet<String>>,
    dependencies: &mut BTreeSet<String>,
    references_self: &mut bool,
) {
    if let Some((tag, children)) = tagged_children(expr) {
        match tag {
            DeepTag::Var => {
                if let Some(name) = children.first().and_then(symbol_name)
                    && !bound.iter().rev().any(|scope| scope.contains(name))
                    && top_level_names.contains(name)
                {
                    if name == current_def {
                        *references_self = true;
                    } else {
                        dependencies.insert(name.to_string());
                    }
                }
            }
            DeepTag::Fn => {
                if children.len() >= 2 {
                    bound.push(dependency_param_names(&children[0]));
                    collect_top_level_dependencies_scoped(
                        &children[1],
                        current_def,
                        top_level_names,
                        bound,
                        dependencies,
                        references_self,
                    );
                    bound.pop();
                }
            }
            DeepTag::Let => {
                if children.len() < 2 {
                    return;
                }
                let mut let_scope = BTreeSet::new();
                if let Some((DeepTag::Bind, bindings)) = tagged_children(&children[0]) {
                    let mut index = 0;
                    while index + 1 < bindings.len() {
                        collect_top_level_dependencies_scoped(
                            &bindings[index + 1],
                            current_def,
                            top_level_names,
                            bound,
                            dependencies,
                            references_self,
                        );
                        dependency_binding_names(&bindings[index], &mut let_scope);
                        index += 2;
                    }
                }
                bound.push(let_scope);
                collect_top_level_dependencies_scoped(
                    &children[1],
                    current_def,
                    top_level_names,
                    bound,
                    dependencies,
                    references_self,
                );
                bound.pop();
            }
            DeepTag::Match => {
                let Some((scrutinee, arms)) = children.split_first() else {
                    return;
                };
                collect_top_level_dependencies_scoped(
                    scrutinee,
                    current_def,
                    top_level_names,
                    bound,
                    dependencies,
                    references_self,
                );
                for arm in arms {
                    let Some((DeepTag::Arm, arm_children)) = tagged_children(arm) else {
                        continue;
                    };
                    let Some((pattern, scoped_children)) = arm_children.split_first() else {
                        continue;
                    };
                    let mut arm_scope = BTreeSet::new();
                    dependency_binding_names(pattern, &mut arm_scope);
                    bound.push(arm_scope);
                    for child in scoped_children {
                        collect_top_level_dependencies_scoped(
                            child,
                            current_def,
                            top_level_names,
                            bound,
                            dependencies,
                            references_self,
                        );
                    }
                    bound.pop();
                }
            }
            _ => {
                for child in children {
                    collect_top_level_dependencies_scoped(
                        child,
                        current_def,
                        top_level_names,
                        bound,
                        dependencies,
                        references_self,
                    );
                }
            }
        }
        return;
    }

    match expr {
        Expr::BareList(elements, _) => {
            for child in elements {
                collect_top_level_dependencies_scoped(
                    child,
                    current_def,
                    top_level_names,
                    bound,
                    dependencies,
                    references_self,
                );
            }
        }
        Expr::MetaExpr(meta, _) => collect_top_level_dependencies_scoped(
            &meta.expr,
            current_def,
            top_level_names,
            bound,
            dependencies,
            references_self,
        ),
        Expr::UnknownForm(data) => {
            for child in &data.children {
                collect_top_level_dependencies_scoped(
                    child,
                    current_def,
                    top_level_names,
                    bound,
                    dependencies,
                    references_self,
                );
            }
        }
        Expr::Atom(_, _) | Expr::Map(_, _) | Expr::Node(_, _) | Expr::List(_, _) => {}
    }
}

fn dependency_param_names(params_expr: &Expr) -> BTreeSet<String> {
    let Some((DeepTag::Params, params)) = tagged_children(params_expr) else {
        return BTreeSet::new();
    };
    params.iter().filter_map(dependency_param_name).collect()
}

fn dependency_param_name(param: &Expr) -> Option<String> {
    match param {
        Expr::Atom(Atom::Name(name), _) => Some(name.clone()),
        Expr::MetaExpr(meta, _) => dependency_param_name(&meta.expr),
        Expr::BareList(elements, _) => elements.first().and_then(symbol_name).map(str::to_string),
        Expr::List(list, _) => list
            .elements
            .first()
            .and_then(symbol_name)
            .map(str::to_string),
        Expr::UnknownForm(data) => Some(data.head.clone()),
        Expr::Node(node, _) => node
            .children_slice()
            .first()
            .and_then(symbol_name)
            .map(str::to_string),
        Expr::Map(_, _) | Expr::Atom(_, _) => None,
    }
}

fn dependency_binding_names(expr: &Expr, names: &mut BTreeSet<String>) {
    if let Some(name) = symbol_name(expr) {
        names.insert(name.to_string());
        return;
    }
    if let Some((tag, children)) = tagged_children(expr) {
        match tag {
            DeepTag::PatVar => {
                if let Some(name) = children.first().and_then(symbol_name) {
                    names.insert(name.to_string());
                }
            }
            DeepTag::PatAs => {
                if let Some(name) = children.first().and_then(symbol_name) {
                    names.insert(name.to_string());
                }
                if let Some(inner) = children.get(1) {
                    dependency_binding_names(inner, names);
                }
            }
            _ => {
                for child in children {
                    dependency_binding_names(child, names);
                }
            }
        }
        return;
    }
    match expr {
        Expr::MetaExpr(meta, _) => dependency_binding_names(&meta.expr, names),
        Expr::BareList(elements, _) => {
            for child in elements {
                dependency_binding_names(child, names);
            }
        }
        Expr::UnknownForm(data) => {
            for child in &data.children {
                dependency_binding_names(child, names);
            }
        }
        Expr::Atom(_, _) | Expr::Map(_, _) | Expr::Node(_, _) | Expr::List(_, _) => {}
    }
}

fn type_expr_contains_tensor(expr: &Expr) -> bool {
    if expr.tag() == Some(DeepTag::TTensor) {
        return true;
    }
    match expr {
        Expr::Node(node, _) => node.children_slice().iter().any(type_expr_contains_tensor),
        Expr::List(list, _) => get_children(list).iter().any(type_expr_contains_tensor),
        Expr::BareList(elements, _) => elements.iter().any(type_expr_contains_tensor),
        Expr::MetaExpr(meta, _) => type_expr_contains_tensor(&meta.expr),
        Expr::UnknownForm(data) => data.children.iter().any(type_expr_contains_tensor),
        Expr::Atom(_, _) | Expr::Map(_, _) => false,
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(s), _) => Some(s.as_str()),
        _ => None,
    }
}

fn app_callee_name(children: &[Expr]) -> Option<String> {
    let callee = children.first()?;
    let (tag, children) = tagged_children(callee)?;
    (tag == DeepTag::Var)
        .then(|| children.first().and_then(symbol_name))
        .flatten()
        .map(str::to_string)
}

/// Check if an app expression's type metadata indicates a scalar (t-prim) type.
fn app_type_is_scalar(meta: Option<&Metadata>) -> bool {
    meta.and_then(Metadata::ty)
        .is_some_and(|ty| ty.expression().tag() == Some(DeepTag::TPrim))
}

/// Extract a def's name and body from a top-level expression.
fn extract_def(expr: &Expr) -> Option<(String, &Expr)> {
    let (tag, children) = tagged_children(expr)?;
    if tag != DeepTag::Def {
        return None;
    }
    let name = children.first().and_then(symbol_name)?;
    let body = children.get(1)?;
    // Skip fn-typed defs (they're callable, not roots).
    if let Some((DeepTag::Fn, fn_children)) = tagged_children(body) {
        // Still register the def for transitive analysis, but the body
        // is the fn body, not the fn itself.
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

fn type_expr_has_primitive_result(expr: &Expr) -> bool {
    match tagged_children(expr) {
        Some((DeepTag::TPrim, _)) => true,
        Some((DeepTag::TFn, children)) => {
            children.last().is_some_and(type_expr_has_primitive_result)
        }
        _ => false,
    }
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
        Expr::Node(node, _) => {
            for elem in node.children_slice() {
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
    let effects_by_def = crate::def_effect_rows(program);
    let mut entries = Vec::new();

    for expr in exprs {
        collect_manifest_entries(
            expr,
            type_env,
            program.adt_registry(),
            &effects_by_def,
            realizability,
            &mut entries,
        );
    }

    RootManifest { entries }
}

fn collect_manifest_entries(
    expr: &Expr,
    type_env: &BTreeMap<String, Expr>,
    adt_registry: &chelis_types::adt::AdtRegistry,
    effects_by_def: &BTreeMap<String, chelis_types::types::EffectSet>,
    realizability: &RealizabilityResult,
    out: &mut Vec<RootEntry>,
) {
    let Some((tag, children)) = tagged_children(expr) else {
        return;
    };

    if tag == DeepTag::Module {
        for child in children.iter().skip(1) {
            collect_manifest_entries(
                child,
                type_env,
                adt_registry,
                effects_by_def,
                realizability,
                out,
            );
        }
        return;
    }

    if tag != DeepTag::Def {
        return;
    }

    let Some(name) = children.first().and_then(symbol_name) else {
        return;
    };
    let body = children.get(1);

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
        .or_else(|| expr_type_metadata(expr))
        .or_else(|| body.and_then(expr_type_metadata))
        .cloned()
        .unwrap_or_else(|| {
            Expr::Atom(
                Atom::Name("unknown".to_string()),
                chelis_deep::Span::new(0, 0),
            )
        });

    // Function-typed entries, including aliases, are callable rather than
    // observations. Only an actual nullary function declaration is applied.
    // A nullary arrow-form def is an owed root: evaluation applies its thunk
    // and surfaces the return value (chelis#947), so unwrap its sole return
    // type for dotted expansion.
    let is_declaration = body.is_some_and(|body| body.tag() == Some(DeepTag::Fn));
    let (observation_ty, nullary_declaration) = match tagged_children(&ty) {
        Some((DeepTag::TFn, [return_ty])) if is_declaration => {
            // Auto-applying an effectful thunk merely to observe it would run
            // an effect that an unselected declaration otherwise runs zero
            // times. Only effect-free nullary definitions are value roots;
            // effectful nullaries remain callable declarations.
            if effects_by_def.get(name).is_some_and(|row| !row.is_empty()) {
                return;
            }
            (return_ty, true)
        }
        Some((DeepTag::TFn, _)) => return,
        _ => (&ty, false),
    };
    // [05-OBS-7]'s "value result" is concrete. A nullary generic helper
    // such as `empty[a]() -> Box[a]` has no standalone value or ABI until a
    // call site instantiates `a`; treating its declaration as an automatic
    // root asks the Host lane to realize a symbol that specialization quite
    // correctly omitted. Concrete selected calls are expanded separately by
    // `expand_manifest_root`.
    if nullary_declaration && type_expr_has_unresolved_observation_parameter(observation_ty) {
        return;
    }

    let template = RootEntry {
        name: name.to_string(),
        path: Vec::new(),
        def_name: name.to_string(),
        ty: observation_ty.clone(),
        lane,
        required_inputs,
        reasons,
    };
    // A nullary declaration's checked value is an `Fn` wrapper. Tuple
    // topology can be recovered from its result type alone, but a static ADT
    // needs the concrete constructor expression to choose field labels. Feed
    // the shared expander the function's result body, just as selected
    // parameterized callables do, instead of the outer callable wrapper.
    let observation_value = if nullary_declaration {
        body.and_then(|value| {
            tagged_children(value)
                .filter(|(tag, _)| *tag == DeepTag::Fn)
                .and_then(|(_, fn_children)| fn_children.last())
        })
        .or(body)
    } else {
        body
    };
    out.extend(expand_manifest_root(
        template,
        observation_value,
        adt_registry,
    ));
}

fn type_expr_has_unresolved_observation_parameter(expr: &Expr) -> bool {
    let Some((tag, children)) = tagged_children(expr) else {
        return false;
    };
    match tag {
        DeepTag::TVar | DeepTag::DVar | DeepTag::DRank => true,
        DeepTag::DName => children.first().and_then(symbol_name) == Some("*"),
        _ => children
            .iter()
            .any(type_expr_has_unresolved_observation_parameter),
    }
}

/// Expand one already-classified root through the same tuple/ADT topology
/// authority used by [`compute_root_manifest`]. Concrete eval selection uses
/// this for a fully supplied parameterized callable: selecting the call makes
/// its result an owed root, but must not create a second, bare-only topology
/// implementation in the compiler API.
pub fn expand_manifest_root(
    template: RootEntry,
    value: Option<&Expr>,
    adt_registry: &chelis_types::adt::AdtRegistry,
) -> Vec<RootEntry> {
    let name = template.name.clone();
    let ty = template.ty.clone();
    let mut entries = Vec::new();
    expand_manifest_entry(
        &name,
        &ty,
        value,
        adt_registry,
        &[],
        &template,
        &mut entries,
    );
    entries
}

/// Expand statically fixed product topology in the same depth-first order and
/// with the same labels used by IR lowering and runtime dotted lookup.
fn expand_manifest_entry(
    name: &str,
    ty: &Expr,
    value: Option<&Expr>,
    adt_registry: &chelis_types::adt::AdtRegistry,
    path: &[RootPathStep],
    template: &RootEntry,
    out: &mut Vec<RootEntry>,
) {
    if let Some((DeepTag::TTuple, element_types)) = tagged_children(ty) {
        let tuple_values = value
            .and_then(tagged_children)
            .filter(|(tag, _)| *tag == DeepTag::Tuple)
            .map(|(_, children)| children);
        for (index, element_ty) in element_types.iter().enumerate() {
            let mut child_path = path.to_vec();
            child_path.push(RootPathStep::Tuple(index));
            expand_manifest_entry(
                &format!("{name}.{index}"),
                element_ty,
                tuple_values.and_then(|values| values.get(index)),
                adt_registry,
                &child_path,
                template,
                out,
            );
        }
        return;
    }

    // A top-level unannotated tuple binding may not have a type-env entry;
    // its checked value still has fixed product topology. Use the value shape
    // and each checked child's own metadata rather than collapsing it to one
    // bare root.
    if let Some((DeepTag::Tuple, tuple_values)) = value.and_then(tagged_children) {
        for (index, component) in tuple_values.iter().enumerate() {
            let component_ty = expr_type_metadata(component).unwrap_or(ty);
            let mut child_path = path.to_vec();
            child_path.push(RootPathStep::Tuple(index));
            expand_manifest_entry(
                &format!("{name}.{index}"),
                component_ty,
                Some(component),
                adt_registry,
                &child_path,
                template,
                out,
            );
        }
        return;
    }

    if let Some((DeepTag::TAdt, type_children)) = tagged_children(ty)
        && let Some(adt_name) = type_children.first().and_then(symbol_name)
        // These language ADTs have dedicated variable-sized or opaque host
        // representations. They are not fixed products, even when a source
        // expression happens to expose one constructor. Treating `List`'s
        // recursive Cons cells as ordinary ADT fields creates paths the C ABI
        // cannot represent and makes root count depend on literal length.
        && !matches!(adt_name, "Option" | "List" | "Dict" | "MappedFile")
        && let Some(components) = static_adt_components(value, adt_name, adt_registry)
        && !components.is_empty()
    {
        for (index, label, component_value) in components {
            let component_ty = expr_type_metadata(component_value).unwrap_or(ty);
            let mut child_path = path.to_vec();
            child_path.push(RootPathStep::Adt(index));
            expand_manifest_entry(
                &format!("{name}.{label}"),
                component_ty,
                Some(component_value),
                adt_registry,
                &child_path,
                template,
                out,
            );
        }
        return;
    }

    let mut entry = template.clone();
    entry.name = name.to_string();
    entry.path = path.to_vec();
    entry.ty = ty.clone();
    out.push(entry);
}

fn static_adt_components<'a>(
    value: Option<&'a Expr>,
    adt_name: &str,
    adt_registry: &chelis_types::adt::AdtRegistry,
) -> Option<Vec<(usize, String, &'a Expr)>> {
    let value = value?;
    let (tag, children) = tagged_children(value)?;
    let adt = adt_registry.lookup(adt_name)?;

    match tag {
        DeepTag::Record => {
            let ctor = children.first().and_then(symbol_name)?;
            let variant = adt.variants.iter().find(|variant| variant.name == ctor)?;
            let mut components = Vec::with_capacity(variant.fields.len());
            for (index, (declared_name, _)) in variant.fields.iter().enumerate() {
                let label = declared_name.clone().unwrap_or_else(|| index.to_string());
                let field_value = children.iter().skip(1).find_map(|field| {
                    let (DeepTag::Kv, field_children) = tagged_children(field)? else {
                        return None;
                    };
                    (field_children.first().and_then(symbol_name) == Some(label.as_str()))
                        .then(|| field_children.get(1))
                        .flatten()
                })?;
                components.push((index, label, field_value));
            }
            Some(components)
        }
        DeepTag::App => {
            let ctor = app_callee_name(children)?;
            let variant = adt.variants.iter().find(|variant| variant.name == ctor)?;
            let values = children.get(1..)?;
            if values.len() != variant.fields.len() {
                return None;
            }
            Some(
                variant
                    .fields
                    .iter()
                    .zip(values)
                    .enumerate()
                    .map(|(index, ((declared_name, _), value))| {
                        (
                            index,
                            declared_name.clone().unwrap_or_else(|| index.to_string()),
                            value,
                        )
                    })
                    .collect(),
            )
        }
        _ => None,
    }
}

/// Extract type metadata from an expression's metadata map.
fn expr_type_metadata(expr: &Expr) -> Option<&Expr> {
    tagged_meta(expr)?.ty().map(|ty| ty.expression())
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::ast::Metadata;
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

    #[test]
    fn top_level_host_value_dependency_propagates_without_touching_a_sibling() {
        let checked = check_program_from_source(
            "source = to_tensor([1.0, 2.0])\n\
             dependent = mul(source, source)\n\
             def sibling(value: tensor[2, f32]) -> tensor[2, f32] = mul(value, value)\n",
        );
        let result = infer_realizability(&checked, C_PRIMS);
        assert_eq!(result.lane_by_def.get("source"), Some(&Lane::Host));
        assert_eq!(result.lane_by_def.get("dependent"), Some(&Lane::Host));
        assert_eq!(result.lane_by_def.get("sibling"), Some(&Lane::Tensor));
    }

    #[test]
    fn host_root_inherits_only_the_referenced_top_level_runtime_input() {
        let checked = check_program_from_source(
            "input: tensor[2, f64] = input\nother: tensor[2, f64] = other\n\
             values = to_list(input)\n",
        );
        let input_def = checked
            .annotated_exprs()
            .iter()
            .find(|expr| extract_def(expr).is_some_and(|(name, _)| name == "input"))
            .expect("input definition");
        let (_, input_body) = extract_def(input_def).expect("input definition body");
        let mut dependencies = BTreeSet::new();
        let mut references_self = false;
        collect_top_level_dependencies(
            input_body,
            "input",
            &[
                "input".to_string(),
                "other".to_string(),
                "values".to_string(),
            ]
            .into_iter()
            .collect(),
            &BTreeSet::new(),
            &mut dependencies,
            &mut references_self,
        );
        assert!(
            references_self,
            "the stamped self reference must be detected"
        );
        let input_ty = checked
            .type_env()
            .get("input")
            .or_else(|| expr_type_metadata(input_def))
            .or_else(|| expr_type_metadata(input_body))
            .expect("input declared type");
        assert!(
            type_expr_contains_tensor(input_ty),
            "the declared tensor type must retain its tensor constructor: {input_ty:?}"
        );
        let result = infer_realizability(&checked, C_PRIMS);
        assert_eq!(
            result.required_inputs_by_def.get("input"),
            Some(&["input".to_string()].into_iter().collect()),
            "a self-referential tensor declaration is its own runtime input"
        );
        assert_eq!(
            result.required_inputs_by_def.get("values"),
            Some(&["input".to_string()].into_iter().collect()),
            "per-root closure must inherit the referenced input without a sibling"
        );
    }

    #[test]
    fn function_parameter_names_do_not_become_manifest_runtime_inputs() {
        let checked = check_program_from_source(
            "value: List[int32] = [cast(1, int32)]\n\
             def square(value: tensor[2, f32]) -> tensor[2, f32] = mul(value, value)\n",
        );
        let square_body = checked
            .annotated_exprs()
            .iter()
            .find_map(|expr| {
                let (name, body) = extract_def(expr)?;
                (name == "square").then_some(body)
            })
            .expect("square body");
        let square_params = checked
            .signature_inference()
            .functions
            .get("square")
            .expect("square signature metadata")
            .params
            .iter()
            .map(|param| param.name.clone())
            .collect::<BTreeSet<_>>();
        assert_eq!(
            square_params,
            ["value".to_string()].into_iter().collect(),
            "checker-owned function parameter metadata is the lexical boundary"
        );
        let mut dependencies = BTreeSet::new();
        let mut references_self = false;
        collect_top_level_dependencies(
            square_body,
            "square",
            &["value".to_string(), "square".to_string()]
                .into_iter()
                .collect(),
            &square_params,
            &mut dependencies,
            &mut references_self,
        );
        assert_eq!(
            dependencies,
            BTreeSet::new(),
            "a same-spelled parameter is not a top-level dependency"
        );
        let result = infer_realizability(&checked, C_PRIMS);
        assert_eq!(
            result.required_inputs_by_def.get("square"),
            Some(&BTreeSet::new()),
            "declaration-local parameters are not top-level runtime inputs"
        );
        assert_eq!(
            result.lane_by_def.get("square"),
            Some(&Lane::Tensor),
            "a parameter shadows a same-spelled Host top-level binding"
        );
    }

    #[test]
    fn recursive_function_name_does_not_become_a_runtime_input() {
        let checked = check_program_from_source(
            "def recur[n](x: tensor[n, f32], i: int64) -> tensor[n, f32] =\n\
               if lte(i, cast(0, int64)) then x else recur(x, sub(i, cast(1, int64)))\n\
             out = recur(to_tensor([1.0, 2.0]), cast(2, int64))\n",
        );
        let result = infer_realizability(&checked, C_PRIMS);
        assert_eq!(
            result.required_inputs_by_def.get("recur"),
            Some(&BTreeSet::new()),
            "a recursive callee is a declaration dependency, not a runtime tensor input"
        );
        assert_eq!(
            result.required_inputs_by_def.get("out"),
            Some(&BTreeSet::new()),
            "the caller must not inherit the recursive function's own name as input"
        );
    }

    #[test]
    fn module_wrapped_defs_receive_realizability_entries() {
        let checked = check_program_from_source("module Demo.Root\nvalue = cast(7, int64)\n");
        let result = infer_realizability(&checked, C_PRIMS);
        assert!(
            result.lane_by_def.contains_key("value"),
            "module wrapping must not separate manifest discovery from lane inference"
        );
        assert_eq!(
            result.required_inputs_by_def.get("value"),
            Some(&BTreeSet::new())
        );
    }

    #[test]
    fn nullary_observation_thunk_routes_to_the_host_value_path() {
        let checked = check_program_from_source("def answer() -> int64 = cast(42, int64)\n");
        let result = infer_realizability(&checked, EVAL_PRIMS);
        assert_eq!(result.lane_by_def.get("answer"), Some(&Lane::Host));
        assert!(
            result.reasons_by_def["answer"].contains(&HostReason::StructuralForm {
                tag: "fn(nullary-root)".to_string(),
            })
        );
    }

    #[test]
    fn checked_external_callee_routes_to_the_library_host_path() {
        let span = chelis_deep::Span::new(0, 0);
        let callee = Expr::node(
            DeepTag::Var,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("library_add".to_string()), span)],
            span,
        );
        let expression = Expr::node(
            DeepTag::App,
            Metadata::default(),
            vec![callee, Expr::Atom(Atom::Int(1), span)],
            span,
        );
        let external_type = Expr::node(
            DeepTag::TFn,
            Metadata::default(),
            vec![
                Expr::node(
                    DeepTag::TPrim,
                    Metadata::default(),
                    vec![Expr::Atom(Atom::Name("int64".to_string()), span)],
                    span,
                ),
                Expr::node(
                    DeepTag::TPrim,
                    Metadata::default(),
                    vec![Expr::Atom(Atom::Name("int64".to_string()), span)],
                    span,
                ),
            ],
            span,
        );
        let lane_by_def = BTreeMap::new();
        let target = EVAL_PRIMS.iter().copied().collect();
        let type_env = BTreeMap::from([("library_add".to_string(), external_type)]);
        let mut reasons = Vec::new();
        let mut inputs = BTreeSet::new();

        assert!(expr_needs_host(
            &expression,
            &lane_by_def,
            &target,
            &type_env,
            &mut reasons,
            &mut inputs,
        ));
        assert!(reasons.contains(&HostReason::TransitiveCaller {
            callee: "library_add".to_string(),
        }));
    }

    #[test]
    fn scalar_result_function_and_its_value_caller_route_host() {
        let checked = check_program_from_source("def f(x: f32) -> f32 = atan(x)\nout = f(3.5)\n");
        let result = infer_realizability(&checked, C_PRIMS);
        assert_eq!(result.lane_by_def.get("f"), Some(&Lane::Host));
        assert_eq!(result.lane_by_def.get("out"), Some(&Lane::Host));
        assert!(
            result.reasons_by_def["out"].contains(&HostReason::TransitiveCaller {
                callee: "f".to_string(),
            })
        );
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

    // chelis#1082: checked programs carry stamped Nodes. Manifest discovery
    // must observe that carrier directly; a List-only walk returns an empty
    // manifest and makes every downstream completeness check vacuous.
    #[test]
    fn compute_root_manifest_discovers_stamped_node_defs() {
        let span = chelis_deep::Span::new(0, 0);
        let body = Expr::node(
            DeepTag::Lit,
            Metadata::default(),
            vec![Expr::Atom(Atom::Int(42), span)],
            span,
        );
        let def = Expr::node(
            DeepTag::Def,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("answer".to_string()), span), body],
            span,
        );
        let ty = Expr::node(
            DeepTag::TPrim,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("int32".to_string()), span)],
            span,
        );
        let type_env = BTreeMap::from([("answer".to_string(), ty)]);
        let realizability = RealizabilityResult {
            lane_by_def: BTreeMap::from([("answer".to_string(), Lane::Tensor)]),
            required_inputs_by_def: BTreeMap::new(),
            reasons_by_def: BTreeMap::new(),
        };
        let effects_by_def = BTreeMap::new();
        let mut entries = Vec::new();
        collect_manifest_entries(
            &def,
            &type_env,
            &chelis_types::adt::AdtRegistry::new(),
            &effects_by_def,
            &realizability,
            &mut entries,
        );
        assert_eq!(
            entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["answer"]
        );
    }

    // chelis#1083: tuple topology is part of the checked root contract, not
    // something an individual runtime may rediscover after lowering.
    #[test]
    fn compute_root_manifest_expands_nested_tuple_roots_depth_first() {
        let checked = check_program_from_source(
            "result = (cast(1, int32), (cast(2, int32), cast(3, int32)))\n",
        );
        let realizability = infer_realizability(&checked, C_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert_eq!(
            manifest
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["result.0", "result.1.0", "result.1.1"]
        );
    }

    // Record constructors have a statically known variant and therefore use
    // the same declared field labels as lowering and runtime lookup.
    #[test]
    fn compute_root_manifest_expands_static_record_adt_roots() {
        let checked = check_program_from_source(
            "type Pair = | Pair { left: int32, right: int32 }\n\
             result = Pair { left: cast(1, int32), right: cast(2, int32) }\n",
        );
        let realizability = infer_realizability(&checked, C_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert_eq!(
            manifest
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["result.left", "result.right"]
        );
    }

    #[test]
    fn compute_root_manifest_expands_static_positional_adt_roots() {
        let checked = check_program_from_source(
            "type Pair = | Pair(int32, int32)\n\
             result = Pair(cast(1, int32), cast(2, int32))\n",
        );
        let realizability = infer_realizability(&checked, C_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert_eq!(
            manifest
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["result.0", "result.1"]
        );
    }

    #[test]
    fn compute_root_manifest_keeps_dynamic_adt_variant_as_bare_root() {
        let checked = check_program_from_source(
            "type Choice = | First(int32) | Second(int32)\n\
             def choose(flag: bool) -> Choice = \
               if flag then First(cast(1, int32)) else Second(cast(2, int32))\n\
             result = choose(true)\n",
        );
        let realizability = infer_realizability(&checked, C_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert_eq!(
            manifest
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["result"]
        );
    }

    #[test]
    fn compute_root_manifest_keeps_builtin_recursive_list_as_bare_root() {
        let checked =
            check_program_from_source("result: List[int32] = [cast(1, int32), cast(2, int32)]\n");
        let realizability = infer_realizability(&checked, C_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert_eq!(
            manifest
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["result"],
            "a variable-sized recursive container is not fixed dotted product topology"
        );
        assert!(manifest.entries[0].path.is_empty());
    }

    #[test]
    fn compute_root_manifest_excludes_effectful_nullary_thunks() {
        let checked = check_program_from_source(
            "def logged() -> string = debug(\"must-not-run-for-observation\")\n",
        );
        let realizability = infer_realizability(&checked, EVAL_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert!(
            manifest.entries.is_empty(),
            "observing an unselected effectful thunk must not run its effect"
        );
    }

    #[test]
    fn compute_root_manifest_keeps_pure_nullary_thunks() {
        let checked = check_program_from_source("def answer() -> int32 = cast(42, int32)\n");
        let realizability = infer_realizability(&checked, EVAL_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert_eq!(
            manifest
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["answer"]
        );
    }

    #[test]
    fn compute_root_manifest_expands_pure_nullary_tuple_and_record_roots() {
        let checked = check_program_from_source(
            "type Pair = | Pair { left: int32, right: int32 }\n\
             def tupled() -> (int32, int32) = (cast(1, int32), cast(2, int32))\n\
             def answer() -> Pair = Pair { left: cast(3, int32), right: cast(4, int32) }\n",
        );
        let realizability = infer_realizability(&checked, C_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert_eq!(
            manifest
                .entries
                .iter()
                .map(|entry| entry.name.as_str())
                .collect::<Vec<_>>(),
            ["tupled.0", "tupled.1", "answer.left", "answer.right"]
        );
    }

    #[test]
    fn compute_root_manifest_excludes_uninstantiated_generic_nullary_thunks() {
        let checked = check_program_from_source(
            "type Box[a] =\n  | Empty\n  | Full { value: a }\n\
             def empty[a]() -> Box[a] = Empty\n",
        );
        let realizability = infer_realizability(&checked, EVAL_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert!(
            manifest.entries.is_empty(),
            "a generic nullary has no standalone value until a concrete call instantiates it"
        );
    }

    #[test]
    fn compute_root_manifest_keeps_dynamic_shape_value_roots() {
        let checked = check_program_from_source(
            "rows: List[List[int64]] = [[cast(1, int64)], [cast(2, int64)]]\n\
             padded = pad_sequences(rows, cast(0, int64))\n",
        );
        let realizability = infer_realizability(&checked, C_PRIMS);
        let manifest = compute_root_manifest(&checked, &realizability);
        assert!(
            manifest.entries.iter().any(|entry| entry.name == "padded"),
            "a concrete value root with a dynamic tensor dimension remains observable"
        );
    }

    // #1086: a headless list must fail closed to Host with a recorded reason,
    // not silently route Tensor.
    #[test]
    fn bare_list_routes_host_fail_closed() {
        let expr = Expr::BareList(vec![], chelis_deep::Span::new(0, 0));
        let lane_by_def: BTreeMap<String, Lane> = BTreeMap::new();
        let target: UnordSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: BTreeMap<String, Expr> = BTreeMap::new();
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
        use chelis_deep::ast::{Metadata, UnknownFormData};
        let expr = Expr::UnknownForm(Box::new(UnknownFormData {
            head: "mystery".to_string(),
            meta: Metadata::default(),
            children: vec![],
            span: chelis_deep::Span::new(0, 0),
        }));
        let lane_by_def: BTreeMap<String, Lane> = BTreeMap::new();
        let target: UnordSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: BTreeMap<String, Expr> = BTreeMap::new();
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
        let lane_by_def: BTreeMap<String, Lane> = BTreeMap::new();
        let target: UnordSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: BTreeMap<String, Expr> = BTreeMap::new();
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
        let lane_by_def: BTreeMap<String, Lane> = BTreeMap::new();
        let target: UnordSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: BTreeMap<String, Expr> = BTreeMap::new();
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
        let lane_by_def: BTreeMap<String, Lane> = BTreeMap::new();
        let target: UnordSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: BTreeMap<String, Expr> = BTreeMap::new();
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
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("R".to_string()), span)],
            span,
        );
        let expr = Expr::node(DeepTag::Block, Metadata::default(), vec![record], span);
        let lane_by_def: BTreeMap<String, Lane> = BTreeMap::new();
        let target: UnordSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: BTreeMap<String, Expr> = BTreeMap::new();
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
            Metadata::default(),
            vec![Expr::Atom(Atom::Int(1), span)],
            span,
        );
        let expr = Expr::node(DeepTag::Block, Metadata::default(), vec![literal], span);
        let lane_by_def: BTreeMap<String, Lane> = BTreeMap::new();
        let target: UnordSet<Prim> = C_PRIMS.iter().copied().collect();
        let type_env: BTreeMap<String, Expr> = BTreeMap::new();
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
        let checked =
            check_program_from_source("x = insert(scalar_to_tensor(cast(1.0, f32)), 0, 1i64)\n");
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
