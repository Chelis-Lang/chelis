//! Check-time dependency proof for `vmap` movement extents.
//!
//! `spec/06-transformations.md` section 3.7 makes an extent derived from
//! vmapped tensor elements a type error. The IR verifier keeps the same rule
//! as defense in depth, but the public checker must reject before lowering.

use super::*;
use std::collections::{BTreeMap, BTreeSet};

type ParamDeps = BTreeSet<usize>;

#[derive(Clone)]
struct FunctionDef {
    params: Vec<String>,
    tensor_params: BTreeSet<usize>,
    body: deep::Expr,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct MovementWitness {
    operation: String,
    span_id: Option<String>,
    span_offset: usize,
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct FunctionSummary {
    return_deps: ParamDeps,
    movement_deps: BTreeMap<usize, MovementWitness>,
}

pub(super) fn validate_vmap_extent_dependencies(
    exprs: &[deep::Expr],
    type_env: &IrTypeEnv,
    errors: &mut DiagnosticSink<'_>,
) {
    let defs = collect_functions(exprs, type_env);
    if defs.is_empty() {
        return;
    }
    let summaries = summarize_functions(&defs);
    for expr in exprs {
        walk_vmap_sites(expr, &defs, &summaries, errors);
    }
}

fn collect_functions(exprs: &[deep::Expr], type_env: &IrTypeEnv) -> BTreeMap<String, FunctionDef> {
    let mut defs = BTreeMap::new();
    for expr in top_level_decl_items(exprs) {
        let Some((DeepTag::Def, _, kids)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = kids.first().and_then(symbol_name) else {
            continue;
        };
        let Some((params, body)) = kids.get(1).and_then(extract_fn_params_and_body) else {
            continue;
        };
        let param_types = build_def_param_scope(expr, type_env);
        let tensor_params = params
            .iter()
            .enumerate()
            .filter_map(|(index, param)| {
                param_types
                    .get(param)
                    .is_some_and(type_expr_contains_tensor)
                    .then_some(index)
            })
            .collect();
        defs.insert(
            name.to_string(),
            FunctionDef {
                params,
                tensor_params,
                body,
            },
        );
    }
    defs
}

fn type_expr_contains_tensor(expr: &deep::Expr) -> bool {
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return false;
    };
    match tag {
        DeepTag::TTensor => true,
        DeepTag::TRef | DeepTag::TFn | DeepTag::TTuple | DeepTag::TAdt => {
            kids.iter().any(type_expr_contains_tensor)
        }
        _ => false,
    }
}

fn summarize_functions(defs: &BTreeMap<String, FunctionDef>) -> BTreeMap<String, FunctionSummary> {
    let mut summaries = defs
        .keys()
        .map(|name| (name.clone(), FunctionSummary::default()))
        .collect::<BTreeMap<_, _>>();
    loop {
        let mut changed = false;
        for (name, def) in defs {
            let next = summarize_function(def, &summaries);
            if summaries.get(name) != Some(&next) {
                summaries.insert(name.clone(), next);
                changed = true;
            }
        }
        if !changed {
            return summaries;
        }
    }
}

fn summarize_function(
    def: &FunctionDef,
    summaries: &BTreeMap<String, FunctionSummary>,
) -> FunctionSummary {
    let mut locals = def
        .params
        .iter()
        .enumerate()
        .map(|(index, name)| (name.clone(), ParamDeps::from([index])))
        .collect::<UnordMap<_, _>>();
    let mut movement_deps = BTreeMap::new();
    let return_deps = analyze_expr(&def.body, &mut locals, summaries, &mut movement_deps);
    FunctionSummary {
        return_deps,
        movement_deps,
    }
}

fn analyze_expr(
    expr: &deep::Expr,
    locals: &mut UnordMap<String, ParamDeps>,
    summaries: &BTreeMap<String, FunctionSummary>,
    movement_deps: &mut BTreeMap<usize, MovementWitness>,
) -> ParamDeps {
    stack_guard!("analyze_vmap_extent_dependencies", expr, ParamDeps::new());

    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return match expr {
            deep::Expr::BareList(elements, _) => {
                union_children(elements, locals, summaries, movement_deps)
            }
            deep::Expr::MetaExpr(meta, _) => {
                analyze_expr(&meta.expr, locals, summaries, movement_deps)
            }
            deep::Expr::Map(map, _) => {
                let values = map
                    .entries
                    .iter()
                    .map(|(_, value)| value.clone())
                    .collect::<Vec<_>>();
                union_children(&values, locals, summaries, movement_deps)
            }
            deep::Expr::UnknownForm(data) => {
                union_children(&data.children, locals, summaries, movement_deps)
            }
            deep::Expr::Atom(_, _) => ParamDeps::new(),
            deep::Expr::List(_, _) | deep::Expr::Node(_, _) => ParamDeps::new(),
        };
    };

    match tag {
        DeepTag::Var => kids
            .first()
            .and_then(symbol_name)
            .and_then(|name| locals.get(name))
            .cloned()
            .unwrap_or_default(),
        DeepTag::Lit | DeepTag::Quote => ParamDeps::new(),
        DeepTag::Let => analyze_let(kids, locals, summaries, movement_deps),
        DeepTag::Fn => ParamDeps::new(),
        DeepTag::App => analyze_app(expr, kids, locals, summaries, movement_deps),
        _ => union_children(kids, locals, summaries, movement_deps),
    }
}

fn analyze_let(
    kids: &[deep::Expr],
    locals: &mut UnordMap<String, ParamDeps>,
    summaries: &BTreeMap<String, FunctionSummary>,
    movement_deps: &mut BTreeMap<usize, MovementWitness>,
) -> ParamDeps {
    let mut let_locals = locals.clone();
    if let Some(bindings) = kids.first()
        && let Some((DeepTag::Bind, _, binding_kids)) = stamped_parts(bindings)
    {
        let (pairs, _) = binding_kids.as_chunks::<2>();
        for pair in pairs {
            let Some(name) = param_name_for_refs(&pair[0]) else {
                continue;
            };
            let deps = analyze_expr(&pair[1], &mut let_locals, summaries, movement_deps);
            let_locals.insert(name, deps);
        }
    }
    kids.get(1)
        .map(|body| analyze_expr(body, &mut let_locals, summaries, movement_deps))
        .unwrap_or_default()
}

fn analyze_app(
    expr: &deep::Expr,
    kids: &[deep::Expr],
    locals: &mut UnordMap<String, ParamDeps>,
    summaries: &BTreeMap<String, FunctionSummary>,
    movement_deps: &mut BTreeMap<usize, MovementWitness>,
) -> ParamDeps {
    let callee = kids.first().and_then(var_expr_name);
    let arg_deps = kids
        .iter()
        .skip(1)
        .map(|arg| analyze_expr(arg, locals, summaries, movement_deps))
        .collect::<Vec<_>>();

    if callee == Some("shape") {
        return ParamDeps::new();
    }

    if let Some(operation) = callee.and_then(movement_operation) {
        let bound_deps = movement_bound_dependencies(operation, &arg_deps);
        let witness = MovementWitness {
            operation: operation.to_string(),
            span_id: expr.span_id().map(str::to_string),
            span_offset: expr.span().offset,
        };
        for dependency in bound_deps {
            movement_deps
                .entry(dependency)
                .or_insert_with(|| witness.clone());
        }
    }

    if let Some(summary) = callee.and_then(|name| summaries.get(name)) {
        for (callee_param, witness) in &summary.movement_deps {
            if let Some(deps) = arg_deps.get(*callee_param) {
                for dependency in deps {
                    movement_deps
                        .entry(*dependency)
                        .or_insert_with(|| witness.clone());
                }
            }
        }
        return summary
            .return_deps
            .iter()
            .filter_map(|param| arg_deps.get(*param))
            .flatten()
            .copied()
            .collect();
    }

    arg_deps.into_iter().flatten().collect()
}

fn movement_operation(name: &str) -> Option<&str> {
    matches!(name, "expand" | "reshape" | "shrink" | "pad" | "stride").then_some(name)
}

fn movement_bound_dependencies(operation: &str, args: &[ParamDeps]) -> ParamDeps {
    let selected: Box<dyn Iterator<Item = &ParamDeps> + '_> = match operation {
        "expand" => Box::new(args.get(2).into_iter()),
        "reshape" => Box::new(args.get(1).into_iter()),
        "shrink" | "pad" | "stride" => Box::new(args.iter().skip(1)),
        _ => Box::new(std::iter::empty()),
    };
    selected.flatten().copied().collect()
}

fn union_children(
    children: &[deep::Expr],
    locals: &mut UnordMap<String, ParamDeps>,
    summaries: &BTreeMap<String, FunctionSummary>,
    movement_deps: &mut BTreeMap<usize, MovementWitness>,
) -> ParamDeps {
    children
        .iter()
        .flat_map(|child| analyze_expr(child, locals, summaries, movement_deps))
        .collect()
}

fn var_expr_name(expr: &deep::Expr) -> Option<&str> {
    let (DeepTag::Var, _, kids) = stamped_parts(expr)? else {
        return None;
    };
    kids.first().and_then(symbol_name)
}

fn walk_vmap_sites(
    expr: &deep::Expr,
    defs: &BTreeMap<String, FunctionDef>,
    summaries: &BTreeMap<String, FunctionSummary>,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("walk_vmap_extent_sites", expr);
    if let Some((tag, _, kids)) = stamped_parts(expr) {
        if tag == DeepTag::Vmap
            && let Some(callee_name) = kids.first().and_then(transformed_callee_name)
            && let (Some(def), Some(summary)) = (defs.get(callee_name), summaries.get(callee_name))
            && let Some((param_index, witness)) = summary
                .movement_deps
                .iter()
                .find(|(param, _)| def.tensor_params.contains(param))
        {
            let param_name = def
                .params
                .get(*param_index)
                .map(String::as_str)
                .unwrap_or("<tensor>");
            let location = witness
                .span_id
                .as_deref()
                .map(|span| format!("at {span}"))
                .or_else(|| {
                    (witness.span_offset > 0).then(|| format!("at byte {}", witness.span_offset))
                })
                .unwrap_or_else(|| "at its movement site".to_string());
            let mut error = CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!(
                    "batch_varying_extent: vmap cannot vectorize an extent that depends on \
                     batched tensor elements: the {} bound {location} reads elements of \
                     vmapped argument '{param_name}'. Compute the extent from shape() or a \
                     scalar argument, or apply the movement outside vmap.",
                    witness.operation
                ),
                vec![
                    "Compute the extent from shape() or a scalar argument, or apply the movement outside vmap"
                        .to_string(),
                ],
            );
            if let Some(span_id) = &witness.span_id {
                error.span_offset = parse_span_offset(span_id);
                error.span_id = Some(span_id.clone());
            } else if witness.span_offset > 0 {
                error.span_offset = Some(witness.span_offset);
            }
            errors.push(error);
        }
        for child in kids {
            walk_vmap_sites(child, defs, summaries, errors);
        }
        return;
    }

    match expr {
        deep::Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                walk_vmap_sites(value, defs, summaries, errors);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            walk_vmap_sites(&meta.expr, defs, summaries, errors);
            for (_, value) in &meta.entries {
                walk_vmap_sites(value, defs, summaries, errors);
            }
        }
        deep::Expr::BareList(elements, _) => {
            for child in elements {
                walk_vmap_sites(child, defs, summaries, errors);
            }
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                walk_vmap_sites(child, defs, summaries, errors);
            }
        }
        deep::Expr::Atom(_, _) | deep::Expr::List(_, _) | deep::Expr::Node(_, _) => {}
    }
}

fn transformed_callee_name(expr: &deep::Expr) -> Option<&str> {
    if let Some(name) = var_expr_name(expr) {
        return Some(name);
    }
    let (tag, _, kids) = stamped_parts(expr)?;
    if matches!(
        tag,
        DeepTag::Grad | DeepTag::Jit | DeepTag::Realize | DeepTag::Copy
    ) {
        return kids.first().and_then(transformed_callee_name);
    }
    None
}
