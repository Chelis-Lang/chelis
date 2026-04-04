//! Type inference engine for the Chelis type checker.
//!
//! Walks Deep AST nodes and assigns types using Hindley-Milner inference.

use std::collections::HashMap;

use chelis_deep::ast as deep;

use crate::adt::AdtRegistry;
use crate::builtins;
use crate::env::Env;
use crate::errors::*;
use crate::types::*;
use crate::unify::*;

/// Result of running type inference on a program.
#[derive(Debug)]
pub struct InferResult {
    pub errors: Vec<CheckError>,
    pub typed_nodes: usize,
    pub total_nodes: usize,
}

/// Run type inference on a list of top-level Deep expressions.
pub fn infer_program(exprs: &[deep::Expr]) -> InferResult {
    let (mut env, mut vg) = builtins::builtin_env();
    let mut subst = Subst::new();
    let mut adt_reg = AdtRegistry::new();
    let mut errors = Vec::new();
    let mut typed_nodes = 0;
    let mut total_nodes = 0;

    // First pass: collect deftype and defsig declarations
    for expr in exprs {
        collect_declarations(
            expr,
            &mut env,
            &mut vg,
            &mut subst,
            &mut adt_reg,
            &mut errors,
        );
    }

    // Second pass: infer def bodies
    for expr in exprs {
        infer_top_level(
            expr,
            &mut env,
            &mut vg,
            &mut subst,
            &adt_reg,
            &mut errors,
            &mut typed_nodes,
            &mut total_nodes,
        );
    }

    InferResult {
        errors,
        typed_nodes,
        total_nodes,
    }
}

// ── Helpers ──────────────────────────────────────────────────────

fn get_tag(list: &deep::List) -> Option<&str> {
    if let Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)) = list.elements.first() {
        Some(tag.as_str())
    } else {
        None
    }
}

fn children(list: &deep::List) -> &[deep::Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

/// Get metadata map from element[1] of a list.
fn get_meta(list: &deep::List) -> Option<&deep::MetaMap> {
    if list.elements.len() > 1
        && let deep::Expr::Map(meta, _) = &list.elements[1]
    {
        return Some(meta);
    }
    None
}

/// Extract a symbol name from an Expr.
fn symbol_name(expr: &deep::Expr) -> Option<&str> {
    match expr {
        deep::Expr::Atom(deep::Atom::Symbol(s), _) => Some(s.as_str()),
        _ => None,
    }
}

// ── Declaration collection (first pass) ──────────────────────────

fn collect_declarations(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &mut AdtRegistry,
    _errors: &mut Vec<CheckError>,
) {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return,
    };

    let tag = match get_tag(list) {
        Some(t) => t,
        None => return,
    };

    let kids = children(list);

    match tag {
        "deftype" => {
            let ctors = adt_reg.register_deftype(kids, vg);
            for (name, scheme) in ctors {
                env.bind(name, scheme);
            }
        }
        "defsig" => {
            // (defsig {} name type_expr)
            if kids.len() >= 2
                && let Some(name) = symbol_name(&kids[0])
            {
                let ty = deep_type_to_type(&kids[1], vg, &mut HashMap::new());
                let scheme = env.generalize(&ty, subst);
                env.bind(name.to_string(), scheme);
            }
        }
        _ => {}
    }
}

// ── Top-level inference (second pass) ────────────────────────────

#[allow(clippy::too_many_arguments)]
fn infer_top_level(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return,
    };

    let tag = match get_tag(list) {
        Some(t) => t,
        None => return,
    };

    // Skip deftype/defsig (already processed)
    if tag == "deftype" || tag == "defsig" {
        return;
    }

    let kids = children(list);

    if tag == "def" && kids.len() >= 2 {
        let name = match symbol_name(&kids[0]) {
            Some(n) => n.to_string(),
            None => return,
        };
        let body_ty = infer_expr(
            &kids[1],
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
        let scheme = env.generalize(&body_ty, subst);
        env.bind(name, scheme);
    } else {
        // Any other top-level expression
        let _ = infer_expr(
            expr,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
    }
}

// ── Core inference ───────────────────────────────────────────────

#[allow(clippy::too_many_arguments)]
fn infer_expr(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    *total_nodes += 1;

    let result = match expr {
        deep::Expr::Atom(atom, _) => infer_atom(atom),
        deep::Expr::List(list, _) => {
            let tag = get_tag(list);
            match tag {
                Some("var") => infer_var(list, env, vg, subst, errors),
                Some("lit") => infer_lit(list, vg),
                Some("app") => infer_app(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("fn") => infer_fn(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("let") => infer_let(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("if") => infer_if(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("match") => infer_match(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("pipe") => infer_pipe(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("tuple") => infer_tuple(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("tuple-get") => infer_tuple_get(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("cast") => infer_cast(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("grad") => infer_grad(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("def") => infer_def(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                ),
                Some("defsig") => {
                    // Already handled in first pass
                    Type::Unit
                }
                Some("deftype") => {
                    // Already handled in first pass
                    Type::Unit
                }
                _ => {
                    // Unknown tag -- try to infer children
                    Type::Error
                }
            }
        }
        deep::Expr::Map(_, _) => Type::Unit,
        deep::Expr::MetaExpr(meta, _) => infer_expr(
            &meta.expr,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        ),
    };

    if !matches!(result, Type::Error) {
        *typed_nodes += 1;
    }

    result
}

fn infer_atom(atom: &deep::Atom) -> Type {
    match atom {
        deep::Atom::Int(_) => Type::Prim(Prim::Int32),
        deep::Atom::Float(_) => Type::Prim(Prim::F32),
        deep::Atom::Bool(_) => Type::Prim(Prim::Bool),
        deep::Atom::Str(_) => Type::Prim(Prim::String),
        deep::Atom::Symbol(_) | deep::Atom::Keyword(_) => Type::Error,
    }
}

fn infer_var(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    let kids = children(list);
    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
        if let Some(scheme) = env.lookup(name) {
            let scheme = scheme.clone();
            let ty = env.instantiate(&scheme, vg);
            subst.apply(&ty)
        } else {
            errors.push(CheckError {
                kind: CheckErrorKind::UnboundVariable,
                message: format!("unbound variable: {name}"),
                suggestions: vec![],
            });
            Type::Error
        }
    } else {
        Type::Error
    }
}

fn infer_lit(list: &deep::List, vg: &mut VarGen) -> Type {
    let meta = get_meta(list);
    let kids = children(list);

    // Check metadata for type annotation
    if let Some(meta) = meta {
        for (key, val) in &meta.entries {
            if key == "type" {
                return deep_type_to_type(val, vg, &mut HashMap::new());
            }
        }
    }

    // Fall back to value-based defaults
    if let Some(val) = kids.first() {
        match val {
            deep::Expr::Atom(deep::Atom::Int(_), _) => Type::Prim(Prim::Int32),
            deep::Expr::Atom(deep::Atom::Float(_), _) => Type::Prim(Prim::F32),
            deep::Expr::Atom(deep::Atom::Bool(_), _) => Type::Prim(Prim::Bool),
            deep::Expr::Atom(deep::Atom::Str(_), _) => Type::Prim(Prim::String),
            _ => Type::Error,
        }
    } else {
        Type::Error
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_app(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    // Check if func is a comparison op (for special return type handling)
    let func_name = if let deep::Expr::List(flist, _) = &kids[0] {
        if get_tag(flist) == Some("var") {
            children(flist)
                .first()
                .and_then(|e| symbol_name(e))
                .map(|s| s.to_string())
        } else {
            None
        }
    } else {
        None
    };

    let func_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let arg_tys: Vec<Type> = kids[1..]
        .iter()
        .map(|a| infer_expr(a, env, vg, subst, adt_reg, errors, typed_nodes, total_nodes))
        .collect();

    // If func or any arg is Error, propagate
    if matches!(func_ty, Type::Error) || arg_tys.iter().any(|t| matches!(t, Type::Error)) {
        return Type::Error;
    }

    let ret_tv = vg.fresh_type();
    let expected_fn = Type::Fn(arg_tys.clone(), Box::new(ret_tv.clone()));

    match unify(&func_ty, &expected_fn, subst) {
        Ok(()) => {
            let result_ty = subst.apply(&ret_tv);

            // Special case: comparison ops return tensor[D, bool]
            if let Some(ref fname) = func_name
                && builtins::COMPARISON_OPS.contains(&fname.as_str())
            {
                // Try to extract dims from arg types
                if let Some(first_arg) = arg_tys.first() {
                    let resolved_arg = subst.apply(first_arg);
                    if let Type::Tensor(dims, _) = resolved_arg {
                        return Type::Tensor(dims, Prim::Bool);
                    }
                }
            }

            result_ty
        }
        Err(te) => {
            errors.push(te.into());
            Type::Error
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_fn(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    // kids[0] = (params {} x1 ... xn)
    // kids[1] = body
    let param_names = extract_params(&kids[0]);
    let mut param_types = Vec::new();
    let mut fn_env = env.clone();

    for pname in &param_names {
        let tv = vg.fresh_type();
        fn_env.bind(pname.clone(), Scheme::mono(tv.clone()));
        param_types.push(tv);
    }

    let body = if kids.len() > 1 {
        &kids[1]
    } else {
        return Type::Error;
    };
    let body_ty = infer_expr(
        body,
        &mut fn_env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    let resolved_params: Vec<Type> = param_types.iter().map(|t| subst.apply(t)).collect();
    let resolved_body = subst.apply(&body_ty);

    Type::Fn(resolved_params, Box::new(resolved_body))
}

/// Extract parameter names from (params {} x1 ... xn).
fn extract_params(expr: &deep::Expr) -> Vec<String> {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list);
            if tag == Some("params") {
                children(list)
                    .iter()
                    .filter_map(|e| symbol_name(e).map(|s| s.to_string()))
                    .collect()
            } else {
                // Try treating all elements as param names
                list.elements
                    .iter()
                    .filter_map(|e| symbol_name(e).map(|s| s.to_string()))
                    .collect()
            }
        }
        _ => vec![],
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_let(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    // kids[0] = (bind {} x1 e1 x2 e2 ...)
    // kids[1] = body
    let mut let_env = env.clone();

    if let deep::Expr::List(bind_list, _) = &kids[0] {
        let bind_children = children(bind_list);
        // Process pairs: name, expr
        let mut i = 0;
        while i + 1 < bind_children.len() {
            if let Some(name) = symbol_name(&bind_children[i]) {
                let expr_ty = infer_expr(
                    &bind_children[i + 1],
                    &mut let_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );
                let scheme = let_env.generalize(&expr_ty, subst);
                let_env.bind(name.to_string(), scheme);
            }
            i += 2;
        }
    }

    infer_expr(
        &kids[1],
        &mut let_env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    )
}

#[allow(clippy::too_many_arguments)]
fn infer_if(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 3 {
        return Type::Error;
    }

    let cond_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    // Condition should be bool (or tensor[D, bool])
    // Try unifying with bool; if that fails, it might be a tensor condition
    let _ = unify(&cond_ty, &Type::Prim(Prim::Bool), subst);

    let then_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let else_ty = infer_expr(
        &kids[2],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    match unify(&then_ty, &else_ty, subst) {
        Ok(()) => subst.apply(&then_ty),
        Err(te) => {
            errors.push(te.into());
            subst.apply(&then_ty)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_match(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let scrutinee_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    let mut result_ty: Option<Type> = None;
    let mut covered_variants: Vec<String> = Vec::new();

    for arm_expr in &kids[1..] {
        if let deep::Expr::List(arm_list, _) = arm_expr
            && get_tag(arm_list) == Some("arm")
        {
            let arm_kids = children(arm_list);
            // arm_kids[0] = pattern, arm_kids[1] = guard (usually ()), arm_kids[2] = body
            if arm_kids.len() >= 3 {
                let mut arm_env = env.clone();
                let pat = &arm_kids[0];
                pattern_bindings(
                    pat,
                    &scrutinee_ty,
                    &mut arm_env,
                    vg,
                    subst,
                    &mut covered_variants,
                );

                let body_ty = infer_expr(
                    &arm_kids[2],
                    &mut arm_env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    typed_nodes,
                    total_nodes,
                );

                match &result_ty {
                    None => result_ty = Some(body_ty),
                    Some(prev) => {
                        if let Err(te) = unify(prev, &body_ty, subst) {
                            errors.push(te.into());
                        }
                        result_ty = Some(subst.apply(prev));
                    }
                }
            }
        }
    }

    // Exhaustiveness check
    let resolved_scrutinee = subst.apply(&scrutinee_ty);
    if let Type::Adt(ref adt_name, _) = resolved_scrutinee
        && let Some(all_variants) = adt_reg.variant_names(adt_name)
    {
        let missing: Vec<&String> = all_variants
            .iter()
            .filter(|v| !covered_variants.contains(v))
            .collect();
        if !missing.is_empty() {
            let names: Vec<&str> = missing.iter().map(|s| s.as_str()).collect();
            errors.push(CheckError {
                kind: CheckErrorKind::NonExhaustiveMatch,
                message: format!("non-exhaustive match: missing variants {:?}", names),
                suggestions: vec![],
            });
        }
    }

    result_ty.unwrap_or(Type::Error)
}

fn pattern_bindings(
    pat: &deep::Expr,
    scrutinee_ty: &Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    covered_variants: &mut Vec<String>,
) {
    if let deep::Expr::List(list, _) = pat {
        let tag = get_tag(list).unwrap_or("");
        let kids = children(list);
        match tag {
            "pat-var" => {
                if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                    let resolved = subst.apply(scrutinee_ty);
                    env.bind(name.to_string(), Scheme::mono(resolved));
                }
            }
            "pat-wild" => {
                // Wildcard covers everything -- mark all variants if we had an ADT
                // (conservative: treat as covering nothing specific for exhaustiveness)
            }
            "pat-lit" => {
                // No bindings, but value should match scrutinee type
            }
            "pat-ctor" => {
                if let Some(ctor_name) = kids.first().and_then(|e| symbol_name(e)) {
                    covered_variants.push(ctor_name.to_string());

                    // Look up constructor in env and decompose
                    if let Some(scheme) = env.lookup(ctor_name) {
                        let scheme = scheme.clone();
                        let ctor_ty = env.instantiate(&scheme, vg);
                        // Unify the result of the constructor with scrutinee type
                        match &ctor_ty {
                            Type::Fn(arg_types, ret) => {
                                let _ = unify(ret, scrutinee_ty, subst);
                                // Bind sub-patterns to argument types
                                for (i, sub_pat) in kids[1..].iter().enumerate() {
                                    if i < arg_types.len() {
                                        let resolved = subst.apply(&arg_types[i]);
                                        pattern_bindings(
                                            sub_pat,
                                            &resolved,
                                            env,
                                            vg,
                                            subst,
                                            covered_variants,
                                        );
                                    }
                                }
                            }
                            _ => {
                                // Nullary constructor
                                let _ = unify(&ctor_ty, scrutinee_ty, subst);
                            }
                        }
                    }
                }
            }
            _ => {}
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_pipe(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let mut current_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );

    for stage in &kids[1..] {
        let stage_ty = infer_expr(
            stage,
            env,
            vg,
            subst,
            adt_reg,
            errors,
            typed_nodes,
            total_nodes,
        );
        let ret_tv = vg.fresh_type();
        let expected = Type::Fn(vec![current_ty.clone()], Box::new(ret_tv.clone()));

        match unify(&stage_ty, &expected, subst) {
            Ok(()) => {
                current_ty = subst.apply(&ret_tv);
            }
            Err(te) => {
                errors.push(te.into());
                return Type::Error;
            }
        }
    }

    current_ty
}

#[allow(clippy::too_many_arguments)]
fn infer_tuple(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    let elems: Vec<Type> = kids
        .iter()
        .map(|e| infer_expr(e, env, vg, subst, adt_reg, errors, typed_nodes, total_nodes))
        .collect();
    Type::Tuple(elems)
}

#[allow(clippy::too_many_arguments)]
fn infer_tuple_get(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    let tuple_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&tuple_ty);

    let index = match &kids[1] {
        deep::Expr::Atom(deep::Atom::Int(n), _) => *n as usize,
        _ => {
            return Type::Error;
        }
    };

    match resolved {
        Type::Tuple(ref elems) => {
            if index < elems.len() {
                elems[index].clone()
            } else {
                errors.push(CheckError {
                    kind: CheckErrorKind::TupleIndexOutOfBounds,
                    message: format!(
                        "tuple index {} out of bounds for tuple of size {}",
                        index,
                        elems.len()
                    ),
                    suggestions: vec![],
                });
                Type::Error
            }
        }
        Type::Error => Type::Error,
        _ => {
            errors.push(CheckError {
                kind: CheckErrorKind::TypeMismatch,
                message: format!("expected tuple type, got {resolved}"),
                suggestions: vec![],
            });
            Type::Error
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_cast(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    let expr_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&expr_ty);

    // kids[1] = (t-prim {} new_precision)
    let new_prec = match deep_type_to_type(&kids[1], vg, &mut HashMap::new()) {
        Type::Prim(p) => p,
        _ => return Type::Error,
    };

    match resolved {
        Type::Tensor(dims, _) => Type::Tensor(dims, new_prec),
        Type::Prim(_) => Type::Prim(new_prec),
        Type::Error => Type::Error,
        _ => {
            errors.push(CheckError {
                kind: CheckErrorKind::CastNonTensor,
                message: format!("cast requires tensor or prim type, got {resolved}"),
                suggestions: vec![],
            });
            Type::Error
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_grad(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.is_empty() {
        return Type::Error;
    }

    let f_ty = infer_expr(
        &kids[0],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let resolved = subst.apply(&f_ty);

    match resolved {
        Type::Fn(args, ret) => {
            // grad(f) : Fn([T1,...,Tn], Tuple([S, Tuple([T1,...,Tn])]))
            let grad_ret = Type::Tuple(vec![*ret, Type::Tuple(args.clone())]);
            Type::Fn(args, Box::new(grad_ret))
        }
        Type::Error => Type::Error,
        _ => {
            // Can't determine function structure, return fresh var
            vg.fresh_type()
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn infer_def(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    typed_nodes: &mut usize,
    total_nodes: &mut usize,
) -> Type {
    let kids = children(list);
    if kids.len() < 2 {
        return Type::Error;
    }

    let name = match symbol_name(&kids[0]) {
        Some(n) => n.to_string(),
        None => return Type::Error,
    };

    let body_ty = infer_expr(
        &kids[1],
        env,
        vg,
        subst,
        adt_reg,
        errors,
        typed_nodes,
        total_nodes,
    );
    let scheme = env.generalize(&body_ty, subst);
    env.bind(name, scheme);
    body_ty
}

// ── Type conversion from Deep AST ───────────────────────────────

/// Convert a Deep type expression to an internal Type.
/// `tvar_map` maps type variable names to TypeVars (created on demand).
fn deep_type_to_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    tvar_map: &mut HashMap<String, TypeVar>,
) -> Type {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list).unwrap_or("");
            let kids = children(list);
            match tag {
                "t-prim" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        Prim::parse_name(name)
                            .map(Type::Prim)
                            .unwrap_or(Type::Error)
                    } else {
                        Type::Error
                    }
                }
                "t-var" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        if name == "_" {
                            vg.fresh_type()
                        } else {
                            let tv = *tvar_map
                                .entry(name.to_string())
                                .or_insert_with(|| vg.fresh_tvar());
                            Type::Var(tv)
                        }
                    } else {
                        Type::Error
                    }
                }
                "t-fn" => {
                    if kids.is_empty() {
                        return Type::Error;
                    }
                    let args: Vec<Type> = kids[..kids.len() - 1]
                        .iter()
                        .map(|c| deep_type_to_type(c, vg, tvar_map))
                        .collect();
                    let ret = deep_type_to_type(&kids[kids.len() - 1], vg, tvar_map);
                    Type::Fn(args, Box::new(ret))
                }
                "t-tensor" => {
                    if kids.is_empty() {
                        return Type::Error;
                    }
                    let prec_expr = &kids[kids.len() - 1];
                    let prec = match deep_type_to_type(prec_expr, vg, tvar_map) {
                        Type::Prim(p) => p,
                        _ => return Type::Error,
                    };
                    let dims: Vec<Dim> = kids[..kids.len() - 1]
                        .iter()
                        .filter_map(|c| parse_dim(c, vg))
                        .collect();
                    Type::Tensor(dims, prec)
                }
                "t-adt" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        let args: Vec<Type> = kids[1..]
                            .iter()
                            .map(|c| deep_type_to_type(c, vg, tvar_map))
                            .collect();
                        Type::Adt(name.to_string(), args)
                    } else {
                        Type::Error
                    }
                }
                "t-tuple" => {
                    let elems: Vec<Type> = kids
                        .iter()
                        .map(|c| deep_type_to_type(c, vg, tvar_map))
                        .collect();
                    Type::Tuple(elems)
                }
                "t-unit" => Type::Unit,
                _ => Type::Error,
            }
        }
        _ => Type::Error,
    }
}

/// Parse a dimension expression from Deep AST, with support for dim variables.
fn parse_dim(expr: &deep::Expr, vg: &mut VarGen) -> Option<Dim> {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list).unwrap_or("");
            let kids = children(list);
            match tag {
                "d-name" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        if name == "*" {
                            Some(Dim::Wildcard)
                        } else {
                            Some(Dim::Name(name.to_string()))
                        }
                    } else {
                        None
                    }
                }
                "d-var" => Some(vg.fresh_dim()),
                "d-lit" => {
                    if let Some(deep::Expr::Atom(deep::Atom::Int(n), _)) = kids.first() {
                        Some(Dim::Lit(*n))
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn check(src: &str) -> InferResult {
        let exprs = chelis_deep::parser::parse_str(src).unwrap();
        infer_program(&exprs)
    }

    fn check_ok(src: &str) {
        let result = check(src);
        assert!(
            result.errors.is_empty(),
            "expected no errors, got: {:?}",
            result.errors
        );
    }

    fn check_err(src: &str, expected_kind: CheckErrorKind) {
        let result = check(src);
        assert!(
            !result.errors.is_empty(),
            "expected error {expected_kind:?}, got none"
        );
        assert!(
            result
                .errors
                .iter()
                .any(|e| std::mem::discriminant(&e.kind) == std::mem::discriminant(&expected_kind)),
            "expected {expected_kind:?}, got: {:?}",
            result.errors
        );
    }

    // ── Variable / literal tests ─────────────────────────────────

    #[test]
    fn lit_int32() {
        check_ok("(def {} x (lit {type: (t-prim {} int32)} 42))");
    }

    #[test]
    fn lit_f32() {
        check_ok("(def {} x (lit {type: (t-prim {} f32)} 3.0))");
    }

    #[test]
    fn lit_bool() {
        check_ok("(def {} x (lit {type: (t-prim {} bool)} true))");
    }

    #[test]
    fn lit_string() {
        check_ok(r#"(def {} x (lit {type: (t-prim {} string)} "hello"))"#);
    }

    #[test]
    fn lit_default_int() {
        check_ok("(def {} x (lit {} 42))");
    }

    #[test]
    fn lit_default_float() {
        check_ok("(def {} x (lit {} 3.14))");
    }

    #[test]
    fn unbound_variable() {
        check_err(
            "(def {} x (var {} unknown))",
            CheckErrorKind::UnboundVariable,
        );
    }

    #[test]
    fn var_lookup_defined() {
        check_ok(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (var {} x))",
        );
    }

    // ── Application tests ────────────────────────────────────────

    #[test]
    fn app_add_tensors() {
        check_ok(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn app_precision_mismatch() {
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bf16))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    #[test]
    fn app_not_a_function() {
        check_err(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (app {} (var {} x) (lit {type: (t-prim {} int32)} 1)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn app_dimension_mismatch() {
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} seq) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    // ── Lambda tests ─────────────────────────────────────────────

    #[test]
    fn fn_identity() {
        check_ok("(def {} id (fn {} (params {} x) (var {} x)))");
    }

    #[test]
    fn fn_two_params() {
        check_ok("(def {} f (fn {} (params {} x y) (var {} x)))");
    }

    #[test]
    fn fn_with_body_app() {
        check_ok("(def {} f (fn {} (params {} x y) (app {} (var {} add) (var {} x) (var {} y))))");
    }

    // ── Let tests ────────────────────────────────────────────────

    #[test]
    fn let_simple() {
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 42))
                 (var {} x)))",
        );
    }

    #[test]
    fn let_multiple_bindings() {
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 1) y (lit {type: (t-prim {} f32)} 2.0))
                 (var {} x)))",
        );
    }

    #[test]
    fn let_scoping() {
        // Variable defined in let should be usable in body
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 42))
                 (var {} x)))",
        );
    }

    // ── If tests ─────────────────────────────────────────────────

    #[test]
    fn if_correct() {
        check_ok(
            "(def {} x (lit {type: (t-prim {} bool)} true))
             (def {} a (lit {type: (t-prim {} int32)} 1))
             (def {} b (lit {type: (t-prim {} int32)} 2))
             (def {} c (if {} (var {} x) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn if_branch_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-prim {} bool)} true))
             (def {} a (lit {type: (t-prim {} int32)} 1))
             (def {} b (lit {type: (t-prim {} f32)} 2.0))
             (def {} c (if {} (var {} x) (var {} a) (var {} b)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // ── Pipe tests ───────────────────────────────────────────────

    #[test]
    fn pipe_simple() {
        check_ok(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))
             (def {} result (pipe {} (lit {type: (t-prim {} f32)} 1.0) (var {} f)))",
        );
    }

    #[test]
    fn pipe_chain() {
        check_ok(
            "(defsig {} f (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (var {} x)))
             (def {} result (pipe {} (lit {type: (t-prim {} f32)} 1.0) (var {} f) (var {} f)))",
        );
    }

    // ── Tuple tests ──────────────────────────────────────────────

    #[test]
    fn tuple_creation() {
        check_ok(
            "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} f32)} 2.0)))",
        );
    }

    #[test]
    fn tuple_get_valid() {
        check_ok(
            "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} f32)} 2.0)))
             (def {} x (tuple-get {} (var {} t) 0))",
        );
    }

    #[test]
    fn tuple_get_out_of_bounds() {
        check_err(
            "(def {} t (tuple {} (lit {type: (t-prim {} int32)} 1)))
             (def {} x (tuple-get {} (var {} t) 5))",
            CheckErrorKind::TupleIndexOutOfBounds,
        );
    }

    // ── Cast tests ───────────────────────────────────────────────

    #[test]
    fn cast_tensor() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (cast {} (var {} x) (t-prim {} bf16)))",
        );
    }

    #[test]
    fn cast_prim() {
        check_ok(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (cast {} (var {} x) (t-prim {} f32)))",
        );
    }

    // ── Grad tests ───────────────────────────────────────────────

    #[test]
    fn grad_function() {
        check_ok(
            "(defsig {} loss (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} loss (fn {} (params {} x) (var {} x)))
             (def {} g (grad {} (var {} loss)))",
        );
    }

    // ── ADT tests ────────────────────────────────────────────────

    #[test]
    fn adt_deftype_and_construct() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42)))",
        );
    }

    #[test]
    fn adt_nullary_constructor() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))
             (def {} x (var {} None))",
        );
    }

    #[test]
    fn adt_no_type_params() {
        check_ok(
            "(deftype {} Color () (variant {} Red) (variant {} Green) (variant {} Blue))
             (def {} c (var {} Red))",
        );
    }

    // ── Match tests ──────────────────────────────────────────────

    #[test]
    fn match_simple_adt() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42)))
             (def {} result
               (match {} (var {} x)
                 (arm {} (pat-ctor {} Some (pat-var {} v)) () (var {} v))
                 (arm {} (pat-ctor {} None) () (lit {type: (t-prim {} int32)} 0))))",
        );
    }

    #[test]
    fn match_non_exhaustive() {
        check_err(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None))
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42)))
             (def {} result
               (match {} (var {} x)
                 (arm {} (pat-ctor {} Some (pat-var {} v)) () (var {} v))))",
            CheckErrorKind::NonExhaustiveMatch,
        );
    }

    // ── Defsig tests ─────────────────────────────────────────────

    #[test]
    fn defsig_fn_signature() {
        check_ok(
            "(defsig {} double (t-fn {} (t-prim {} f32) (t-prim {} f32)))
             (def {} double (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn defsig_tensor_signature() {
        check_ok(
            "(defsig {} normalize_fn
               (t-fn {} (t-tensor {} (d-name {} batch) (t-prim {} f32))
                        (t-tensor {} (d-name {} batch) (t-prim {} f32))))
             (def {} normalize_fn (fn {} (params {} x) (var {} x)))",
        );
    }

    // ── Partial inference tests ──────────────────────────────────

    #[test]
    fn partial_inference_continues_after_error() {
        // First def has an error, second should still be processed
        let result = check(
            "(def {} x (var {} nonexistent))
             (def {} y (lit {type: (t-prim {} int32)} 42))",
        );
        assert!(!result.errors.is_empty(), "expected at least one error");
        // y should still have been typed
        assert!(result.typed_nodes > 0, "expected some typed nodes");
    }

    #[test]
    fn multiple_errors_collected() {
        let result = check(
            "(def {} x (var {} unknown1))
             (def {} y (var {} unknown2))",
        );
        assert!(
            result.errors.len() >= 2,
            "expected at least 2 errors, got {}",
            result.errors.len()
        );
    }

    // ── Builtin operation tests ──────────────────────────────────

    #[test]
    fn builtin_neg() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} neg) (var {} x)))",
        );
    }

    #[test]
    fn builtin_exp() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} exp) (var {} x)))",
        );
    }

    #[test]
    fn builtin_matmul() {
        check_ok(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} matmul) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn builtin_relu() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} relu) (var {} x)))",
        );
    }

    // ── Tensor type tests ────────────────────────────────────────

    #[test]
    fn tensor_2d() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))",
        );
    }

    #[test]
    fn tensor_literal_dim() {
        check_ok("(def {} x (lit {type: (t-tensor {} (d-lit {} 512) (t-prim {} f32))} 0))");
    }

    // ── Edge cases ───────────────────────────────────────────────

    #[test]
    fn empty_program() {
        let result = check("");
        assert!(result.errors.is_empty());
        assert_eq!(result.typed_nodes, 0);
        assert_eq!(result.total_nodes, 0);
    }

    #[test]
    fn def_with_fn_body() {
        check_ok(
            "(def {} double (fn {} (params {} x) (app {} (var {} add) (var {} x) (var {} x))))",
        );
    }

    #[test]
    fn nested_let() {
        check_ok(
            "(def {} result
               (let {} (bind {} x (lit {type: (t-prim {} int32)} 1))
                 (let {} (bind {} y (var {} x))
                   (var {} y))))",
        );
    }

    #[test]
    fn if_with_tensor_branches() {
        check_ok(
            "(def {} cond (lit {type: (t-prim {} bool)} true))
             (def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (if {} (var {} cond) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn fn_applied_to_args() {
        check_ok(
            "(def {} f (fn {} (params {} x) (var {} x)))
             (def {} result (app {} (var {} f) (lit {type: (t-prim {} int32)} 42)))",
        );
    }

    #[test]
    fn adt_with_fields() {
        check_ok(
            "(deftype {} Pair ()
               (variant {} MkPair
                 (field {} fst (t-prim {} int32))
                 (field {} snd (t-prim {} f32))))
             (def {} p (app {} (var {} MkPair) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} f32)} 2.0)))",
        );
    }

    #[test]
    fn pipe_with_lambda() {
        check_ok(
            "(def {} result
               (pipe {} (lit {type: (t-prim {} f32)} 1.0)
                        (fn {} (params {} x) (var {} x))))",
        );
    }

    #[test]
    fn total_nodes_counted() {
        let result = check("(def {} x (lit {type: (t-prim {} int32)} 42))");
        assert!(result.total_nodes > 0, "expected some total nodes");
    }

    #[test]
    fn tuple_three_elems() {
        check_ok(
            "(def {} t (tuple {}
               (lit {type: (t-prim {} int32)} 1)
               (lit {type: (t-prim {} f32)} 2.0)
               (lit {type: (t-prim {} bool)} true)))",
        );
    }
}
