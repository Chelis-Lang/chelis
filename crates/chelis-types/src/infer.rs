//! Type inference engine for the Chelis type checker.
//!
//! Walks Deep AST nodes and assigns types using Hindley-Milner inference.

use std::collections::{HashMap, HashSet};

use chelis_deep::Span;
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

#[derive(Debug, Clone)]
pub struct CheckedProgram {
    annotated_exprs: Vec<deep::Expr>,
    type_env: HashMap<String, deep::Expr>,
}

impl CheckedProgram {
    pub fn from_parts(
        annotated_exprs: Vec<deep::Expr>,
        type_env: HashMap<String, deep::Expr>,
    ) -> Self {
        Self {
            annotated_exprs,
            type_env,
        }
    }

    pub fn exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn annotated_exprs(&self) -> &[deep::Expr] {
        &self.annotated_exprs
    }

    pub fn type_env(&self) -> &HashMap<String, deep::Expr> {
        &self.type_env
    }
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

pub fn check_phase0e_program(exprs: &[deep::Expr]) -> Result<CheckedProgram, InferResult> {
    let type_env = build_phase0e_type_env(exprs);
    let mut result = infer_phase0e_program_with_env(exprs, &type_env);
    validate_phase0e_program(exprs, &type_env, &mut result.errors);
    if result.errors.is_empty() {
        let annotated_exprs = annotate_phase0e_program(exprs);
        let annotated_type_env = build_phase0e_type_env(&annotated_exprs);
        Ok(CheckedProgram {
            annotated_exprs,
            type_env: annotated_type_env,
        })
    } else {
        Err(result)
    }
}

pub fn infer_phase0e_program(exprs: &[deep::Expr]) -> InferResult {
    let type_env = build_phase0e_type_env(exprs);
    let mut result = infer_phase0e_program_with_env(exprs, &type_env);
    validate_phase0e_program(exprs, &type_env, &mut result.errors);
    result
}

fn infer_phase0e_program_with_env(exprs: &[deep::Expr], _type_env: &Phase0eTypeEnv) -> InferResult {
    let mut result = infer_program(exprs);
    result
        .errors
        .retain(|error| !matches!(error.kind, CheckErrorKind::UnboundVariable));
    for warning in chelis_deep::validate::validate(exprs) {
        result.errors.push(CheckError::new(
            match warning.kind {
                chelis_deep::validate::WarningKind::Arity => CheckErrorKind::ArityMismatch,
                _ => CheckErrorKind::Other,
            },
            warning.message,
            vec!["Use canonical Deep 3-tuple forms from spec/03".to_string()],
        ));
    }
    result
}

type Phase0eTypeEnv = HashMap<String, deep::Expr>;

fn build_phase0e_type_env(exprs: &[deep::Expr]) -> Phase0eTypeEnv {
    let mut env = HashMap::new();
    for expr in exprs {
        collect_phase0e_types(expr, &mut env);
    }
    env
}

fn collect_phase0e_types(expr: &deep::Expr, env: &mut Phase0eTypeEnv) {
    let deep::Expr::List(list, _) = expr else {
        return;
    };
    if get_tag(list) != Some("def") {
        return;
    }
    let kids = children(list);
    if kids.len() < 2 {
        return;
    }
    if let Some(name) = symbol_name(&kids[0])
        && let Some(ty) = expr_type_expr(&kids[1], env)
    {
        env.insert(name.to_string(), ty);
    }
}

fn validate_phase0e_program(
    exprs: &[deep::Expr],
    type_env: &Phase0eTypeEnv,
    errors: &mut Vec<CheckError>,
) {
    for expr in exprs {
        validate_phase0e_expr(expr, type_env, errors);
    }
}

fn validate_phase0e_expr(
    expr: &deep::Expr,
    type_env: &Phase0eTypeEnv,
    errors: &mut Vec<CheckError>,
) {
    match expr {
        deep::Expr::List(list, _) => {
            if get_tag(list) == Some("fn") {
                let scoped_env = extend_phase0e_env_with_fn_params(list, type_env);
                for elem in &list.elements {
                    validate_phase0e_expr(elem, &scoped_env, errors);
                }
                return;
            }
            if let Some(tag) = get_tag(list) {
                if matches!(
                    tag,
                    "if" | "tuple" | "tuple-get" | "match" | "grad" | "par" | "vmap" | "jit"
                ) {
                    errors.push(CheckError::new(
                        CheckErrorKind::Other,
                        format!("`{tag}` is not supported by Phase 0e lowering"),
                        vec!["Remove this construct or defer it to a later phase".to_string()],
                    ));
                }

                if tag == "app"
                    && let Some(func_name) = phase0e_builtin_name(list)
                    && is_phase0e_shape_sensitive_builtin(func_name)
                {
                    validate_phase0e_builtin_symbolic_requirements(
                        list, func_name, type_env, errors,
                    );
                }
            }

            for elem in &list.elements {
                validate_phase0e_expr(elem, type_env, errors);
            }
        }
        deep::Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                validate_phase0e_expr(value, type_env, errors);
            }
        }
        deep::Expr::MetaExpr(meta, _) => {
            for (_, value) in &meta.entries {
                validate_phase0e_expr(value, type_env, errors);
            }
            validate_phase0e_expr(&meta.expr, type_env, errors);
        }
        deep::Expr::Atom(_, _) => {}
    }
}

fn annotate_phase0e_program(exprs: &[deep::Expr]) -> Vec<deep::Expr> {
    let (mut env, mut vg) = builtins::builtin_env();
    let mut subst = Subst::new();
    let mut adt_reg = AdtRegistry::new();
    let mut declaration_errors = Vec::new();

    for expr in exprs {
        collect_declarations(
            expr,
            &mut env,
            &mut vg,
            &mut subst,
            &mut adt_reg,
            &mut declaration_errors,
        );
    }

    let mut annotated = Vec::with_capacity(exprs.len());
    for expr in exprs {
        annotated.push(annotate_expr_with_scope(expr, &env, &vg, &subst, &adt_reg));

        let mut step_errors = Vec::new();
        let mut typed_nodes = 0;
        let mut total_nodes = 0;
        infer_top_level(
            expr,
            &mut env,
            &mut vg,
            &mut subst,
            &adt_reg,
            &mut step_errors,
            &mut typed_nodes,
            &mut total_nodes,
        );
    }

    annotated
}

fn annotate_expr_with_scope(
    expr: &deep::Expr,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> deep::Expr {
    match expr {
        deep::Expr::Atom(_, _) => expr.clone(),
        deep::Expr::Map(map, span) => deep::Expr::Map(
            deep::MetaMap {
                entries: map
                    .entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            annotate_expr_with_scope(value, env, vg, subst, adt_reg),
                        )
                    })
                    .collect(),
            },
            *span,
        ),
        deep::Expr::MetaExpr(meta, span) => deep::Expr::MetaExpr(
            deep::MetaExpr {
                expr: Box::new(annotate_expr_with_scope(
                    &meta.expr, env, vg, subst, adt_reg,
                )),
                entries: meta
                    .entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            annotate_expr_with_scope(value, env, vg, subst, adt_reg),
                        )
                    })
                    .collect(),
            },
            *span,
        ),
        deep::Expr::List(list, span) => {
            if !matches!(list.elements.get(1), Some(deep::Expr::Map(_, _))) {
                return deep::Expr::List(
                    deep::List {
                        elements: list
                            .elements
                            .iter()
                            .map(|element| {
                                annotate_expr_with_scope(element, env, vg, subst, adt_reg)
                            })
                            .collect(),
                    },
                    *span,
                );
            }

            let tag = get_tag(list);
            let annotated_children = match tag {
                Some("fn") => annotate_fn_children(list, env, vg, subst, adt_reg),
                Some("let") => annotate_let_children(list, env, vg, subst, adt_reg),
                Some("match") => annotate_match_children(list, env, vg, subst, adt_reg),
                _ => children(list)
                    .iter()
                    .map(|child| annotate_expr_with_scope(child, env, vg, subst, adt_reg))
                    .collect(),
            };

            let mut elements = vec![
                list.elements[0].clone(),
                annotated_meta_map(list, expr, env, vg, subst, adt_reg),
            ];
            elements.extend(annotated_children);
            deep::Expr::List(deep::List { elements }, *span)
        }
    }
}

fn annotate_fn_children(
    list: &deep::List,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Vec<deep::Expr> {
    let kids = children(list);
    if kids.is_empty() {
        return vec![];
    }

    let fn_ty = infer_expr_in_scope(
        &deep::Expr::List(list.clone(), span_of_list(list)),
        env,
        vg,
        subst,
        adt_reg,
    );
    let resolved_fn_ty = subst.apply(&fn_ty);
    let param_types = match resolved_fn_ty {
        Type::Fn(args, _) => args,
        _ => Vec::new(),
    };

    let mut param_vg = vg.clone();
    let raw_params = extract_params(&kids[0], &mut param_vg, adt_reg);
    let mut fn_env = env.clone();
    for (index, (name, maybe_ty)) in raw_params.iter().enumerate() {
        let ty = maybe_ty
            .clone()
            .or_else(|| param_types.get(index).cloned())
            .unwrap_or(Type::Error);
        fn_env.bind(name.clone(), Scheme::mono(ty));
    }

    let mut result = vec![annotate_expr_with_scope(&kids[0], env, vg, subst, adt_reg)];
    if let Some(body) = kids.get(1) {
        result.push(annotate_expr_with_scope(body, &fn_env, vg, subst, adt_reg));
    }
    result
}

fn annotate_let_children(
    list: &deep::List,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Vec<deep::Expr> {
    let kids = children(list);
    if kids.len() < 2 {
        return kids.to_vec();
    }

    let mut let_env = env.clone();
    let annotated_bind = if let deep::Expr::List(bind_list, bind_span) = &kids[0] {
        let bind_kids = children(bind_list);
        let mut bind_elements = vec![bind_list.elements[0].clone(), bind_list.elements[1].clone()];
        let mut i = 0;
        while i + 1 < bind_kids.len() {
            bind_elements.push(bind_kids[i].clone());
            let value = annotate_expr_with_scope(&bind_kids[i + 1], &let_env, vg, subst, adt_reg);
            let value_ty = infer_expr_in_scope(&bind_kids[i + 1], &let_env, vg, subst, adt_reg);
            if let Some(name) = symbol_name(&bind_kids[i]) {
                let_env.bind(name.to_string(), let_env.generalize(&value_ty, subst));
            }
            bind_elements.push(value);
            i += 2;
        }
        deep::Expr::List(
            deep::List {
                elements: bind_elements,
            },
            *bind_span,
        )
    } else {
        annotate_expr_with_scope(&kids[0], env, vg, subst, adt_reg)
    };

    vec![
        annotated_bind,
        annotate_expr_with_scope(&kids[1], &let_env, vg, subst, adt_reg),
    ]
}

fn annotate_match_children(
    list: &deep::List,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Vec<deep::Expr> {
    let kids = children(list);
    if kids.is_empty() {
        return vec![];
    }

    let scrutinee = annotate_expr_with_scope(&kids[0], env, vg, subst, adt_reg);
    let scrutinee_ty = infer_expr_in_scope(&kids[0], env, vg, subst, adt_reg);
    let mut result = vec![scrutinee];

    for arm in &kids[1..] {
        if let deep::Expr::List(arm_list, arm_span) = arm
            && get_tag(arm_list) == Some("arm")
        {
            let arm_kids = children(arm_list);
            let mut arm_env = env.clone();
            let mut pattern_vg = vg.clone();
            let mut pattern_subst = subst.clone();
            let mut pattern_errors = Vec::new();
            let mut covered = Vec::new();
            let mut wildcard = false;
            if let Some(pattern) = arm_kids.first() {
                pattern_bindings(
                    pattern,
                    &scrutinee_ty,
                    &mut arm_env,
                    &mut pattern_vg,
                    &mut pattern_subst,
                    adt_reg,
                    &mut pattern_errors,
                    &mut covered,
                    &mut wildcard,
                );
            }

            let mut elements = vec![arm_list.elements[0].clone(), arm_list.elements[1].clone()];
            if let Some(pattern) = arm_kids.first() {
                elements.push(annotate_expr_with_scope(pattern, env, vg, subst, adt_reg));
            }
            if let Some(guard) = arm_kids.get(1) {
                elements.push(annotate_expr_with_scope(
                    guard, &arm_env, vg, subst, adt_reg,
                ));
            }
            if let Some(body) = arm_kids.get(2) {
                elements.push(annotate_expr_with_scope(body, &arm_env, vg, subst, adt_reg));
            }
            result.push(deep::Expr::List(deep::List { elements }, *arm_span));
            continue;
        }
        result.push(annotate_expr_with_scope(arm, env, vg, subst, adt_reg));
    }

    result
}

fn annotated_meta_map(
    list: &deep::List,
    expr: &deep::Expr,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> deep::Expr {
    let meta_span = match list.elements.get(1) {
        Some(deep::Expr::Map(_, span)) => *span,
        _ => span_of_expr(expr),
    };
    let mut entries = get_meta(list)
        .map(|meta| meta.entries.clone())
        .unwrap_or_default();

    if let Some(tag) = get_tag(list)
        && should_attach_type_metadata(tag)
    {
        let ty = infer_expr_in_scope(expr, env, vg, subst, adt_reg);
        if !matches!(ty, Type::Error) {
            let ty_expr = type_to_deep_expr(&ty);
            if let Some((_, existing)) = entries.iter_mut().find(|(key, _)| key == "type") {
                *existing = ty_expr;
            } else {
                entries.push(("type".to_string(), ty_expr));
            }
        }
    }

    deep::Expr::Map(deep::MetaMap { entries }, meta_span)
}

fn infer_expr_in_scope(
    expr: &deep::Expr,
    env: &Env,
    vg: &VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
) -> Type {
    let mut env = env.clone();
    let mut vg = vg.clone();
    let mut subst = subst.clone();
    let mut errors = Vec::new();
    let mut typed_nodes = 0;
    let mut total_nodes = 0;
    let ty = infer_expr(
        expr,
        &mut env,
        &mut vg,
        &mut subst,
        adt_reg,
        &mut errors,
        &mut typed_nodes,
        &mut total_nodes,
    );
    subst.apply(&ty)
}

fn should_attach_type_metadata(tag: &str) -> bool {
    !matches!(
        tag,
        "module"
            | "import"
            | "import-all"
            | "export"
            | "defsig"
            | "deftype"
            | "typealias"
            | "variant"
            | "field"
            | "defdim"
            | "params"
            | "bind"
            | "kv"
            | "arm"
            | "effects"
            | "resource"
            | "pat-var"
            | "pat-lit"
            | "pat-ctor"
            | "pat-tuple"
            | "pat-record"
            | "pat-wild"
            | "pat-as"
            | "t-prim"
            | "t-fn"
            | "t-tensor"
            | "t-adt"
            | "t-var"
            | "t-unit"
            | "t-tuple"
            | "d-name"
            | "d-var"
            | "d-lit"
    )
}

fn type_to_deep_expr(ty: &Type) -> deep::Expr {
    match ty {
        Type::Prim(prim) => node_expr("t-prim", vec![symbol_expr(prim.name())]),
        Type::Fn(args, ret) => {
            let mut children: Vec<deep::Expr> = args.iter().map(type_to_deep_expr).collect();
            children.push(type_to_deep_expr(ret));
            node_expr("t-fn", children)
        }
        Type::Tensor(dims, prim) => {
            let mut children: Vec<deep::Expr> = dims.iter().map(dim_to_deep_expr).collect();
            children.push(type_to_deep_expr(&Type::Prim(*prim)));
            node_expr("t-tensor", children)
        }
        Type::Adt(name, args) => {
            let mut children = vec![symbol_expr(name)];
            children.extend(args.iter().map(type_to_deep_expr));
            node_expr("t-adt", children)
        }
        Type::Var(var) => node_expr("t-var", vec![symbol_expr(&format!("t{}", var.0))]),
        Type::Tuple(types) => node_expr("t-tuple", types.iter().map(type_to_deep_expr).collect()),
        Type::Unit => node_expr("t-unit", vec![]),
        Type::Error => node_expr("t-var", vec![symbol_expr("_")]),
    }
}

fn dim_to_deep_expr(dim: &Dim) -> deep::Expr {
    match dim {
        Dim::Name(name) => node_expr("d-name", vec![symbol_expr(name)]),
        Dim::Var(var) => node_expr("d-var", vec![symbol_expr(&format!("d{}", var.0))]),
        Dim::Lit(value) => node_expr(
            "d-lit",
            vec![deep::Expr::Atom(deep::Atom::Int(*value), zero_span())],
        ),
        Dim::Wildcard => node_expr("d-name", vec![symbol_expr("*")]),
    }
}

fn node_expr(tag: &str, children: Vec<deep::Expr>) -> deep::Expr {
    let mut elements = vec![
        symbol_expr(tag),
        deep::Expr::Map(deep::MetaMap::default(), zero_span()),
    ];
    elements.extend(children);
    deep::Expr::List(deep::List { elements }, zero_span())
}

fn symbol_expr(name: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Symbol(name.to_string()), zero_span())
}

fn zero_span() -> Span {
    Span::new(0, 0)
}

fn span_of_expr(expr: &deep::Expr) -> Span {
    match expr {
        deep::Expr::Atom(_, span)
        | deep::Expr::List(_, span)
        | deep::Expr::Map(_, span)
        | deep::Expr::MetaExpr(_, span) => *span,
    }
}

fn span_of_list(list: &deep::List) -> Span {
    list.elements
        .first()
        .map(span_of_expr)
        .unwrap_or_else(zero_span)
}

fn phase0e_builtin_name(list: &deep::List) -> Option<&str> {
    let func_expr = list.elements.get(2)?;
    let func_list = match func_expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    match (func_list.elements.first(), func_list.elements.get(2)) {
        (
            Some(deep::Expr::Atom(deep::Atom::Symbol(tag), _)),
            Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)),
        ) if tag == "var" => Some(name.as_str()),
        _ => None,
    }
}

fn is_phase0e_shape_sensitive_builtin(name: &str) -> bool {
    matches!(
        name,
        "matmul"
            | "softmax"
            | "mean"
            | "layer_norm"
            | "conv2d"
            | "sum"
            | "max_reduce"
            | "reshape"
            | "permute"
            | "expand"
            | "pad"
            | "shrink"
            | "stride"
    )
}

fn expr_type_expr(expr: &deep::Expr, type_env: &Phase0eTypeEnv) -> Option<deep::Expr> {
    match expr {
        deep::Expr::List(list, _) => {
            if let Some(meta) = get_meta(list)
                && let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type")
            {
                return Some(ty.clone());
            }
            if get_tag(list) == Some("var")
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                return type_env.get(name).cloned();
            }
            None
        }
        deep::Expr::MetaExpr(meta, _) => expr_type_expr(&meta.expr, type_env),
        _ => None,
    }
}

fn extend_phase0e_env_with_fn_params(
    fn_list: &deep::List,
    type_env: &Phase0eTypeEnv,
) -> Phase0eTypeEnv {
    let mut scoped = type_env.clone();
    let Some(params_expr) = children(fn_list).first() else {
        return scoped;
    };
    let deep::Expr::List(params_list, _) = params_expr else {
        return scoped;
    };
    if get_tag(params_list) != Some("params") {
        return scoped;
    }
    for param in children(params_list) {
        let deep::Expr::List(param_list, _) = param else {
            continue;
        };
        let Some(name) = param_list.elements.first().and_then(symbol_name) else {
            continue;
        };
        let Some(meta) = get_meta(param_list) else {
            continue;
        };
        let Some((_, ty)) = meta.entries.iter().find(|(k, _)| k == "type") else {
            continue;
        };
        scoped.insert(name.to_string(), ty.clone());
    }
    scoped
}

fn expr_tensor_type_is_concrete(expr: &deep::Expr, type_env: &Phase0eTypeEnv) -> bool {
    expr_type_expr(expr, type_env)
        .map(|ty| type_expr_is_phase0e_concrete(&ty))
        .unwrap_or(false)
}

fn validate_phase0e_builtin_symbolic_requirements(
    list: &deep::List,
    func_name: &str,
    type_env: &Phase0eTypeEnv,
    errors: &mut Vec<CheckError>,
) {
    match func_name {
        "conv2d" => {
            if !app_result_type_is_concrete(list) {
                errors.push(CheckError::new(
                    CheckErrorKind::Other,
                    "Phase 0e builtin `conv2d` requires concrete output tensor dimensions"
                        .to_string(),
                    vec!["Use concrete d-lit dimensions for Phase 0e lowering".to_string()],
                ));
            }
            for arg in list.elements.iter().skip(3).take(2) {
                if !expr_tensor_type_is_concrete(arg, type_env) {
                    errors.push(CheckError::new(
                        CheckErrorKind::Other,
                        "Phase 0e builtin `conv2d` requires concrete tensor argument metadata"
                            .to_string(),
                        vec!["Use concrete d-lit dimensions for Phase 0e lowering".to_string()],
                    ));
                    break;
                }
            }
        }
        "mean" => {
            if phase0e_builtin_axis_dim(list, type_env, 0, 1) == Some(DeepDimKind::NonConcrete) {
                errors.push(CheckError::new(
                    CheckErrorKind::Other,
                    "Phase 0e builtin `mean` requires a concrete reduced axis extent".to_string(),
                    vec!["Use a concrete d-lit dimension on the reduced axis".to_string()],
                ));
            }
        }
        "layer_norm" => {
            let x_dims = list
                .elements
                .get(3)
                .and_then(|expr| expr_type_expr(expr, type_env))
                .and_then(|ty| tensor_dims_from_type_expr(&ty));
            if matches!(
                x_dims.as_ref().and_then(|dims| dims.last()),
                Some(DeepDimKind::NonConcrete)
            ) {
                errors.push(CheckError::new(
                    CheckErrorKind::Other,
                    "Phase 0e builtin `layer_norm` requires a concrete normalized axis extent"
                        .to_string(),
                    vec!["Use a concrete d-lit dimension for the final axis".to_string()],
                ));
            }
        }
        _ => {}
    }
}

fn phase0e_builtin_axis_dim(
    list: &deep::List,
    type_env: &Phase0eTypeEnv,
    tensor_arg_index: usize,
    axis_arg_index: usize,
) -> Option<DeepDimKind> {
    let tensor_dims = list
        .elements
        .get(3 + tensor_arg_index)
        .and_then(|expr| expr_type_expr(expr, type_env))
        .and_then(|ty| tensor_dims_from_type_expr(&ty))?;
    let axis = list
        .elements
        .get(3 + axis_arg_index)
        .and_then(extract_axis_literal)?;
    tensor_dims.get(axis).copied()
}

fn app_result_type_is_concrete(list: &deep::List) -> bool {
    get_meta(list)
        .and_then(|meta| meta.entries.iter().find(|(k, _)| k == "type"))
        .map(|(_, ty)| type_expr_is_phase0e_concrete(ty))
        .unwrap_or(false)
}

fn extract_axis_literal(expr: &deep::Expr) -> Option<usize> {
    match expr {
        deep::Expr::Atom(deep::Atom::Int(n), _) => Some(*n as usize),
        deep::Expr::List(list, _) => match list.elements.get(2) {
            Some(deep::Expr::Atom(deep::Atom::Int(n), _)) => Some(*n as usize),
            _ => None,
        },
        _ => None,
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DeepDimKind {
    Lit(i64),
    NonConcrete,
}

fn tensor_dims_from_type_expr(expr: &deep::Expr) -> Option<Vec<DeepDimKind>> {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(list) != Some("t-tensor") {
        return None;
    }
    let kids = children(list);
    if kids.is_empty() {
        return None;
    }
    let mut dims = Vec::new();
    for kid in &kids[..kids.len().saturating_sub(1)] {
        dims.push(match kid {
            deep::Expr::List(dim_list, _) if get_tag(dim_list) == Some("d-lit") => {
                match children(dim_list).first() {
                    Some(deep::Expr::Atom(deep::Atom::Int(n), _)) => DeepDimKind::Lit(*n),
                    _ => DeepDimKind::NonConcrete,
                }
            }
            _ => DeepDimKind::NonConcrete,
        });
    }
    Some(dims)
}

fn type_expr_is_phase0e_concrete(expr: &deep::Expr) -> bool {
    match expr {
        deep::Expr::List(list, _) if get_tag(list) == Some("t-prim") => true,
        _ => tensor_dims_from_type_expr(expr)
            .map(|dims| dims.iter().all(|d| matches!(d, DeepDimKind::Lit(_))))
            .unwrap_or(false),
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

fn resolve_type_aliases(ty: &Type, adt_reg: &AdtRegistry) -> Type {
    let mut seen = HashSet::new();
    resolve_type_aliases_inner(ty, adt_reg, &mut seen)
}

fn resolve_type_aliases_inner(
    ty: &Type,
    adt_reg: &AdtRegistry,
    seen: &mut HashSet<String>,
) -> Type {
    match ty {
        Type::Adt(name, args) => {
            let resolved_args: Vec<Type> = args
                .iter()
                .map(|arg| resolve_type_aliases_inner(arg, adt_reg, seen))
                .collect();

            if seen.contains(name) {
                return Type::Adt(name.clone(), resolved_args);
            }

            if let Some(expanded) = adt_reg.instantiate_alias(name, &resolved_args) {
                seen.insert(name.clone());
                let resolved = resolve_type_aliases_inner(&expanded, adt_reg, seen);
                seen.remove(name);
                resolved
            } else {
                Type::Adt(name.clone(), resolved_args)
            }
        }
        Type::Fn(args, ret) => Type::Fn(
            args.iter()
                .map(|a| resolve_type_aliases_inner(a, adt_reg, seen))
                .collect(),
            Box::new(resolve_type_aliases_inner(ret, adt_reg, seen)),
        ),
        Type::Tuple(ts) => Type::Tuple(
            ts.iter()
                .map(|t| resolve_type_aliases_inner(t, adt_reg, seen))
                .collect(),
        ),
        _ => ty.clone(),
    }
}

fn deep_type_to_resolved_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    tvar_map: &mut HashMap<String, TypeVar>,
) -> Type {
    let ty = deep_type_to_type(expr, vg, tvar_map);
    resolve_type_aliases(&ty, adt_reg)
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
                let ty = deep_type_to_resolved_type(&kids[1], vg, adt_reg, &mut HashMap::new());
                let scheme = env.generalize(&ty, subst);
                env.bind(name.to_string(), scheme);
            }
        }
        "typealias" => {
            // (typealias {} Name (params...) type_expr)
            if kids.len() >= 3
                && let Some(name) = symbol_name(&kids[0])
            {
                let params = match &kids[1] {
                    deep::Expr::List(list, _) => list
                        .elements
                        .iter()
                        .filter_map(symbol_name)
                        .map(str::to_string)
                        .collect::<Vec<_>>(),
                    _ => Vec::new(),
                };

                let mut tvar_map = HashMap::new();
                let mut param_vars = Vec::with_capacity(params.len());
                for param in &params {
                    let tv = vg.fresh_tvar();
                    tvar_map.insert(param.clone(), tv);
                    param_vars.push(tv);
                }

                let aliased_ty = deep_type_to_type(&kids[2], vg, &mut tvar_map);
                adt_reg.register_alias(name.to_string(), params, param_vars, aliased_ty);
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

    // Skip deftype/defsig/typealias (already processed in first pass)
    if tag == "deftype" || tag == "defsig" || tag == "typealias" {
        return;
    }

    let kids = children(list);

    if tag == "def" && kids.len() >= 2 {
        let name = match symbol_name(&kids[0]) {
            Some(n) => n.to_string(),
            None => return,
        };

        // Save declared type from defsig BEFORE inferring (it may get overwritten)
        let declared_ty = env.lookup(&name).map(|s| {
            let s = s.clone();
            env.instantiate(&s, vg)
        });

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

        // Enforce defsig: body must match declared signature
        if let Some(decl_ty) = declared_ty
            && let Err(_te) = unify(&body_ty, &decl_ty, subst)
        {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("def '{}' body doesn't match declared signature", name),
                vec![],
            ));
        }

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
                Some("lit") => infer_lit(list, vg, adt_reg),
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
                Some("deftype") | Some("typealias") => {
                    // Already handled in first pass
                    Type::Unit
                }
                Some("par") => {
                    // par: evaluate all children, return type of last (v1: sequential)
                    let kids = children(list);
                    let mut last_ty = Type::Unit;
                    for kid in kids {
                        last_ty = infer_expr(
                            kid,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        );
                    }
                    last_ty
                }
                Some("realize") => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        infer_expr(
                            inner,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        )
                    } else {
                        Type::Error
                    }
                }
                Some("copy") => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        let inner_ty = infer_expr(
                            inner,
                            env,
                            vg,
                            subst,
                            adt_reg,
                            errors,
                            typed_nodes,
                            total_nodes,
                        );
                        let resolved = subst.apply(&inner_ty);
                        match resolved {
                            Type::Tensor(_, _) | Type::Error => inner_ty,
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    format!("copy requires tensor input, got {resolved}"),
                                    vec!["Wrap only tensor values in copy".to_string()],
                                ));
                                Type::Error
                            }
                        }
                    } else {
                        Type::Error
                    }
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
            errors.push(CheckError::new(
                CheckErrorKind::UnboundVariable,
                format!("unbound variable: {name}"),
                vec![format!("Check spelling of '{}'", name)],
            ));
            Type::Error
        }
    } else {
        Type::Error
    }
}

fn infer_lit(list: &deep::List, vg: &mut VarGen, adt_reg: &AdtRegistry) -> Type {
    let meta = get_meta(list);
    let kids = children(list);

    // Check metadata for type annotation
    if let Some(meta) = meta {
        for (key, val) in &meta.entries {
            if key == "type" {
                return deep_type_to_resolved_type(val, vg, adt_reg, &mut HashMap::new());
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

    const TENSOR_OPS: &[&str] = &[
        "add",
        "mul",
        "sub",
        "div",
        "neg",
        "exp",
        "log",
        "sin",
        "sqrt",
        "relu",
        "sigmoid",
        "matmul",
        "layer_norm",
        "max_elem",
        "min_elem",
        "normalize",
        "cmplt",
        "eq",
        "neq",
        "gt",
        "lte",
        "gte",
        "and",
        "or",
        "not",
    ];

    const LOGICAL_OPS: &[&str] = &["and", "or", "not"];

    match unify(&func_ty, &expected_fn, subst) {
        Ok(()) => {
            let mut result_ty = subst.apply(&ret_tv);

            // Post-check: tensor ops require tensor arguments
            if let Some(ref fname) = func_name
                && TENSOR_OPS.contains(&fname.as_str())
            {
                for arg_ty in &arg_tys {
                    let resolved = subst.apply(arg_ty);
                    match &resolved {
                        Type::Tensor(_, _) | Type::Var(_) | Type::Error => {} // OK
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                format!("{} expects tensor arguments, got {}", fname, resolved),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }
            }

            if let Some(ref fname) = func_name
                && matches!(fname.as_str(), "softmax" | "mean" | "sum" | "max_reduce")
            {
                if let Some(first_arg) = arg_tys.first() {
                    let resolved = subst.apply(first_arg);
                    match &resolved {
                        Type::Tensor(_, _) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                format!("{} expects tensor input, got {}", fname, resolved),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }

                if let Some(axis_arg) = arg_tys.get(1) {
                    let resolved = subst.apply(axis_arg);
                    match &resolved {
                        Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                format!("{} expects int32 axis, got {}", fname, resolved),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }
            }

            if let Some(ref fname) = func_name
                && fname == "dropout"
            {
                if let Some(first_arg) = arg_tys.first() {
                    let resolved = subst.apply(first_arg);
                    match &resolved {
                        Type::Tensor(_, _) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                format!("dropout expects tensor input, got {}", resolved),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }

                if let Some(rate_arg) = arg_tys.get(1) {
                    let resolved = subst.apply(rate_arg);
                    match &resolved {
                        Type::Prim(Prim::F32) | Type::Var(_) | Type::Error => {}
                        _ => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                format!("dropout expects f32 rate, got {}", resolved),
                                vec![],
                            ));
                            return Type::Error;
                        }
                    }
                }
            }

            if let Some(ref fname) = func_name
                && fname == "conv2d"
            {
                for (index, arg_ty) in arg_tys.iter().enumerate() {
                    let resolved = subst.apply(arg_ty);
                    if index < 2 {
                        match &resolved {
                            Type::Tensor(_, _) | Type::Var(_) | Type::Error => {}
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    format!(
                                        "conv2d expects tensor inputs for args 1-2, got {}",
                                        resolved
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    } else {
                        match &resolved {
                            Type::Prim(Prim::Int32) | Type::Var(_) | Type::Error => {}
                            _ => {
                                errors.push(CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    format!(
                                        "conv2d expects int32 stride/padding, got {}",
                                        resolved
                                    ),
                                    vec![],
                                ));
                                return Type::Error;
                            }
                        }
                    }
                }
            }

            if let Some(ref fname) = func_name {
                match fname.as_str() {
                    "matmul" => {
                        result_ty = check_matmul_signature(&arg_tys, &result_ty, subst, errors);
                    }
                    "sum" | "max_reduce" | "mean" => {
                        result_ty = check_reduction_signature(
                            fname,
                            &kids[1..],
                            &arg_tys,
                            &result_ty,
                            subst,
                            errors,
                        );
                    }
                    "expand" => {
                        result_ty =
                            check_expand_signature(&kids[1..], &arg_tys, &result_ty, subst, errors);
                    }
                    "layer_norm" => {
                        result_ty =
                            check_layer_norm_signature(&arg_tys, &result_ty, vg, subst, errors);
                    }
                    "conv2d" => {
                        result_ty = check_conv2d_signature(&arg_tys, &result_ty, vg, subst, errors);
                    }
                    _ => {}
                }
            }

            // Post-check: logical ops require tensor[D, bool] arguments
            if let Some(ref fname) = func_name
                && LOGICAL_OPS.contains(&fname.as_str())
            {
                for arg_ty in &arg_tys {
                    let resolved = subst.apply(arg_ty);
                    match &resolved {
                        Type::Tensor(_, Prim::Bool) | Type::Var(_) | Type::Error => {} // OK
                        Type::Tensor(_, prec) => {
                            errors.push(CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                format!(
                                    "{} requires tensor[D, bool] arguments, got tensor[D, {}]",
                                    fname,
                                    prec.name()
                                ),
                                vec!["Logical ops only work on bool tensors".to_string()],
                            ));
                            return Type::Error;
                        }
                        _ => {} // Already caught by tensor-op check above
                    }
                }
            }

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

fn check_layer_norm_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    _vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 3 {
        return Type::Error;
    }

    let x_ty = subst.apply(&arg_tys[0]);
    let gamma_ty = subst.apply(&arg_tys[1]);
    let beta_ty = subst.apply(&arg_tys[2]);

    let (x_dims, x_prec) = match x_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("layer_norm expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let (gamma_dims, gamma_prec) = match gamma_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("layer_norm expects tensor gamma, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let (beta_dims, beta_prec) = match beta_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("layer_norm expects tensor beta, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if x_dims.is_empty() {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            "layer_norm expects rank >= 1 input tensor".to_string(),
            vec![],
        ));
        return Type::Error;
    }
    if gamma_dims.len() != 1 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "layer_norm expects rank-1 gamma, got rank {}",
                gamma_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if beta_dims.len() != 1 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "layer_norm expects rank-1 beta, got rank {}",
                beta_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if x_prec != gamma_prec || x_prec != beta_prec {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "layer_norm requires matching precisions, got {}, {}, {}",
                x_prec.name(),
                gamma_prec.name(),
                beta_prec.name()
            ),
            vec!["Insert explicit cast".to_string()],
        ));
        return Type::Error;
    }

    let hidden_dim = x_dims.last().cloned().expect("checked non-empty");
    if let Err(te) = unify_dim(&hidden_dim, &gamma_dims[0], subst) {
        errors.push(te.into());
        return Type::Error;
    }
    if let Err(te) = unify_dim(&hidden_dim, &beta_dims[0], subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let canonical = Type::Tensor(
        x_dims.into_iter().map(|d| subst.apply_dim(&d)).collect(),
        x_prec,
    );
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

fn check_conv2d_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    vg: &mut VarGen,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() < 2 {
        return Type::Error;
    }

    let input_ty = subst.apply(&arg_tys[0]);
    let kernel_ty = subst.apply(&arg_tys[1]);

    let (input_dims, input_prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("conv2d expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };
    let (kernel_dims, kernel_prec) = match kernel_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("conv2d expects tensor kernel, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if input_dims.len() != 4 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "conv2d expects rank-4 input tensor, got rank {}",
                input_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if kernel_dims.len() != 4 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "conv2d expects rank-4 kernel tensor, got rank {}",
                kernel_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if input_prec != kernel_prec {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "conv2d requires matching input/kernel precision, got {} and {}",
                input_prec.name(),
                kernel_prec.name()
            ),
            vec!["Insert explicit cast".to_string()],
        ));
        return Type::Error;
    }
    if let Err(te) = unify_dim(&input_dims[1], &kernel_dims[1], subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let output_template = Type::Tensor(
        vec![
            subst.apply_dim(&input_dims[0]),
            subst.apply_dim(&kernel_dims[0]),
            Dim::Var(vg.fresh_dvar()),
            Dim::Var(vg.fresh_dvar()),
        ],
        input_prec,
    );
    if let Err(te) = unify(result_ty, &output_template, subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let resolved_output = subst.apply(&output_template);
    if let Type::Tensor(out_dims, out_prec) = &resolved_output {
        if out_dims.len() != 4 {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("conv2d result must be rank 4, got rank {}", out_dims.len()),
                vec![],
            ));
            return Type::Error;
        }
        if *out_prec != input_prec {
            errors.push(CheckError::new(
                CheckErrorKind::PrecisionMismatch,
                format!(
                    "conv2d result precision must match input/kernel precision {}, got {}",
                    input_prec.name(),
                    out_prec.name()
                ),
                vec!["Insert explicit cast".to_string()],
            ));
            return Type::Error;
        }
        if let Err(te) = unify_dim(&out_dims[0], &input_dims[0], subst) {
            errors.push(te.into());
            return Type::Error;
        }
        if let Err(te) = unify_dim(&out_dims[1], &kernel_dims[0], subst) {
            errors.push(te.into());
            return Type::Error;
        }
    }

    subst.apply(&output_template)
}

fn check_matmul_signature(
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 2 {
        return Type::Error;
    }

    let lhs = subst.apply(&arg_tys[0]);
    let rhs = subst.apply(&arg_tys[1]);

    let (lhs_dims, lhs_prec) = match lhs {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("matmul expects tensor lhs, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };
    let (rhs_dims, rhs_prec) = match rhs {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("matmul expects tensor rhs, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if lhs_prec != rhs_prec {
        errors.push(CheckError::new(
            CheckErrorKind::PrecisionMismatch,
            format!(
                "matmul requires matching precisions, got {} and {}",
                lhs_prec.name(),
                rhs_prec.name()
            ),
            vec!["Insert explicit cast".to_string()],
        ));
        return Type::Error;
    }
    if lhs_dims.len() != 2 || rhs_dims.len() != 2 {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "matmul expects rank-2 tensors, got rank {} and {}",
                lhs_dims.len(),
                rhs_dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }
    if let Err(te) = unify_dim(&lhs_dims[1], &rhs_dims[0], subst) {
        errors.push(te.into());
        return Type::Error;
    }

    let canonical = Type::Tensor(
        vec![subst.apply_dim(&lhs_dims[0]), subst.apply_dim(&rhs_dims[1])],
        lhs_prec,
    );
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

fn check_reduction_signature(
    name: &str,
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 2 {
        return Type::Error;
    }

    let input_ty = subst.apply(&arg_tys[0]);
    let (dims, prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("{name} expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let axis = match arg_exprs.get(1).and_then(extract_int_literal) {
        Some(axis) if axis >= 0 => axis as usize,
        Some(axis) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("{name} requires non-negative axis, got {axis}"),
                vec![],
            ));
            return Type::Error;
        }
        None => return subst.apply(result_ty),
    };

    if axis >= dims.len() {
        errors.push(CheckError::new(
            CheckErrorKind::DimensionMismatch,
            format!(
                "{name} axis {axis} is out of bounds for rank {} tensor",
                dims.len()
            ),
            vec![],
        ));
        return Type::Error;
    }

    let mut out_dims = dims;
    out_dims.remove(axis);
    let canonical = Type::Tensor(out_dims, prec);
    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

fn check_expand_signature(
    arg_exprs: &[deep::Expr],
    arg_tys: &[Type],
    result_ty: &Type,
    subst: &mut Subst,
    errors: &mut Vec<CheckError>,
) -> Type {
    if arg_tys.len() != 3 {
        return Type::Error;
    }

    let input_ty = subst.apply(&arg_tys[0]);
    let (input_dims, input_prec) = match input_ty {
        Type::Tensor(dims, prec) => (dims, prec),
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expand expects tensor input, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    let axis = match arg_exprs.get(1).and_then(extract_int_literal) {
        Some(axis) if axis >= 0 => axis as usize,
        Some(axis) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("expand requires non-negative axis, got {axis}"),
                vec![],
            ));
            return Type::Error;
        }
        None => return subst.apply(result_ty),
    };
    let size = match arg_exprs.get(2).and_then(extract_int_literal) {
        Some(size) if size > 0 => Dim::Lit(size),
        Some(size) => {
            errors.push(CheckError::new(
                CheckErrorKind::DimensionMismatch,
                format!("expand requires positive size, got {size}"),
                vec![],
            ));
            return Type::Error;
        }
        None => return subst.apply(result_ty),
    };

    let resolved_result = subst.apply(result_ty);
    let canonical = match resolved_result {
        Type::Tensor(out_dims, out_prec) => {
            if out_prec != input_prec {
                errors.push(CheckError::new(
                    CheckErrorKind::PrecisionMismatch,
                    format!(
                        "expand output precision {} does not match input precision {}",
                        out_prec.name(),
                        input_prec.name()
                    ),
                    vec![],
                ));
                return Type::Error;
            }
            if out_dims.len() == input_dims.len() + 1 {
                if axis > input_dims.len() {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "expand insert axis {axis} is out of bounds for rank {} tensor",
                            input_dims.len()
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                let mut expected = input_dims.clone();
                expected.insert(axis, size.clone());
                Type::Tensor(expected, input_prec)
            } else if out_dims.len() == input_dims.len() {
                if axis >= input_dims.len() {
                    errors.push(CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        format!(
                            "expand axis {axis} is out of bounds for rank {} tensor",
                            input_dims.len()
                        ),
                        vec![],
                    ));
                    return Type::Error;
                }
                let mut expected = input_dims.clone();
                expected[axis] = size.clone();
                Type::Tensor(expected, input_prec)
            } else {
                errors.push(CheckError::new(
                    CheckErrorKind::DimensionMismatch,
                    format!(
                        "expand output rank {} must equal input rank {} or {}",
                        out_dims.len(),
                        input_dims.len(),
                        input_dims.len() + 1
                    ),
                    vec![],
                ));
                return Type::Error;
            }
        }
        Type::Var(_) | Type::Error => return subst.apply(result_ty),
        other => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expand expects tensor output, got {other}"),
                vec![],
            ));
            return Type::Error;
        }
    };

    if let Err(te) = unify(result_ty, &canonical, subst) {
        errors.push(te.into());
        return Type::Error;
    }
    subst.apply(&canonical)
}

fn extract_int_literal(expr: &deep::Expr) -> Option<i64> {
    match expr {
        deep::Expr::Atom(deep::Atom::Int(n), _) => Some(*n),
        deep::Expr::List(list, _) if get_tag(list) == Some("lit") => {
            children(list).first().and_then(|child| match child {
                deep::Expr::Atom(deep::Atom::Int(n), _) => Some(*n),
                _ => None,
            })
        }
        _ => None,
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
    let params = extract_params(&kids[0], vg, adt_reg);
    let mut param_types = Vec::new();
    let mut fn_env = env.clone();

    for (pname, ty_ann) in &params {
        let ty = ty_ann.clone().unwrap_or_else(|| vg.fresh_type());
        fn_env.bind(pname.clone(), Scheme::mono(ty.clone()));
        param_types.push(ty);
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

/// Extract parameter names (and optional type annotations) from (params {} x1 ... xn).
/// Each param can be a bare symbol or `(name {type: T})` for a typed param.
fn extract_params(
    expr: &deep::Expr,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
) -> Vec<(String, Option<Type>)> {
    match expr {
        deep::Expr::List(list, _) => {
            let tag = get_tag(list);
            let elems = if tag == Some("params") {
                children(list)
            } else {
                &list.elements
            };
            elems
                .iter()
                .filter_map(|e| match e {
                    deep::Expr::Atom(deep::Atom::Symbol(s), _) => Some((s.to_string(), None)),
                    deep::Expr::List(plist, _) => {
                        // Typed param: (name {type: T}) — elements[0] is the name symbol,
                        // elements[1] is the metadata map with type annotation
                        if let Some(deep::Expr::Atom(deep::Atom::Symbol(name), _)) =
                            plist.elements.first()
                        {
                            let mut ty_ann = None;
                            if let Some(deep::Expr::Map(meta, _)) = plist.elements.get(1) {
                                for (key, val) in &meta.entries {
                                    if key == "type" {
                                        ty_ann = Some(deep_type_to_resolved_type(
                                            val,
                                            vg,
                                            adt_reg,
                                            &mut HashMap::new(),
                                        ));
                                    }
                                }
                            }
                            Some((name.to_string(), ty_ann))
                        } else {
                            None
                        }
                    }
                    _ => None,
                })
                .collect()
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
    if let Err(_te) = unify(&cond_ty, &Type::Prim(Prim::Bool), subst) {
        errors.push(CheckError::new(
            CheckErrorKind::TypeMismatch,
            format!("if condition must be bool, got {}", subst.apply(&cond_ty)),
            vec![],
        ));
    }

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
    let mut has_wildcard = false;

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
                    adt_reg,
                    errors,
                    &mut covered_variants,
                    &mut has_wildcard,
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

    // Exhaustiveness check (wildcard covers everything)
    if !has_wildcard {
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
                errors.push(CheckError::new(
                    CheckErrorKind::NonExhaustiveMatch,
                    format!("non-exhaustive match: missing variants {:?}", names),
                    vec![],
                ));
            }
        }
    }

    result_ty.unwrap_or(Type::Error)
}

#[allow(clippy::too_many_arguments)]
fn pattern_bindings(
    pat: &deep::Expr,
    scrutinee_ty: &Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut Vec<CheckError>,
    covered_variants: &mut Vec<String>,
    has_wildcard: &mut bool,
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
                // Wildcard covers everything
                *has_wildcard = true;
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
                                            adt_reg,
                                            errors,
                                            covered_variants,
                                            has_wildcard,
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
            "pat-as" => {
                // (pat-as {} name inner_pat): bind name to scrutinee type, recurse into inner_pat
                if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                    let resolved = subst.apply(scrutinee_ty);
                    env.bind(name.to_string(), Scheme::mono(resolved));
                }
                if kids.len() >= 2 {
                    pattern_bindings(
                        &kids[1],
                        scrutinee_ty,
                        env,
                        vg,
                        subst,
                        adt_reg,
                        errors,
                        covered_variants,
                        has_wildcard,
                    );
                }
            }
            "pat-record" => {
                // (pat-record {} TypeName (kv {} k1 p1) ...): validate against ADT registry
                // kids[0] = TypeName, kids[1..] = (kv {} key pat)
                if let Some(ctor_name) = kids.first().and_then(|e| symbol_name(e)) {
                    covered_variants.push(ctor_name.to_string());

                    // Look up variant in ADT registry to get field types
                    let variant_info = adt_reg.lookup_variant(ctor_name);
                    let declared_fields: std::collections::HashMap<&str, &Type> = variant_info
                        .map(|(_, vi)| {
                            vi.fields
                                .iter()
                                .filter_map(|(name, ty)| name.as_deref().map(|n| (n, ty)))
                                .collect()
                        })
                        .unwrap_or_default();

                    for kv_expr in kids.iter().skip(1) {
                        if let deep::Expr::List(kv_list, _) = kv_expr
                            && get_tag(kv_list) == Some("kv")
                        {
                            let kv_kids = children(kv_list);
                            if kv_kids.len() >= 2 {
                                let field_name = symbol_name(&kv_kids[0]);
                                // Look up declared field type — reject unknown fields
                                let field_ty = match field_name {
                                    Some(n) => match declared_fields.get(n) {
                                        Some(ty) => (*ty).clone(),
                                        None if !declared_fields.is_empty() => {
                                            // Unknown field name — error
                                            errors.push(CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                format!(
                                                    "unknown record field '{}' in pattern for {}",
                                                    n, ctor_name
                                                ),
                                                vec![format!(
                                                    "known fields: {:?}",
                                                    declared_fields.keys().collect::<Vec<_>>()
                                                )],
                                            ));
                                            Type::Error
                                        }
                                        None => vg.fresh_type(), // no ADT info available
                                    },
                                    None => vg.fresh_type(),
                                };
                                pattern_bindings(
                                    &kv_kids[1],
                                    &field_ty,
                                    env,
                                    vg,
                                    subst,
                                    adt_reg,
                                    errors,
                                    covered_variants,
                                    has_wildcard,
                                );
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
                errors.push(CheckError::new(
                    CheckErrorKind::TupleIndexOutOfBounds,
                    format!(
                        "tuple index {} out of bounds for tuple of size {}",
                        index,
                        elems.len()
                    ),
                    vec![],
                ));
                Type::Error
            }
        }
        Type::Error => Type::Error,
        _ => {
            errors.push(CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("expected tuple type, got {resolved}"),
                vec![],
            ));
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
    let new_prec = match deep_type_to_resolved_type(&kids[1], vg, adt_reg, &mut HashMap::new()) {
        Type::Prim(p) => p,
        _ => return Type::Error,
    };

    match resolved {
        Type::Tensor(dims, _) => Type::Tensor(dims, new_prec),
        Type::Prim(_) => Type::Prim(new_prec),
        Type::Error => Type::Error,
        _ => {
            errors.push(CheckError::new(
                CheckErrorKind::CastNonTensor,
                format!("cast requires tensor or prim type, got {resolved}"),
                vec![],
            ));
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
            let ret = *ret;
            if !grad_output_supported(&ret) {
                errors.push(CheckError::new(
                    CheckErrorKind::Other,
                    format!("grad requires a scalar floating output, got {}", ret),
                    vec!["Reduce the function result to a scalar before applying grad".to_string()],
                ));
                return Type::Error;
            }

            let grad_payload = grad_argument_payload(&args);
            let grad_ret = Type::Tuple(vec![ret, grad_payload]);
            Type::Fn(args, Box::new(grad_ret))
        }
        Type::Error => Type::Error,
        _ => {
            // Can't determine function structure, return fresh var
            vg.fresh_type()
        }
    }
}

fn grad_output_supported(ty: &Type) -> bool {
    match ty {
        Type::Prim(prim) => prim.is_float(),
        Type::Tensor(dims, prim) => dims.is_empty() && prim.is_float(),
        _ => false,
    }
}

fn grad_argument_payload(args: &[Type]) -> Type {
    if args.len() == 1 {
        grad_argument_type(&args[0])
    } else {
        Type::Tuple(args.iter().map(grad_argument_type).collect())
    }
}

fn grad_argument_type(arg: &Type) -> Type {
    match arg {
        Type::Prim(prim) if prim.is_float() => Type::Prim(*prim),
        Type::Tensor(dims, prim) if prim.is_float() => Type::Tensor(dims.clone(), *prim),
        _ => Type::Unit,
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
/// `dvar_map` maps dimension variable names to DimVars (created on demand).
fn deep_type_to_type(
    expr: &deep::Expr,
    vg: &mut VarGen,
    tvar_map: &mut HashMap<String, TypeVar>,
) -> Type {
    let mut dvar_map = HashMap::new();
    deep_type_to_type_inner(expr, vg, tvar_map, &mut dvar_map)
}

fn deep_type_to_type_inner(
    expr: &deep::Expr,
    vg: &mut VarGen,
    tvar_map: &mut HashMap<String, TypeVar>,
    dvar_map: &mut HashMap<String, DimVar>,
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
                        .map(|c| deep_type_to_type_inner(c, vg, tvar_map, dvar_map))
                        .collect();
                    let ret =
                        deep_type_to_type_inner(&kids[kids.len() - 1], vg, tvar_map, dvar_map);
                    Type::Fn(args, Box::new(ret))
                }
                "t-tensor" => {
                    if kids.is_empty() {
                        return Type::Error;
                    }
                    let prec_expr = &kids[kids.len() - 1];
                    let prec = match deep_type_to_type_inner(prec_expr, vg, tvar_map, dvar_map) {
                        Type::Prim(p) => p,
                        _ => return Type::Error,
                    };
                    let dims: Vec<Dim> = kids[..kids.len() - 1]
                        .iter()
                        .filter_map(|c| parse_dim(c, vg, dvar_map))
                        .collect();
                    Type::Tensor(dims, prec)
                }
                "t-adt" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        let args: Vec<Type> = kids[1..]
                            .iter()
                            .map(|c| deep_type_to_type_inner(c, vg, tvar_map, dvar_map))
                            .collect();
                        Type::Adt(name.to_string(), args)
                    } else {
                        Type::Error
                    }
                }
                "t-tuple" => {
                    let elems: Vec<Type> = kids
                        .iter()
                        .map(|c| deep_type_to_type_inner(c, vg, tvar_map, dvar_map))
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
fn parse_dim(
    expr: &deep::Expr,
    vg: &mut VarGen,
    dvar_map: &mut HashMap<String, DimVar>,
) -> Option<Dim> {
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
                "d-var" => {
                    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
                        let dv = *dvar_map
                            .entry(name.to_string())
                            .or_insert_with(|| vg.fresh_dvar());
                        Some(Dim::Var(dv))
                    } else {
                        Some(vg.fresh_dim())
                    }
                }
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

    #[test]
    fn cast_accepts_alias_precision() {
        check_ok(
            "(typealias {} Floaty () (t-prim {} f32))
             (def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (cast {} (var {} x) (t-adt {} Floaty)))",
        );
    }

    #[test]
    fn copy_accepts_tensor() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} y (copy {} (var {} x)))",
        );
    }

    #[test]
    fn copy_rejects_scalar() {
        check_err(
            "(def {} x (lit {type: (t-prim {} int32)} 42))
             (def {} y (copy {} (var {} x)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn copy_rejects_unconstrained_generic() {
        check_err(
            "(def {} id (fn {} (params {} x) (copy {} (var {} x))))",
            CheckErrorKind::TypeMismatch,
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

    #[test]
    fn grad_non_float_param_gets_unit_gradient() {
        let exprs = chelis_deep::parser::parse_str(
            "(defsig {} f (t-fn {} (t-prim {} bool) (t-prim {} f32)))
             (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 1.0)))
             (def {} g (grad {} (var {} f)))",
        )
        .unwrap();
        let result = infer_program(&exprs);
        assert!(
            result.errors.is_empty(),
            "unexpected errors: {:?}",
            result.errors
        );
    }

    #[test]
    fn grad_rejects_non_scalar_output() {
        check_err(
            "(defsig {} f (t-fn {}
                (t-prim {} f32)
                (t-tensor {} (d-lit {} 2) (t-prim {} f32))))
             (def {} f
                (fn {} (params {} x)
                  (lit {type: (t-tensor {} (d-lit {} 2) (t-prim {} f32))} 1.0)))
             (def {} g (grad {} (var {} f)))",
            CheckErrorKind::Other,
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

    #[test]
    fn typealias_zero_param_resolves_in_defsig() {
        check_ok(
            "(typealias {} Scalar () (t-prim {} f32))
             (defsig {} id (t-fn {} (t-adt {} Scalar) (t-adt {} Scalar)))
             (def {} id (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn typealias_parameterized_resolves_in_defsig() {
        check_ok(
            "(typealias {} Boxed (a) (t-tuple {} (t-var {} a)))
             (defsig {} wrap (t-fn {} (t-adt {} Boxed (t-prim {} f32)) (t-adt {} Boxed (t-prim {} f32))))
             (def {} wrap (fn {} (params {} x) (var {} x)))",
        );
    }

    #[test]
    fn typealias_resolves_in_typed_param_metadata() {
        check_ok(
            "(typealias {} Scalar () (t-prim {} f32))
             (def {} id (fn {} (params {} (x {type: (t-adt {} Scalar)})) (var {} x)))",
        );
    }

    #[test]
    fn typealias_resolves_in_literal_metadata() {
        check_ok(
            "(typealias {} Scalar () (t-prim {} f32))
             (def {} x (lit {type: (t-adt {} Scalar)} 1.0))",
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
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} hidden) (d-name {} classes) (t-prim {} f32))} 0))
             (def {} c (app {type: (t-tensor {} (d-name {} batch) (d-name {} classes) (t-prim {} f32))}
                 (var {} matmul) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn builtin_layer_norm() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
        );
    }

    #[test]
    fn builtin_layer_norm_rejects_rank2_gamma() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (d-name {} extra) (t-prim {} f32))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn builtin_layer_norm_rejects_precision_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} bf16))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    #[test]
    fn builtin_conv2d_accepts_int_stride_padding() {
        check_ok(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_c) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_c) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} conv2d) (var {} x) (var {} k) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 1)))",
        );
    }

    #[test]
    fn builtin_conv2d_rejects_channel_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_a) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_b) (d-lit {} 3) (d-lit {} 3) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} conv2d) (var {} x) (var {} k) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 1)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    #[test]
    fn builtin_conv2d_rejects_kernel_precision_mismatch() {
        check_err(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} in_c) (d-name {} h) (d-name {} w) (t-prim {} f32))} 0))
             (def {} k (lit {type: (t-tensor {} (d-name {} out_c) (d-name {} in_c) (d-lit {} 3) (d-lit {} 3) (t-prim {} bf16))} 0))
             (def {} y (app {} (var {} conv2d) (var {} x) (var {} k) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 1)))",
            CheckErrorKind::PrecisionMismatch,
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

    // ── Regression tests for bug fixes ──────────────────────────────

    // Fix 1: add(int32, int32) should fail — tensor ops require tensor args
    #[test]
    fn fix1_tensor_op_rejects_non_tensor_args() {
        check_err(
            "(def {} r (app {} (var {} add) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 2)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // Fix 2: unsound generalization — fn x -> let y = x in (y 1, y true) should fail
    #[test]
    fn fix2_unsound_generalization_rejected() {
        // x is a monomorphic param, y = x so y is also monomorphic.
        // Applying y to both int32 and bool should fail.
        check_err(
            "(def {} test \
               (fn {} (params {} x) \
                 (let {} (bind {} y (var {} x)) \
                   (tuple {} \
                     (app {} (var {} y) (lit {type: (t-prim {} int32)} 1)) \
                     (app {} (var {} y) (lit {type: (t-prim {} bool)} true))))))",
            CheckErrorKind::PrecisionMismatch,
        );
    }

    // Fix 3: defsig not enforced — body must match declared signature
    #[test]
    fn fix3_defsig_enforced() {
        check_err(
            "(defsig {} f (t-fn {} (t-prim {} int32) (t-prim {} int32))) \
             (def {} f (lit {type: (t-prim {} bool)} true))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // Fix 4: d-var names shared within a type — same d-var name maps to same DimVar
    #[test]
    fn fix4_dvar_names_shared() {
        // Declare a function requiring same dim 'a' in both args.
        // Call with tensor[batch,f32] and tensor[seq,f32] — should fail.
        check_err(
            "(defsig {} myfn \
               (t-fn {} \
                 (t-tensor {} (d-var {} a) (t-prim {} f32)) \
                 (t-tensor {} (d-var {} a) (t-prim {} f32)) \
                 (t-tensor {} (d-var {} a) (t-prim {} f32)))) \
             (def {} myfn (fn {} (params {} x y) (var {} x))) \
             (def {} result (app {} (var {} myfn) \
               (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0) \
               (lit {type: (t-tensor {} (d-name {} seq) (t-prim {} f32))} 0)))",
            CheckErrorKind::DimensionMismatch,
        );
    }

    // Fix 5: if condition must be bool
    #[test]
    fn fix5_if_condition_must_be_bool() {
        check_err(
            "(if {} (lit {type: (t-prim {} int32)} 0) \
                    (lit {type: (t-prim {} int32)} 1) \
                    (lit {type: (t-prim {} int32)} 2))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // Fix 6a: wildcard satisfies exhaustiveness
    #[test]
    fn fix6a_wildcard_exhaustive() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None)) \
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42))) \
             (def {} result \
               (match {} (var {} x) \
                 (arm {} (pat-wild {}) () (lit {type: (t-prim {} int32)} 0))))",
        );
    }

    // Fix 6b: pat-as binds name
    #[test]
    fn fix6b_pat_as_binds_name() {
        check_ok(
            "(deftype {} Option (a) (variant {} Some (t-var {} a)) (variant {} None)) \
             (def {} x (app {} (var {} Some) (lit {type: (t-prim {} int32)} 42))) \
             (def {} result \
               (match {} (var {} x) \
                 (arm {} (pat-as {} whole (pat-wild {})) () (var {} whole))))",
        );
    }

    // Fix 7: fitness report includes unresolved names
    #[test]
    fn fix7_fitness_unresolved_names() {
        let exprs = chelis_deep::parser::parse_str("(def {} x (var {} unknown))").unwrap();
        let result = crate::infer::infer_program(&exprs);
        let report = crate::fitness::FitnessReport::from_infer_result(&result);
        assert!(
            report.unresolved_names.contains(&"unknown".to_string()),
            "expected 'unknown' in unresolved_names, got {:?}",
            report.unresolved_names
        );
        // Check severity is set
        assert!(report.errors.iter().all(|e| e.severity > 0.0));
    }

    // Fix 7b: suggestions populated for UnboundVariable
    #[test]
    fn fix7b_suggestions_for_unbound() {
        let result = check("(def {} x (var {} typo))");
        let unbound_err = result
            .errors
            .iter()
            .find(|e| matches!(e.kind, CheckErrorKind::UnboundVariable))
            .expect("expected UnboundVariable error");
        assert!(
            !unbound_err.suggestions.is_empty(),
            "expected suggestions for unbound variable"
        );
    }

    // Fix 8: typed params in Deep
    #[test]
    fn fix8_typed_params() {
        // fn with typed param x: f32 — using x should give f32
        check_ok("(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))");
    }

    // Fix 8b: typed param enforces type
    #[test]
    fn fix8b_typed_param_enforced() {
        // Param x is f32, but we try to add it (tensor op) — should fail
        check_err(
            "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) \
               (app {} (var {} add) (var {} x) (var {} x))))",
            CheckErrorKind::TypeMismatch,
        );
    }

    // ── Round 3 regression tests ──────────────────────────────────

    #[test]
    fn fix9_logical_ops_reject_non_bool_tensors() {
        // and(tensor[batch, f32], tensor[batch, f32]) should fail — requires bool
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
             (def {} r (app {} (var {} and) (var {} a) (var {} b)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix9b_logical_ops_reject_non_tensor() {
        // and(int32, int32) should fail — requires tensor
        check_err(
            "(def {} r (app {} (var {} and) (lit {type: (t-prim {} int32)} 1) (lit {type: (t-prim {} int32)} 2)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix9c_not_rejects_non_bool_tensor() {
        // not(tensor[batch, f32]) should fail — requires bool
        check_err(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0)) \
             (def {} r (app {} (var {} not) (var {} a)))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix9d_logical_ops_accept_bool_tensors() {
        // and(tensor[batch, bool], tensor[batch, bool]) should pass
        check_ok(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bool))} 0)) \
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} bool))} 0)) \
             (def {} r (app {} (var {} and) (var {} a) (var {} b)))",
        );
    }

    #[test]
    fn fix10_fitness_has_untyped_nodes() {
        let result = check(
            "(def {} good (lit {type: (t-prim {} int32)} 42)) \
                            (def {} bad (var {} nope))",
        );
        let report = crate::fitness::FitnessReport::from_infer_result(&result);
        assert!(report.untyped_nodes > 0, "expected untyped_nodes > 0");
        // Errors should have severity set
        assert!(
            report.errors.iter().all(|e| e.severity > 0.0),
            "all errors should have severity > 0"
        );
    }

    // ── Round 4 regression tests ──────────────────────────────────

    #[test]
    fn fix11_pat_record_rejects_unknown_field() {
        // Define Adam with field lr, then match on nonexistent field 'nope'
        check_err(
            "(deftype {} Optimizer () \
               (variant {} Adam (field {} lr (t-prim {} f32)))) \
             (def {} x (lit {type: (t-adt {} Optimizer)} 0)) \
             (def {} r \
               (match {} (var {} x) \
                 (arm {} (pat-record {} Adam (kv {} nope (pat-var {} v))) () (var {} v))))",
            CheckErrorKind::TypeMismatch,
        );
    }

    #[test]
    fn fix11b_pat_record_accepts_valid_field() {
        // Match on actual field lr — should pass
        check_ok(
            "(deftype {} Optimizer () \
               (variant {} Adam (field {} lr (t-prim {} f32)))) \
             (def {} x (lit {type: (t-adt {} Optimizer)} 0)) \
             (def {} r \
               (match {} (var {} x) \
                 (arm {} (pat-record {} Adam (kv {} lr (pat-var {} v))) () (var {} v))))",
        );
    }

    #[test]
    fn fix12_structure_score_measured() {
        // check_program runs tag validator — structure should be 1.0 for valid programs
        let exprs = chelis_deep::parser::parse_str("(def {} x (lit {type: (t-prim {} int32)} 42))")
            .unwrap();
        let report = crate::fitness::check_program(&exprs);
        assert!(
            (report.components.structure - 1.0).abs() < 0.01,
            "valid program structure should be ~1.0, got {}",
            report.components.structure
        );
    }

    #[test]
    fn checked_program_annotates_fn_bodies() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} f (fn {} (params {} (x {type: (t-prim {} f32)})) (var {} x)))",
        )
        .unwrap();
        let checked = check_phase0e_program(&exprs).expect("checked program");
        let text = chelis_deep::printer::print_canonical(checked.annotated_exprs());
        assert!(
            text.contains("(fn {type: (t-fn {} (t-prim {} f32) (t-prim {} f32))}"),
            "expected typed fn metadata, got:\n{text}"
        );
    }

    #[test]
    fn checked_program_annotates_apps_and_updates_type_env() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} a (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} b (lit {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))} 0))
             (def {} c (app {} (var {} add) (var {} a) (var {} b)))",
        )
        .unwrap();
        let checked = check_phase0e_program(&exprs).expect("checked program");
        let text = chelis_deep::printer::print_canonical(checked.annotated_exprs());
        assert!(
            text.contains("(app {type: (t-tensor {} (d-name {} batch) (t-prim {} f32))}"),
            "expected typed app metadata, got:\n{text}"
        );
        assert!(checked.type_env().contains_key("c"));
    }

    #[test]
    fn phase0e_rejects_symbolic_normalized_axis_for_layer_norm() {
        let exprs = chelis_deep::parser::parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-name {} batch) (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} gamma (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} beta (lit {type: (t-tensor {} (d-name {} hidden) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} layer_norm) (var {} x) (var {} gamma) (var {} beta)))",
        )
        .unwrap();
        let err =
            check_phase0e_program(&exprs).expect_err("symbolic hidden axis should be rejected");
        assert!(
            err.errors
                .iter()
                .any(|error| error.message.contains("concrete normalized axis extent")),
            "expected layer_norm symbolic normalized-axis error, got: {:?}",
            err.errors
        );
    }
}
