use std::collections::HashMap;

use chelis_deep::Span;
use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_types::CheckedProgram;
use chelis_types::types::{Effect, EffectSet};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectErrorKind {
    UnhandledEffect,
    InvalidHandler,
    BuildTargetMismatch,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectError {
    pub kind: EffectErrorKind,
    pub message: String,
    pub suggestions: Vec<String>,
}

pub fn check_program(program: &CheckedProgram) -> Result<CheckedProgram, Vec<EffectError>> {
    let effects_by_def = infer_program_effects(program.annotated_exprs());
    let annotated_exprs: Vec<Expr> = program
        .annotated_exprs()
        .iter()
        .map(|expr| annotate_effects(expr, &effects_by_def, &HashMap::new()))
        .collect();

    let mut errors = Vec::new();
    validate_handlers(&annotated_exprs, &mut errors);
    validate_unhandled_random_roots(&annotated_exprs, &effects_by_def, &mut errors);

    if errors.is_empty() {
        Ok(CheckedProgram::from_parts(
            annotated_exprs,
            program.type_env().clone(),
        ))
    } else {
        Err(errors)
    }
}

pub fn validate_build_target(
    program: &CheckedProgram,
    target: &str,
) -> Result<(), Vec<EffectError>> {
    let mut errors = Vec::new();
    for expr in program.annotated_exprs() {
        validate_build_target_expr(expr, target, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn infer_program_effects(exprs: &[Expr]) -> HashMap<String, EffectSet> {
    let bodies = top_level_def_bodies(exprs);
    let mut effects = HashMap::<String, EffectSet>::new();

    for _ in 0..=bodies.len() {
        let mut changed = false;
        for (name, body) in &bodies {
            let inferred = infer_expr_effects(body, &effects, &HashMap::new());
            if effects.get(name) != Some(&inferred) {
                effects.insert(name.clone(), inferred);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    effects
}

fn top_level_def_bodies(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut defs = HashMap::new();
    for expr in exprs {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
        {
            let kids = children(list);
            if kids.len() >= 2
                && let Some(name) = symbol_name(&kids[0])
            {
                defs.insert(name.to_string(), kids[1].clone());
            }
        }
    }
    defs
}

fn infer_expr_effects(
    expr: &Expr,
    top_level_effects: &HashMap<String, EffectSet>,
    locals: &HashMap<String, EffectSet>,
) -> EffectSet {
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => EffectSet::new(),
        Expr::MetaExpr(meta, _) => infer_expr_effects(&meta.expr, top_level_effects, locals),
        Expr::List(list, _) => {
            let Some(tag) = get_tag(list) else {
                return list
                    .elements
                    .iter()
                    .map(|elem| infer_expr_effects(elem, top_level_effects, locals))
                    .fold(EffectSet::new(), |mut acc, set| {
                        acc.extend(&set);
                        acc
                    });
            };

            match tag {
                "var" => children(list)
                    .first()
                    .and_then(symbol_name)
                    .and_then(|name| locals.get(name).or_else(|| top_level_effects.get(name)))
                    .cloned()
                    .unwrap_or_default(),
                "app" => infer_app_effects(list, top_level_effects, locals),
                "fn" => {
                    let kids = children(list);
                    kids.get(1)
                        .map(|body| infer_expr_effects(body, top_level_effects, locals))
                        .unwrap_or_default()
                }
                "let" => infer_let_effects(list, top_level_effects, locals),
                "if" | "tuple" | "pipe" | "par" | "record" | "access" | "tuple-get" | "cast"
                | "copy" | "realize" | "jit" | "match" | "def" => children(list)
                    .iter()
                    .map(|kid| infer_expr_effects(kid, top_level_effects, locals))
                    .fold(EffectSet::new(), |mut acc, set| {
                        acc.extend(&set);
                        acc
                    }),
                "grad" => children(list)
                    .first()
                    .map(|kid| infer_expr_effects(kid, top_level_effects, locals))
                    .unwrap_or_default(),
                "handle-effect" => infer_handle_effects(list, top_level_effects, locals),
                _ => children(list)
                    .iter()
                    .map(|kid| infer_expr_effects(kid, top_level_effects, locals))
                    .fold(EffectSet::new(), |mut acc, set| {
                        acc.extend(&set);
                        acc
                    }),
            }
        }
    }
}

fn infer_app_effects(
    list: &List,
    top_level_effects: &HashMap<String, EffectSet>,
    locals: &HashMap<String, EffectSet>,
) -> EffectSet {
    let kids = children(list);
    let mut effects = EffectSet::new();
    for kid in kids {
        effects.extend(&infer_expr_effects(kid, top_level_effects, locals));
    }

    let builtin_name = kids.first().and_then(var_name);
    if builtin_name == Some("dropout") {
        effects.insert(Effect::Random);
    }

    effects
}

fn infer_let_effects(
    list: &List,
    top_level_effects: &HashMap<String, EffectSet>,
    locals: &HashMap<String, EffectSet>,
) -> EffectSet {
    let kids = children(list);
    if kids.len() < 2 {
        return EffectSet::new();
    }

    let mut local_scope = locals.clone();
    let mut effects = EffectSet::new();

    if let Expr::List(bind_list, _) = &kids[0] {
        let bind_kids = children(bind_list);
        let mut i = 0;
        while i + 1 < bind_kids.len() {
            let value_effects =
                infer_expr_effects(&bind_kids[i + 1], top_level_effects, &local_scope);
            effects.extend(&value_effects);
            if let Some(name) = symbol_name(&bind_kids[i]) {
                local_scope.insert(name.to_string(), value_effects);
            }
            i += 2;
        }
    }

    effects.extend(&infer_expr_effects(
        &kids[1],
        top_level_effects,
        &local_scope,
    ));
    effects
}

fn infer_handle_effects(
    list: &List,
    top_level_effects: &HashMap<String, EffectSet>,
    locals: &HashMap<String, EffectSet>,
) -> EffectSet {
    let kids = children(list);
    if kids.len() < 2 {
        return EffectSet::new();
    }
    let mut effects = infer_expr_effects(&kids[0], top_level_effects, locals);
    let mut body_effects = infer_expr_effects(&kids[1], top_level_effects, locals);
    match effect_name(list) {
        Some("random") => body_effects.remove(&Effect::Random),
        Some("resource") => {}
        _ => {}
    }
    effects.extend(&body_effects);
    effects
}

fn annotate_effects(
    expr: &Expr,
    top_level_effects: &HashMap<String, EffectSet>,
    locals: &HashMap<String, EffectSet>,
) -> Expr {
    match expr {
        Expr::Atom(_, _) => expr.clone(),
        Expr::Map(map, span) => Expr::Map(
            MetaMap {
                entries: map
                    .entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            annotate_effects(value, top_level_effects, locals),
                        )
                    })
                    .collect(),
            },
            *span,
        ),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                expr: Box::new(annotate_effects(&meta.expr, top_level_effects, locals)),
                entries: meta
                    .entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            annotate_effects(value, top_level_effects, locals),
                        )
                    })
                    .collect(),
            },
            *span,
        ),
        Expr::List(list, span) => {
            let mut local_scope = locals.clone();
            let tag = get_tag(list);
            let kids = children(list);
            let annotated_children = match tag {
                Some("let") if kids.len() >= 2 => {
                    let bind = if let Expr::List(bind_list, bind_span) = &kids[0] {
                        let bind_kids = children(bind_list);
                        let mut elements =
                            vec![bind_list.elements[0].clone(), bind_list.elements[1].clone()];
                        let mut i = 0;
                        while i + 1 < bind_kids.len() {
                            elements.push(bind_kids[i].clone());
                            let value = annotate_effects(
                                &bind_kids[i + 1],
                                top_level_effects,
                                &local_scope,
                            );
                            let value_effects = infer_expr_effects(
                                &bind_kids[i + 1],
                                top_level_effects,
                                &local_scope,
                            );
                            if let Some(name) = symbol_name(&bind_kids[i]) {
                                local_scope.insert(name.to_string(), value_effects);
                            }
                            elements.push(value);
                            i += 2;
                        }
                        Expr::List(chelis_deep::ast::List { elements }, *bind_span)
                    } else {
                        annotate_effects(&kids[0], top_level_effects, &local_scope)
                    };
                    vec![
                        bind,
                        annotate_effects(&kids[1], top_level_effects, &local_scope),
                    ]
                }
                _ => kids
                    .iter()
                    .map(|kid| annotate_effects(kid, top_level_effects, &local_scope))
                    .collect(),
            };

            let mut elements = list.elements.clone();
            if matches!(elements.get(1), Some(Expr::Map(_, _))) {
                elements.truncate(2);
                elements.extend(annotated_children);
                if let Some(meta) = elements.get_mut(1) {
                    update_effect_metadata(meta, list, top_level_effects, locals);
                }
                Expr::List(chelis_deep::ast::List { elements }, *span)
            } else {
                Expr::List(
                    chelis_deep::ast::List {
                        elements: list
                            .elements
                            .iter()
                            .map(|elem| annotate_effects(elem, top_level_effects, locals))
                            .collect(),
                    },
                    *span,
                )
            }
        }
    }
}

fn update_effect_metadata(
    meta_expr: &mut Expr,
    list: &List,
    top_level_effects: &HashMap<String, EffectSet>,
    locals: &HashMap<String, EffectSet>,
) {
    let Expr::Map(meta, _) = meta_expr else {
        return;
    };
    if let Some("fn") = get_tag(list) {
        let effects = infer_expr_effects(
            &Expr::List(list.clone(), zero_span()),
            top_level_effects,
            locals,
        );
        if !effects.is_empty() {
            upsert_meta(meta, "effects", effect_set_expr(&effects));
        }
    }
}

fn validate_handlers(exprs: &[Expr], errors: &mut Vec<EffectError>) {
    for expr in exprs {
        validate_handler_expr(expr, errors);
    }
}

fn validate_handler_expr(expr: &Expr, errors: &mut Vec<EffectError>) {
    match expr {
        Expr::List(list, _) => {
            if get_tag(list) == Some("handle-effect") {
                let kids = children(list);
                match effect_name(list) {
                    Some("random") => {
                        if kids.first().and_then(int_literal).is_none() {
                            errors.push(EffectError {
                                kind: EffectErrorKind::InvalidHandler,
                                message: "with seed(...) currently requires an int literal seed"
                                    .to_string(),
                                suggestions: vec![
                                    "Use `with seed(42) { ... }` with an explicit integer seed"
                                        .to_string(),
                                ],
                            });
                        }
                    }
                    Some("resource") => {
                        if kids.first().and_then(string_literal).is_none() {
                            errors.push(EffectError {
                                kind: EffectErrorKind::InvalidHandler,
                                message:
                                    "with device(...) currently requires a string literal device"
                                        .to_string(),
                                suggestions: vec![
                                    "Use `with device(\"gpu:0\") { ... }` with an explicit device literal"
                                        .to_string(),
                                ],
                            });
                        }
                    }
                    _ => {}
                }
            }
            for kid in &list.elements {
                validate_handler_expr(kid, errors);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                validate_handler_expr(value, errors);
            }
        }
        Expr::MetaExpr(meta, _) => {
            validate_handler_expr(&meta.expr, errors);
            for (_, value) in &meta.entries {
                validate_handler_expr(value, errors);
            }
        }
        Expr::Atom(_, _) => {}
    }
}

fn validate_unhandled_random_roots(
    exprs: &[Expr],
    effects_by_def: &HashMap<String, EffectSet>,
    errors: &mut Vec<EffectError>,
) {
    for expr in exprs {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some("def")
        {
            let kids = children(list);
            if kids.len() < 2 {
                continue;
            }
            let Some(name) = symbol_name(&kids[0]) else {
                continue;
            };
            if matches!(&kids[1], Expr::List(body, _) if get_tag(body) == Some("fn")) {
                continue;
            }
            if effects_by_def
                .get(name)
                .is_some_and(|effects| effects.contains(&Effect::Random))
            {
                errors.push(EffectError {
                    kind: EffectErrorKind::UnhandledEffect,
                    message: format!(
                        "Function `{name}` has unhandled effect `Random`; `dropout` requires `with seed(...)`"
                    ),
                    suggestions: vec![
                        "Wrap the stochastic region with `with seed(42) { ... }`".to_string(),
                    ],
                });
            }
        }
    }
}

fn validate_build_target_expr(expr: &Expr, target: &str, errors: &mut Vec<EffectError>) {
    match expr {
        Expr::List(list, _) => {
            if get_tag(list) == Some("handle-effect")
                && effect_name(list) == Some("resource")
                && let Some(device) = children(list).first().and_then(string_literal)
            {
                let ok = match target {
                    "c" => !device.starts_with("gpu"),
                    "hip" => device.starts_with("gpu"),
                    _ => true,
                };
                if !ok {
                    errors.push(EffectError {
                        kind: EffectErrorKind::BuildTargetMismatch,
                        message: format!(
                            "`chelis build --target {target}` cannot satisfy resource region `{device}`"
                        ),
                        suggestions: match target {
                            "c" => vec![
                                "Use `with device(\"cpu\") { ... }` or build with `--target hip`"
                                    .to_string(),
                            ],
                            "hip" => vec![
                                "Use a GPU device such as `with device(\"gpu:0\") { ... }`"
                                    .to_string(),
                            ],
                            _ => vec![],
                        },
                    });
                }
            }
            for kid in &list.elements {
                validate_build_target_expr(kid, target, errors);
            }
        }
        Expr::Map(map, _) => {
            for (_, value) in &map.entries {
                validate_build_target_expr(value, target, errors);
            }
        }
        Expr::MetaExpr(meta, _) => {
            validate_build_target_expr(&meta.expr, target, errors);
            for (_, value) in &meta.entries {
                validate_build_target_expr(value, target, errors);
            }
        }
        Expr::Atom(_, _) => {}
    }
}

fn effect_set_expr(effects: &EffectSet) -> Expr {
    let mut elements = vec![
        symbol("effects"),
        Expr::Map(MetaMap::default(), zero_span()),
    ];
    for effect in effects.iter() {
        elements.push(match effect {
            Effect::Random => symbol("random"),
            Effect::Accum => symbol("accum"),
            Effect::Resource(device) => Expr::List(
                chelis_deep::ast::List {
                    elements: vec![
                        symbol("resource"),
                        Expr::Map(MetaMap::default(), zero_span()),
                        Expr::Atom(Atom::Str(device.clone()), zero_span()),
                    ],
                },
                zero_span(),
            ),
        });
    }
    Expr::List(chelis_deep::ast::List { elements }, zero_span())
}

fn upsert_meta(meta: &mut MetaMap, key: &str, value: Expr) {
    if let Some((_, existing)) = meta
        .entries
        .iter_mut()
        .find(|(entry_key, _)| entry_key == key)
    {
        *existing = value;
    } else {
        meta.entries.push((key.to_string(), value));
    }
}

fn var_name(expr: &Expr) -> Option<&str> {
    let Expr::List(list, _) = expr else {
        return None;
    };
    if get_tag(list) != Some("var") {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn effect_name(list: &List) -> Option<&str> {
    match list.elements.get(1) {
        Some(Expr::Map(meta, _)) => meta
            .entries
            .iter()
            .find(|(key, _)| key == "effect")
            .and_then(|(_, value)| match value {
                Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
                _ => None,
            }),
        _ => None,
    }
}

fn get_tag(list: &List) -> Option<&str> {
    match list.elements.first() {
        Some(Expr::Atom(Atom::Symbol(tag), _)) => Some(tag.as_str()),
        _ => None,
    }
}

fn children(list: &List) -> &[Expr] {
    if list.elements.len() > 2 {
        &list.elements[2..]
    } else {
        &[]
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Symbol(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn int_literal(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(value), _) => Some(*value),
        Expr::List(list, _) if get_tag(list) == Some("lit") => match children(list).first() {
            Some(Expr::Atom(Atom::Int(value), _)) => Some(*value),
            _ => None,
        },
        _ => None,
    }
}

fn string_literal(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Str(value), _) => Some(value.as_str()),
        Expr::List(list, _) if get_tag(list) == Some("lit") => match children(list).first() {
            Some(Expr::Atom(Atom::Str(value), _)) => Some(value.as_str()),
            _ => None,
        },
        _ => None,
    }
}

fn symbol(name: &str) -> Expr {
    Expr::Atom(Atom::Symbol(name.to_string()), zero_span())
}

fn zero_span() -> Span {
    Span::new(0, 0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use chelis_deep::parser::parse_str;

    fn checked(src: &str) -> CheckedProgram {
        let exprs = parse_str(src).unwrap();
        let checked = chelis_types::check_phase0e_program(&exprs).unwrap();
        check_program(&checked).unwrap()
    }

    #[test]
    fn infers_random_for_dropout_fn() {
        let exprs = parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        )
        .unwrap();
        let inferred = infer_program_effects(&exprs);
        assert!(
            inferred
                .get("y")
                .is_some_and(|effects| effects.contains(&Effect::Random))
        );
    }

    #[test]
    fn rejects_unhandled_dropout_root() {
        let exprs = parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        )
        .unwrap();
        let checked = chelis_types::check_phase0e_program(&exprs).unwrap();
        let errors = check_program(&checked).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.kind == EffectErrorKind::UnhandledEffect)
        );
    }

    #[test]
    fn build_target_rejects_gpu_region_for_c() {
        let program = checked(
            "(def {} x
               (handle-effect {effect: resource}
                 (lit {type: (t-prim {} string)} \"gpu:0\")
                 (lit {type: (t-prim {} int32)} 1)))",
        );
        let errors = validate_build_target(&program, "c").unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.kind == EffectErrorKind::BuildTargetMismatch)
        );
    }

    #[test]
    fn checked_program_does_not_emit_internal_type_override_metadata() {
        let program = checked(
            "(def {} f
               (fn {} (params {} (x {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))}))
                 (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5))))",
        );
        let text = chelis_deep::printer::print_canonical(program.annotated_exprs());
        assert!(
            !text.contains("__type_override"),
            "effect checker should not leak internal metadata, got:\n{text}"
        );
        assert!(
            text.contains("effects"),
            "expected inferred effect metadata on checked fn body, got:\n{text}"
        );
    }
}
