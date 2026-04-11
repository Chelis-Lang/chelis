use std::collections::{HashMap, HashSet};

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
    let (effects_by_def, top_level_callables) = infer_program_effects(program.annotated_exprs());
    let annotated_exprs: Vec<Expr> = program
        .annotated_exprs()
        .iter()
        .map(|expr| annotate_effects(expr, &effects_by_def, &top_level_callables, &HashMap::new()))
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

fn infer_program_effects(exprs: &[Expr]) -> (HashMap<String, EffectSet>, HashSet<String>) {
    let bodies = top_level_def_bodies(exprs);
    let top_level_callables = top_level_callable_names(&bodies);
    let mut effects = HashMap::<String, EffectSet>::new();

    for _ in 0..=bodies.len() {
        let mut changed = false;
        for (name, body) in &bodies {
            let inferred =
                infer_expr_effects(body, &effects, &top_level_callables, &HashMap::new());
            if effects.get(name) != Some(&inferred) {
                effects.insert(name.clone(), inferred);
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }

    (effects, top_level_callables)
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

fn top_level_callable_names(bodies: &HashMap<String, Expr>) -> HashSet<String> {
    bodies
        .iter()
        .filter_map(|(name, body)| match body {
            Expr::List(list, _) if get_tag(list) == Some("fn") => Some(name.clone()),
            _ => None,
        })
        .collect()
}

fn infer_expr_effects(
    expr: &Expr,
    top_level_effects: &HashMap<String, EffectSet>,
    top_level_callables: &HashSet<String>,
    locals: &HashMap<String, EffectSet>,
) -> EffectSet {
    match expr {
        Expr::Atom(_, _) | Expr::Map(_, _) => EffectSet::new(),
        Expr::MetaExpr(meta, _) => {
            infer_expr_effects(&meta.expr, top_level_effects, top_level_callables, locals)
        }
        Expr::List(list, _) => {
            let Some(tag) = get_tag(list) else {
                return list
                    .elements
                    .iter()
                    .map(|elem| {
                        infer_expr_effects(elem, top_level_effects, top_level_callables, locals)
                    })
                    .fold(EffectSet::new(), |mut acc, set| {
                        acc.extend(&set);
                        acc
                    });
            };

            match tag {
                "var" => children(list)
                    .first()
                    .and_then(symbol_name)
                    .and_then(|name| {
                        locals.get(name).cloned().or_else(|| {
                            top_level_callables
                                .contains(name)
                                .then(|| top_level_effects.get(name).cloned())
                                .flatten()
                        })
                    })
                    .unwrap_or_default(),
                "app" => infer_app_effects(list, top_level_effects, top_level_callables, locals),
                "fn" => {
                    let kids = children(list);
                    kids.get(1)
                        .map(|body| {
                            infer_expr_effects(body, top_level_effects, top_level_callables, locals)
                        })
                        .unwrap_or_default()
                }
                "let" => infer_let_effects(list, top_level_effects, top_level_callables, locals),
                "if" | "tuple" | "pipe" | "par" | "record" | "access" | "tuple-get" | "cast"
                | "copy" | "realize" | "jit" | "match" | "def" => children(list)
                    .iter()
                    .map(|kid| {
                        infer_expr_effects(kid, top_level_effects, top_level_callables, locals)
                    })
                    .fold(EffectSet::new(), |mut acc, set| {
                        acc.extend(&set);
                        acc
                    }),
                "grad" => children(list)
                    .first()
                    .map(|kid| {
                        infer_expr_effects(kid, top_level_effects, top_level_callables, locals)
                    })
                    .unwrap_or_default(),
                "vmap" => children(list)
                    .first()
                    .map(|kid| {
                        infer_expr_effects(kid, top_level_effects, top_level_callables, locals)
                    })
                    .unwrap_or_default(),
                "handle-effect" => {
                    infer_handle_effects(list, top_level_effects, top_level_callables, locals)
                }
                _ => children(list)
                    .iter()
                    .map(|kid| {
                        infer_expr_effects(kid, top_level_effects, top_level_callables, locals)
                    })
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
    top_level_callables: &HashSet<String>,
    locals: &HashMap<String, EffectSet>,
) -> EffectSet {
    let kids = children(list);
    let mut effects = EffectSet::new();
    for kid in kids {
        effects.extend(&infer_expr_effects(
            kid,
            top_level_effects,
            top_level_callables,
            locals,
        ));
    }

    let builtin_name = kids.first().and_then(var_name);
    if builtin_name == Some("dropout") {
        effects.insert(Effect::Random);
    }
    if matches!(
        builtin_name,
        Some(
            "print"
                | "debug"
                | "read_file"
                | "write_file"
                | "read_lines"
                | "read_bytes"
                | "file_exists"
                | "list_dir"
                | "mmap_file"
        )
    ) {
        effects.insert(Effect::Io);
    }

    effects
}

fn infer_let_effects(
    list: &List,
    top_level_effects: &HashMap<String, EffectSet>,
    top_level_callables: &HashSet<String>,
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
            let value = &bind_kids[i + 1];
            let value_effects =
                infer_expr_effects(value, top_level_effects, top_level_callables, &local_scope);
            effects.extend(&value_effects);
            if let Some(name) = symbol_name(&bind_kids[i]) {
                let binding_effects = match value {
                    Expr::List(value_list, _) if get_tag(value_list) == Some("fn") => value_effects,
                    _ => EffectSet::new(),
                };
                local_scope.insert(name.to_string(), binding_effects);
            }
            i += 2;
        }
    }

    effects.extend(&infer_expr_effects(
        &kids[1],
        top_level_effects,
        top_level_callables,
        &local_scope,
    ));
    effects
}

fn infer_handle_effects(
    list: &List,
    top_level_effects: &HashMap<String, EffectSet>,
    top_level_callables: &HashSet<String>,
    locals: &HashMap<String, EffectSet>,
) -> EffectSet {
    let kids = children(list);
    if kids.len() < 2 {
        return EffectSet::new();
    }
    let mut effects = infer_expr_effects(&kids[0], top_level_effects, top_level_callables, locals);
    let mut body_effects =
        infer_expr_effects(&kids[1], top_level_effects, top_level_callables, locals);
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
    top_level_callables: &HashSet<String>,
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
                            annotate_effects(value, top_level_effects, top_level_callables, locals),
                        )
                    })
                    .collect(),
            },
            *span,
        ),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::ast::MetaExpr {
                expr: Box::new(annotate_effects(
                    &meta.expr,
                    top_level_effects,
                    top_level_callables,
                    locals,
                )),
                entries: meta
                    .entries
                    .iter()
                    .map(|(key, value)| {
                        (
                            key.clone(),
                            annotate_effects(value, top_level_effects, top_level_callables, locals),
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
                                top_level_callables,
                                &local_scope,
                            );
                            let value_effects = infer_expr_effects(
                                &bind_kids[i + 1],
                                top_level_effects,
                                top_level_callables,
                                &local_scope,
                            );
                            if let Some(name) = symbol_name(&bind_kids[i]) {
                                let binding_effects = match &bind_kids[i + 1] {
                                    Expr::List(value_list, _)
                                        if get_tag(value_list) == Some("fn") =>
                                    {
                                        value_effects
                                    }
                                    _ => EffectSet::new(),
                                };
                                local_scope.insert(name.to_string(), binding_effects);
                            }
                            elements.push(value);
                            i += 2;
                        }
                        Expr::List(chelis_deep::ast::List { elements }, *bind_span)
                    } else {
                        annotate_effects(
                            &kids[0],
                            top_level_effects,
                            top_level_callables,
                            &local_scope,
                        )
                    };
                    vec![
                        bind,
                        annotate_effects(
                            &kids[1],
                            top_level_effects,
                            top_level_callables,
                            &local_scope,
                        ),
                    ]
                }
                _ => kids
                    .iter()
                    .map(|kid| {
                        annotate_effects(kid, top_level_effects, top_level_callables, &local_scope)
                    })
                    .collect(),
            };

            let mut elements = list.elements.clone();
            if matches!(elements.get(1), Some(Expr::Map(_, _))) {
                elements.truncate(2);
                elements.extend(annotated_children);
                if let Some(meta) = elements.get_mut(1) {
                    update_effect_metadata(
                        meta,
                        list,
                        top_level_effects,
                        top_level_callables,
                        locals,
                    );
                }
                Expr::List(chelis_deep::ast::List { elements }, *span)
            } else {
                Expr::List(
                    chelis_deep::ast::List {
                        elements: list
                            .elements
                            .iter()
                            .map(|elem| {
                                annotate_effects(
                                    elem,
                                    top_level_effects,
                                    top_level_callables,
                                    locals,
                                )
                            })
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
    top_level_callables: &HashSet<String>,
    locals: &HashMap<String, EffectSet>,
) {
    let Expr::Map(meta, _) = meta_expr else {
        return;
    };
    if let Some("fn") = get_tag(list) {
        let effects = infer_expr_effects(
            &Expr::List(list.clone(), zero_span()),
            top_level_effects,
            top_level_callables,
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
            Effect::Io => symbol("io"),
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
    use chelis_surf::{desugar::desugar_program, parser::parse_str as parse_surf};

    fn checked(src: &str) -> CheckedProgram {
        let exprs = parse_str(src).unwrap();
        let checked = chelis_types::check_phase0e_program(&exprs).unwrap();
        check_program(&checked).unwrap()
    }

    fn surf_checked(src: &str) -> CheckedProgram {
        let decls = parse_surf(src).expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_phase0e_program(&deep).expect("type check");
        check_program(&checked).expect("effect check")
    }

    #[test]
    fn infers_random_for_dropout_fn() {
        let exprs = parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        )
        .unwrap();
        let (inferred, _) = infer_program_effects(&exprs);
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

    #[test]
    fn map_propagates_io_effect_from_callback() {
        let program = surf_checked(
            r#"
def emit(x: int64) -> int64 = debug(add(x, cast(1, int64)))
let xs: List[int64] = [cast(1, int64), cast(2, int64)]
let ys = map(emit, xs)
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("ys")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on map result, got {:?}",
            inferred.get("ys")
        );
    }

    #[test]
    fn fold_propagates_io_effect_from_inline_callback() {
        let program = surf_checked(
            r#"
let xs: List[int64] = [cast(1, int64), cast(2, int64)]
let total = fold(fn (acc: int64, x: int64) -> debug(add(acc, x)), cast(0, int64), xs)
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("total")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on fold result, got {:?}",
            inferred.get("total")
        );
    }

    #[test]
    fn scan_propagates_io_effect_from_inline_callback() {
        let program = surf_checked(
            r#"
let xs: List[int64] = [cast(1, int64), cast(2, int64)]
let totals = scan(fn (acc: int64, x: int64) -> debug(add(acc, x)), cast(0, int64), xs)
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("totals")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on scan result, got {:?}",
            inferred.get("totals")
        );
    }

    #[test]
    fn partition_propagates_random_effect_to_root() {
        let decls = parse_surf(
            r#"
def keep(x: tensor[f32]) -> bool = gt(tensor_to_scalar(dropout(x, 0.5)), 0.0)
let xs: List[tensor[f32]] = [(x1 : tensor[f32]), (x2 : tensor[f32])]
let buckets = partition(keep, xs)
"#,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_phase0e_program(&deep).expect("type check");
        let errors = check_program(&checked).expect_err("partition should propagate Random effect");
        assert!(
            errors
                .iter()
                .any(|error| error.kind == EffectErrorKind::UnhandledEffect
                    && error.message.contains("Random")),
            "expected unhandled Random effect, got {:?}",
            errors
        );
    }

    #[test]
    fn flat_map_propagates_io_effect_from_callback() {
        let program = surf_checked(
            r#"
let xs: List[int64] = [cast(1, int64), cast(2, int64)]
let ys = flat_map(fn (x: int64) -> debug([x, add(x, cast(10, int64))]), xs)
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("ys")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on flat_map result, got {:?}",
            inferred.get("ys")
        );
    }

    #[test]
    fn map_propagates_random_effect_to_root() {
        let decls = parse_surf(
            r#"
def step(x: tensor[8, f32]) -> tensor[8, f32] = dropout(x, 0.5)
let xs: List[tensor[8, f32]] = [(x1 : tensor[8, f32]), (x2 : tensor[8, f32])]
let ys = map(step, xs)
"#,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_phase0e_program(&deep).expect("type check");
        let errors = check_program(&checked).expect_err("map should propagate Random effect");
        assert!(
            errors
                .iter()
                .any(|error| error.kind == EffectErrorKind::UnhandledEffect
                    && error.message.contains("Random")),
            "expected unhandled Random effect, got {:?}",
            errors
        );
    }

    #[test]
    fn file_io_builtins_infer_io_but_mmap_reads_stay_pure() {
        let program = surf_checked(
            r#"
let mapped = mmap_file("dataset.txt")
let prefix = mmap_read(mapped, cast(0, int64), cast(4, int64))
let width = mmap_len(mapped)
let contents = read_file("dataset.txt")
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("mapped")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on mmap_file root, got {:?}",
            inferred.get("mapped")
        );
        assert!(
            inferred
                .get("prefix")
                .is_none_or(|effects| effects.is_empty()),
            "expected mmap_read to stay pure, got {:?}",
            inferred.get("prefix")
        );
        assert!(
            inferred
                .get("width")
                .is_none_or(|effects| effects.is_empty()),
            "expected mmap_len to stay pure, got {:?}",
            inferred.get("width")
        );
        assert!(
            inferred
                .get("contents")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on read_file root, got {:?}",
            inferred.get("contents")
        );
    }
}
