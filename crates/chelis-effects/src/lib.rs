use chelis_deep::DeepTag;
use std::collections::{HashMap, HashSet};

use chelis_deep::ast::{Atom, Expr, List, MetaMap};
use chelis_deep::{Span, decode_effect_kind};
use chelis_types::types::{Effect, EffectSet};
use chelis_types::{CheckedProgram, InferResult};
use chelis_vocab::EffectKind;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EffectErrorKind {
    UnhandledEffect,
    InvalidHandler,
    BuildTargetMismatch,
    TypeTotality,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectError {
    pub kind: EffectErrorKind,
    pub message: String,
    pub suggestions: Vec<String>,
}

/// Compute the inferred effect row for every top-level `def` in a
/// checked program, keyed by def name.
///
/// This runs the same iterative-fixed-point inference that
/// [`check_program`] uses internally, but exposes the per-def effect
/// rows directly instead of folding them into validation. It performs
/// NO validation — callers that need handler-arity / unhandled-random /
/// declared-vs-inferred checks must still call [`check_program`].
///
/// The intended consumer is `chelis check --show-inferred --json`,
/// which joins these effect rows with the type-level signature
/// inference so a machine consumer (Hull) can reconstruct each
/// function's `(Type, EffectRow)` without re-parsing a printer.
pub fn def_effect_rows(program: &CheckedProgram) -> std::collections::BTreeMap<String, EffectSet> {
    let (effects_by_def, _top_level_callables) = infer_program_effects(program.annotated_exprs());
    effects_by_def.into_iter().collect()
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
    validate_declared_vs_inferred(&annotated_exprs, &effects_by_def, &mut errors);

    if errors.is_empty() {
        program
            .try_with_effect_annotations(annotated_exprs)
            .map_err(type_totality_errors)
    } else {
        Err(errors)
    }
}

/// Check effects on `new_program` against an outer-scope library context.
///
/// This is the Phase D context-extension counterpart to [`check_program`].
/// `library_program` must already have been validated by [`check_program`];
/// its effect rows are treated as authoritative and are NOT re-validated.
///
/// The new-code effect rows are inferred against the union of the library's
/// effect map and the new-code's own iterative fixed-point pass. When new
/// code calls a library function, the library's effect row is inherited.
///
/// Validation passes (handler arity, unhandled-random roots, declared-vs-
/// inferred) run ONLY on the new code's annotated expressions. Library
/// validation already happened during the original [`check_program`] call.
///
/// The library's effect map is computed inside this function from
/// `library_program.annotated_exprs()`. It is NOT persisted across calls;
/// the function takes `&CheckedProgram` (not `&mut`) so subsequent calls
/// against the same `library_program` see a fresh effect map untouched by
/// any prior new-code inference.
///
/// ## Returned program
///
/// The returned [`CheckedProgram`] contains ONLY the new-code's annotated
/// expressions, with effect metadata composed against the library. The
/// `type_env()` is preserved verbatim from `new_program`.
pub fn check_effects_with_context(
    library_program: &CheckedProgram,
    new_program: &CheckedProgram,
) -> Result<CheckedProgram, Vec<EffectError>> {
    // Pre-compute library effect map from library bodies. Library was
    // already validated by check_program; we only re-infer to surface
    // effect rows that new code can inherit when it references library
    // callables. We do NOT re-run validation on the library.
    let (library_effects, library_callables) =
        infer_program_effects(library_program.annotated_exprs());

    // Iterative fixed-point inference for new-code defs, seeded with
    // library effects. Library callables stay reachable as call targets;
    // new-code defs of the same name shadow on the inner pass.
    let (effects_by_def, top_level_callables) = infer_program_effects_with_context(
        new_program.annotated_exprs(),
        &library_effects,
        &library_callables,
    );

    // Annotate ONLY new-code exprs, but with effect/callable maps that
    // include library entries so cross-context calls resolve.
    let annotated_exprs: Vec<Expr> = new_program
        .annotated_exprs()
        .iter()
        .map(|expr| annotate_effects(expr, &effects_by_def, &top_level_callables, &HashMap::new()))
        .collect();

    let mut errors = Vec::new();
    validate_handlers(&annotated_exprs, &mut errors);
    validate_unhandled_random_roots(&annotated_exprs, &effects_by_def, &mut errors);
    validate_declared_vs_inferred(&annotated_exprs, &effects_by_def, &mut errors);

    if errors.is_empty() {
        new_program
            .try_with_effect_annotations(annotated_exprs)
            .map_err(type_totality_errors)
    } else {
        Err(errors)
    }
}

fn type_totality_errors(result: InferResult) -> Vec<EffectError> {
    if result.errors.is_empty() {
        return vec![EffectError {
            kind: EffectErrorKind::TypeTotality,
            message: "internal: checked-program reconstruction failed without a type diagnostic"
                .to_string(),
            suggestions: vec![],
        }];
    }
    result
        .errors
        .into_iter()
        .map(|error| EffectError {
            kind: EffectErrorKind::TypeTotality,
            message: error.message,
            suggestions: error.suggestions,
        })
        .collect()
}

/// Validate that the program's target-relevant constructs are admissible
/// for the requested build target.
///
/// Today this validates:
///   * resource-region pinning (`with device("gpu:N")` requires
///     `--target hip` or `--target metal`; CPU pinning requires
///     `--target c`).
///
/// Per spec/04-type-system.md §1.1.3 the Metal target additionally
/// rejects FP64 because Apple Silicon GPUs lack FP64 ALUs (software
/// emulation explicitly out of scope). The spec names three rejection
/// surfaces: the CLI gate (`reject_unsupported_metal_ops` in
/// `chelis-cli`), the IR validation pass
/// (`chelis_ir::verify::validate_metal_admissible_precisions`), and
/// the codegen entry (`Emitter::require_metal_admissible` in
/// `chelis-backend-metal`). The IR pass walks the lowered DAG (which
/// is post-typecheck and out of scope here) and the CLI invokes it
/// between type-check and codegen so the f64-on-metal rejection
/// surfaces with the spec-pinned diagnostic at every entry point.
/// This function does not duplicate that DAG walk; it complements it
/// by validating handler-region pinning at the AST surface.
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
    infer_program_effects_with_context(exprs, &HashMap::new(), &HashSet::new())
}

/// Iterative-fixed-point effect inference that lets new-code defs see an
/// outer library scope. The library's `effects_by_def` and callable-name
/// set are merged into the seed maps so `(var libname)` references in new
/// code resolve to the library's effect row.
///
/// New-code defs shadow library defs of the same name on the inner pass.
/// The function does NOT mutate the library inputs; merging is local.
fn infer_program_effects_with_context(
    new_exprs: &[Expr],
    library_effects: &HashMap<String, EffectSet>,
    library_callables: &HashSet<String>,
) -> (HashMap<String, EffectSet>, HashSet<String>) {
    let bodies = top_level_def_bodies(new_exprs);
    let new_callables = top_level_callable_names(&bodies);

    // Union of library + new callables. New-code names are present
    // because they are added below; both must be visible during inference.
    let mut top_level_callables: HashSet<String> = library_callables.clone();
    top_level_callables.extend(new_callables.iter().cloned());

    // Seed the effects map with the library's already-validated effect
    // rows. New-code defs are *not* in this map yet — the loop below will
    // populate them, shadowing library entries for any name the new code
    // re-defines.
    let mut effects: HashMap<String, EffectSet> = library_effects.clone();

    // Track which names are owned by the new code so we don't accidentally
    // overwrite a library entry that has the same name as a new-code def
    // until the new-code's own inference has produced a value. Library
    // shadow happens naturally: as soon as the new-code def's first pass
    // produces an EffectSet, the .insert() below overwrites the library
    // entry for that name.
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

/// Yield each top-level declaration, descending through any `(module {} name
/// ...)` wrapper. Deep sources produced by Surf `module X` desugaring nest
/// every def/defsig inside this wrapper; the whole-program effect validators
/// below compare declared-vs-inferred and hunt unhandled-Random roots over a
/// flat decl list, so without descent a module-wrapped `.dp` would hide every
/// nested def from them. This mirrors `top_level_decl_items` in chelis-types,
/// so the effect pass sees the same flattened decl set the type pass does and
/// the module-wrapped `chelis check` path agrees with the flattened
/// build/eval path. Descends nested wrappers to any depth.
fn flattened_top_level(exprs: &[Expr]) -> Vec<&Expr> {
    fn push<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some(DeepTag::Module)
        {
            // `(module {} name children...)`: skip tag, meta, name.
            for child in list.elements.iter().skip(3) {
                push(child, out);
            }
            return;
        }
        out.push(expr);
    }
    let mut out = Vec::new();
    for expr in exprs {
        push(expr, &mut out);
    }
    out
}

fn top_level_def_bodies(exprs: &[Expr]) -> HashMap<String, Expr> {
    let mut defs = HashMap::new();
    for expr in flattened_top_level(exprs) {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some(DeepTag::Def)
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
            Expr::List(list, _) if get_tag(list) == Some(DeepTag::Fn) => Some(name.clone()),
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
                DeepTag::Var => children(list)
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
                DeepTag::App => {
                    infer_app_effects(list, top_level_effects, top_level_callables, locals)
                }
                DeepTag::Fn => {
                    let kids = children(list);
                    kids.get(1)
                        .map(|body| {
                            infer_expr_effects(body, top_level_effects, top_level_callables, locals)
                        })
                        .unwrap_or_default()
                }
                DeepTag::Let => {
                    infer_let_effects(list, top_level_effects, top_level_callables, locals)
                }
                DeepTag::If
                | DeepTag::Tuple
                | DeepTag::Pipe
                | DeepTag::Par
                | DeepTag::Record
                | DeepTag::Access
                | DeepTag::TupleGet
                | DeepTag::Cast
                | DeepTag::Copy
                | DeepTag::Realize
                | DeepTag::Jit
                | DeepTag::Match
                | DeepTag::Def => children(list)
                    .iter()
                    .map(|kid| {
                        infer_expr_effects(kid, top_level_effects, top_level_callables, locals)
                    })
                    .fold(EffectSet::new(), |mut acc, set| {
                        acc.extend(&set);
                        acc
                    }),
                DeepTag::Grad => children(list)
                    .first()
                    .map(|kid| {
                        infer_expr_effects(kid, top_level_effects, top_level_callables, locals)
                    })
                    .unwrap_or_default(),
                DeepTag::Vmap => children(list)
                    .first()
                    .map(|kid| {
                        infer_expr_effects(kid, top_level_effects, top_level_callables, locals)
                    })
                    .unwrap_or_default(),
                DeepTag::HandleEffect => {
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
        // Transitional arms for new Expr variants (#908)
        Expr::Node(node, _) => node
            .expr_children()
            .map(|child| infer_expr_effects(child, top_level_effects, top_level_callables, locals))
            .fold(EffectSet::new(), |mut acc, set| {
                acc.extend(&set);
                acc
            }),
        Expr::BareList(elems, _) => elems
            .iter()
            .map(|elem| infer_expr_effects(elem, top_level_effects, top_level_callables, locals))
            .fold(EffectSet::new(), |mut acc, set| {
                acc.extend(&set);
                acc
            }),
        Expr::UnknownForm(data) => data
            .children
            .iter()
            .map(|child| infer_expr_effects(child, top_level_effects, top_level_callables, locals))
            .fold(EffectSet::new(), |mut acc, set| {
                acc.extend(&set);
                acc
            }),
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
    if matches!(builtin_name, Some("dropout" | "uniform_like")) {
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
                | "process_run"
        )
    ) {
        effects.insert(Effect::Io);
    }
    if matches!(
        builtin_name,
        Some(
            "test_assert"
                | "test_assert_eq_f32"
                | "test_assert_eq_int"
                | "test_assert_eq_bool"
                | "test_assert_eq_string"
                | "test_assert_close_tensor"
                | "test_assert_eq_tensor_int64"
        )
    ) {
        effects.insert(Effect::Test);
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
                    Expr::List(value_list, _) if get_tag(value_list) == Some(DeepTag::Fn) => {
                        value_effects
                    }
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
    match decode_effect_kind(list) {
        Ok(EffectKind::Random) => body_effects.remove(&Effect::Random),
        Ok(EffectKind::Resource) => {}
        // Validation reports the structural decode error. Inference leaves the
        // body's effects unhandled instead of substituting a known kind.
        Err(_) => {}
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
                Some(DeepTag::Let) if kids.len() >= 2 => {
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
                                        if get_tag(value_list) == Some(DeepTag::Fn) =>
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
                        Expr::List(chelis_deep::ast::List { tag: bind_list.tag, elements }, *bind_span)
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
                Expr::List(chelis_deep::ast::List { tag: list.tag, elements }, *span)
            } else {
                Expr::List(
                    chelis_deep::ast::List {
                        tag: list.tag,
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
        // Transitional arms for new Expr variants (#908)
        Expr::Node(_, _) | Expr::BareList(_, _) | Expr::UnknownForm(_) => expr.clone(),
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
    if let Some(DeepTag::Fn) = get_tag(list) {
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
            if get_tag(list) == Some(DeepTag::HandleEffect) {
                let kids = children(list);
                match decode_effect_kind(list) {
                    Ok(EffectKind::Random) if kids.first().and_then(int_literal).is_none() => {
                        errors.push(EffectError {
                            kind: EffectErrorKind::InvalidHandler,
                            message: "with seed(...) currently requires an int literal seed"
                                .to_string(),
                            suggestions: vec![
                                "Use `with seed(42i64) { ... }` with an explicit int64-suffixed integer seed"
                                    .to_string(),
                            ],
                        });
                    }
                    Ok(EffectKind::Resource) if kids.first().and_then(string_literal).is_none() => {
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
                    Ok(EffectKind::Random) | Ok(EffectKind::Resource) => {}
                    Err(error) => errors.push(EffectError {
                        kind: EffectErrorKind::InvalidHandler,
                        message: format!("{error} in `handle-effect`"),
                        suggestions: vec![
                            "Use one of the closed effect kinds `random` or `resource`".to_string(),
                        ],
                    }),
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
        // Transitional arms for new Expr variants (#908)
        Expr::Node(node, _) => {
            for child in node.expr_children() {
                validate_handler_expr(child, errors);
            }
        }
        Expr::BareList(elems, _) => {
            for elem in elems {
                validate_handler_expr(elem, errors);
            }
        }
        Expr::UnknownForm(data) => {
            for child in &data.children {
                validate_handler_expr(child, errors);
            }
        }
    }
}

fn validate_unhandled_random_roots(
    exprs: &[Expr],
    effects_by_def: &HashMap<String, EffectSet>,
    errors: &mut Vec<EffectError>,
) {
    for expr in flattened_top_level(exprs) {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some(DeepTag::Def)
        {
            let kids = children(list);
            if kids.len() < 2 {
                continue;
            }
            let Some(name) = symbol_name(&kids[0]) else {
                continue;
            };
            if matches!(&kids[1], Expr::List(body, _) if get_tag(body) == Some(DeepTag::Fn)) {
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
                        "Wrap the stochastic region with `with seed(42i64) { ... }`".to_string(),
                    ],
                });
            }
        }
    }
}

/// Extract the effect set declared on a `(defsig name t-fn-with-eff-meta)` expression.
/// Returns `None` when there is no explicit effect annotation (i.e., inference-only mode).
fn declared_effects_from_defsig(list: &List) -> Option<EffectSet> {
    // defsig has form: (defsig {} name t-fn-expr)
    let kids = children(list);
    let t_fn = kids.get(1)?;
    let Expr::List(t_fn_list, _) = t_fn else {
        return None;
    };
    if get_tag(t_fn_list) != Some(DeepTag::TFn) {
        return None;
    }
    // t-fn metadata is element [1] (the meta map).
    let Expr::Map(meta, _) = t_fn_list.elements.get(1)? else {
        return None;
    };
    let (_, eff_expr) = meta.entries.iter().find(|(key, _)| key == "eff")?;
    // eff_expr is (effects {} sym sym ...)
    let Expr::List(eff_list, _) = eff_expr else {
        return None;
    };
    if get_tag(eff_list) != Some(DeepTag::Effects) {
        return None;
    }
    let mut declared = EffectSet::new();
    for child in children(eff_list) {
        if let Some(name) = symbol_name(child) {
            match name {
                "random" => declared.insert(Effect::Random),
                "accum" => declared.insert(Effect::Accum),
                "io" => declared.insert(Effect::Io),
                "test" => declared.insert(Effect::Test),
                // "diff" is currently tracked separately and does not appear in inferred sets.
                _ => {}
            }
        } else if let Expr::List(inner, _) = child
            && get_tag(inner) == Some(DeepTag::Resource)
            && let Some(device) = children(inner).first().and_then(|expr| match expr {
                Expr::Atom(Atom::Str(value), _) => Some(value.clone()),
                _ => None,
            })
        {
            declared.insert(Effect::Resource(device));
        }
    }
    Some(declared)
}

/// Reject `def f` whose inferred effects exceed the effects declared on its `defsig`.
///
/// This enforces the contract that effect annotations are upper bounds: a function
/// signed `! {}` (or `! { IO }`) cannot silently acquire additional effects such as
/// `Test` via a call to `test_assert_*`.
fn validate_declared_vs_inferred(
    exprs: &[Expr],
    effects_by_def: &HashMap<String, EffectSet>,
    errors: &mut Vec<EffectError>,
) {
    let mut declared_by_name: HashMap<String, EffectSet> = HashMap::new();
    for expr in flattened_top_level(exprs) {
        if let Expr::List(list, _) = expr
            && get_tag(list) == Some(DeepTag::Defsig)
        {
            let kids = children(list);
            let Some(name) = kids.first().and_then(symbol_name) else {
                continue;
            };
            if let Some(declared) = declared_effects_from_defsig(list) {
                declared_by_name.insert(name.to_string(), declared);
            }
        }
    }

    for (name, declared) in &declared_by_name {
        let inferred = match effects_by_def.get(name) {
            Some(inferred) => inferred,
            None => continue,
        };
        // Find effects inferred but not declared.
        let missing: Vec<Effect> = inferred
            .iter()
            .filter(|effect| !declared.contains(effect))
            .cloned()
            .collect();
        if missing.is_empty() {
            continue;
        }
        let declared_str = if declared.is_empty() {
            "{}".to_string()
        } else {
            declared.to_string()
        };
        let missing_str = missing
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        errors.push(EffectError {
            kind: EffectErrorKind::UnhandledEffect,
            message: format!(
                "Function `{name}` is declared with effects `{declared_str}` but its body performs effects `{{{missing_str}}}` that were not declared"
            ),
            suggestions: vec![format!(
                "Either add the missing effect(s) to the signature of `{name}` (e.g. `! {{ {missing_str} }}`) or refactor the body so it does not perform them."
            )],
        });
    }
}

fn validate_build_target_expr(expr: &Expr, target: &str, errors: &mut Vec<EffectError>) {
    match expr {
        Expr::List(list, _) => {
            if get_tag(list) == Some(DeepTag::HandleEffect) {
                match decode_effect_kind(list) {
                    Ok(EffectKind::Random) => {}
                    Ok(EffectKind::Resource) => {
                        if let Some(device) = children(list).first().and_then(string_literal) {
                            let ok = match target {
                                "c" => !device.starts_with("gpu"),
                                "hip" | "metal" => device.starts_with("gpu"),
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
                                            "Use `with device(\"cpu\") { ... }` or build with `--target hip` or `--target metal`"
                                                .to_string(),
                                        ],
                                        "hip" | "metal" => vec![
                                            "Use a GPU device such as `with device(\"gpu:0\") { ... }`"
                                                .to_string(),
                                        ],
                                        _ => vec![],
                                    },
                                });
                            }
                        }
                    }
                    // `validate_handlers` is the owning structural diagnostic
                    // pass. Do not assign target semantics after decode fails.
                    Err(_) => {}
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
        // Transitional arms for new Expr variants (#908)
        Expr::Node(node, _) => {
            for child in node.expr_children() {
                validate_build_target_expr(child, target, errors);
            }
        }
        Expr::BareList(elems, _) => {
            for elem in elems {
                validate_build_target_expr(elem, target, errors);
            }
        }
        Expr::UnknownForm(data) => {
            for child in &data.children {
                validate_build_target_expr(child, target, errors);
            }
        }
    }
}

/// Decode-once (chelis#731 Phase 3): both heads here are vocabulary tags
/// (`DeepTag::Effects`, `DeepTag::Resource`), and this runs AFTER the parser
/// and the desugarer, so nothing upstream will stamp them. Spelling them
/// `symbol("effects")` / `symbol("resource")` put raw vocabulary strings back
/// into the tree, and the typed readers that gate on
/// `tag(list) == Some(DeepTag::Effects)` reject that form outright
/// (`declared_effects_from_meta` above; `decompile_effect_suffix_from_type_expr`
/// and `decompile_effect_set_expr` in chelis-surf).
///
/// `chelis_surf::desugar::desugar_effect_set` builds the SAME node shape and
/// was migrated to the typed constructors; this is its post-check twin and
/// now matches it exactly. The effect NAMES stay `Atom::Name` deliberately:
/// `random`, `accum`, `io` and `test` are payload, not vocabulary tags.
fn effect_set_expr(effects: &EffectSet) -> Expr {
    let mut children = Vec::new();
    for effect in effects.iter() {
        children.push(match effect {
            Effect::Random => symbol("random"),
            Effect::Accum => symbol("accum"),
            Effect::Io => symbol("io"),
            Effect::Test => symbol("test"),
            Effect::Resource(device) => Expr::node(
                DeepTag::Resource,
                MetaMap::default(),
                vec![Expr::Atom(Atom::Str(device.clone()), zero_span())],
                zero_span(),
            ),
        });
    }
    Expr::node(DeepTag::Effects, MetaMap::default(), children, zero_span())
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
    if get_tag(list) != Some(DeepTag::Var) {
        return None;
    }
    children(list).first().and_then(symbol_name)
}

fn get_tag(list: &List) -> Option<DeepTag> {
    list.tag()
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
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn int_literal(expr: &Expr) -> Option<i64> {
    match expr {
        Expr::Atom(Atom::Int(value), _) => Some(*value),
        Expr::List(list, _) if get_tag(list) == Some(DeepTag::Lit) => {
            match children(list).first() {
                Some(Expr::Atom(Atom::Int(value), _)) => Some(*value),
                _ => None,
            }
        }
        _ => None,
    }
}

fn string_literal(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Str(value), _) => Some(value.as_str()),
        Expr::List(list, _) if get_tag(list) == Some(DeepTag::Lit) => {
            match children(list).first() {
                Some(Expr::Atom(Atom::Str(value), _)) => Some(value.as_str()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn symbol(name: &str) -> Expr {
    Expr::Atom(Atom::Name(name.to_string()), zero_span())
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
        let checked = chelis_types::check_ir_program(&exprs).unwrap();
        check_program(&checked).unwrap()
    }

    fn surf_checked(src: &str) -> CheckedProgram {
        let decls = parse_surf(src).expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        check_program(&checked).expect("effect check")
    }

    #[test]
    fn effect_annotation_reconstruction_preserves_type_context() {
        let decls = parse_surf("def add_one(x: int32) -> int32 = add(x, 1)").expect("surf parse");
        let deep = desugar_program(&decls);
        let typed = chelis_types::check_ir_program(&deep).expect("type check");
        let expected_type_env = typed.type_env().clone();
        let expected_signatures = typed.signature_inference().clone();

        let reconstructed = check_program(&typed).expect("effect check");
        assert_eq!(reconstructed.type_env(), &expected_type_env);
        assert_eq!(reconstructed.signature_inference(), &expected_signatures);
    }

    #[test]
    fn literal_random_and_resource_handlers_cross_the_type_effect_boundary() {
        for source in [
            r#"(def {} value
                   (handle-effect {effect: random}
                     (lit {type: (t-prim {} int64)} 7)
                     (lit {type: (t-prim {} int32)} 1)))"#,
            r#"(def {} value
                   (handle-effect {effect: resource}
                     (lit {type: (t-prim {} string)} "cpu")
                     (lit {type: (t-prim {} int32)} 1)))"#,
        ] {
            let deep = parse_str(source).expect("Deep handler fixture parses");
            let typed = chelis_types::check_ir_program(&deep).expect("type boundary accepts");
            check_program(&typed).expect("effects boundary accepts literal handler");
        }
    }

    #[test]
    fn nonliteral_handlers_are_rejected_once_by_the_effect_owner() {
        for (effect, expected) in [
            ("random", "requires an int literal seed"),
            ("resource", "requires a string literal device"),
        ] {
            let source = format!(
                "(def {{}} value (handle-effect {{effect: {effect}}} \
                 (var {{}} computed_handler) (lit {{type: (t-prim {{}} int32)}} 1)))"
            );
            let deep = parse_str(&source).expect("Deep handler fixture parses");
            let typed = chelis_types::check_ir_program(&deep)
                .expect("handler payload is owned by the effects gate");
            let errors = check_program(&typed).expect_err("nonliteral handler must reject");
            assert_eq!(errors.len(), 1, "one effects owner diagnostic: {errors:?}");
            assert_eq!(errors[0].kind, EffectErrorKind::InvalidHandler);
            assert!(errors[0].message.contains(expected), "{errors:?}");
        }
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

    // `def_effect_rows` is the public seam `chelis check --show-inferred
    // --json` reads to attach a structured effect row to each function.
    // Positive: a function calling `debug` carries IO. Negative parity:
    // a pure function carries an EMPTY row (present and empty, not
    // missing), so a consumer can tell "pure" from "unknown".
    #[test]
    fn def_effect_rows_reports_io_and_empty_rows() {
        let program = surf_checked(
            r#"
def logged(msg: string) -> string = debug(msg)
def pure_add(x: int64, y: int64) -> int64 = add(x, y)
"#,
        );
        let rows = def_effect_rows(&program);
        let logged = rows.get("logged").expect("logged effect row present");
        assert!(logged.contains(&Effect::Io));
        assert_eq!(logged.iter().count(), 1);

        let pure_add = rows.get("pure_add").expect("pure_add effect row present");
        assert!(
            pure_add.is_empty(),
            "pure function must have an empty effect row, got {pure_add}"
        );
    }

    #[test]
    fn rejects_unhandled_dropout_root() {
        let exprs = parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} dropout) (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        )
        .unwrap();
        let checked = chelis_types::check_ir_program(&exprs).unwrap();
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
xs: List[int64] = [cast(1, int64), cast(2, int64)]
ys = map(emit, xs)
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
xs: List[int64] = [cast(1, int64), cast(2, int64)]
total = fold(fn (acc: int64, x: int64) -> debug(add(acc, x)), cast(0, int64), xs)
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
xs: List[int64] = [cast(1, int64), cast(2, int64)]
totals = scan(fn (acc: int64, x: int64) -> debug(add(acc, x)), cast(0, int64), xs)
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
xs: List[tensor[f32]] = [
  trace(pad_sequences_to([[1.0]], cast(1, int64), cast(0.0, f32)), 0, 1),
  trace(pad_sequences_to([[2.0]], cast(1, int64), cast(0.0, f32)), 0, 1)
]
buckets = partition(keep, xs)
"#,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
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
xs: List[int64] = [cast(1, int64), cast(2, int64)]
ys = flat_map(fn (x: int64) -> debug([x, add(x, cast(10, int64))]), xs)
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
xs: List[tensor[8, f32]] = [
  (to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]) : tensor[8, f32]),
  (to_tensor([8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]) : tensor[8, f32])
]
ys = map(step, xs)
"#,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
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
mapped = mmap_file("dataset.txt")
prefix = mmap_read(mapped, cast(0, int64), cast(4, int64))
width = mmap_len(mapped)
contents = read_file("dataset.txt")
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

    #[test]
    fn process_run_infers_io_but_pure_binding_stays_pure() {
        // Hull subprocess exec: a binding whose value applies process_run
        // acquires Effect::Io (mirroring read_file), while an adjacent pure
        // arithmetic binding stays effect-free.
        let program = surf_checked(
            r#"
result = process_run("echo", ["hi"])
pure_value = add(cast(1, int64), cast(2, int64))
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("result")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on process_run root, got {:?}",
            inferred.get("result")
        );
        assert!(
            inferred
                .get("pure_value")
                .is_none_or(|effects| effects.is_empty()),
            "expected pure arithmetic binding to stay effect-free, got {:?}",
            inferred.get("pure_value")
        );
    }

    // ----- Phase 3t.1: Test effect propagation and enforcement -----

    #[test]
    fn test_assert_adds_test_effect_to_calling_fn() {
        // A top-level def whose body calls test_assert should acquire Effect::Test.
        let program = surf_checked(
            r#"
def my_check() -> unit = test_assert(true, "ok")
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("my_check")
                .is_some_and(|effects| effects.contains(&Effect::Test)),
            "expected Test effect on my_check, got {:?}",
            inferred.get("my_check")
        );
    }

    #[test]
    fn test_assert_close_tensor_propagates_test_effect() {
        let program = surf_checked(
            r#"
def close() -> unit =
  test_assert_close_tensor(
    to_tensor([1.0, 2.0]),
    to_tensor([1.0, 2.0]),
    0.01,
    "close"
  )
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("close")
                .is_some_and(|effects| effects.contains(&Effect::Test)),
            "expected Test effect on close(), got {:?}",
            inferred.get("close")
        );
    }

    #[test]
    fn test_effect_propagates_transitively_through_calls() {
        let program = surf_checked(
            r#"
def inner() -> unit = test_assert(true, "inner")
def outer() -> unit = inner()
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("outer")
                .is_some_and(|effects| effects.contains(&Effect::Test)),
            "expected Test effect to transit inner -> outer, got {:?}",
            inferred.get("outer")
        );
    }

    #[test]
    fn with_seed_does_not_handle_test_effect() {
        // `with seed(...)` must remove Random, but must NOT remove Test.
        let program = surf_checked(
            r#"
def sealed() -> unit =
  with seed(7i64) { test_assert(true, "inside-handler") }
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("sealed")
                .is_some_and(|effects| effects.contains(&Effect::Test)),
            "with seed(...) must not swallow the Test effect, got {:?}",
            inferred.get("sealed")
        );
    }

    #[test]
    fn empty_effect_sig_rejects_test_assert_caller() {
        // `def g() -> unit ! {} = test_assert(...)` should fail: body inferred Test,
        // sig declares no effects.
        let decls = parse_surf(
            r#"
def g() -> unit ! {} = test_assert(true, "leak")
"#,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        let errors = check_program(&checked)
            .expect_err("def with `! {}` that calls test_assert must be rejected");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect && error.message.contains("Test")
            }),
            "expected UnhandledEffect mentioning Test, got {errors:?}"
        );
    }

    #[test]
    fn empty_effect_sig_rejects_transitive_test_caller() {
        // Caller has no assert in its body, but it calls a function that does.
        // The `! {}` signature on g must still reject this.
        let decls = parse_surf(
            r#"
def f() -> unit = test_assert(true, "x")
def g() -> unit ! {} = f()
"#,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        let errors =
            check_program(&checked).expect_err("transitive caller with `! {}` must be rejected");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect
                    && error.message.contains("g")
                    && error.message.contains("Test")
            }),
            "expected UnhandledEffect on g with Test, got {errors:?}"
        );
    }

    #[test]
    fn declared_test_effect_accepts_test_assert_caller() {
        // When the caller honestly declares `! { Test }`, no error.
        let decls = parse_surf(
            r#"
def test_ok() -> unit ! {Test} = test_assert(true, "ok")
"#,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        check_program(&checked).expect("declared Test should accept test_assert caller");
    }

    #[test]
    fn declared_io_effect_still_rejects_test_effect_leakage() {
        // A sig with `! { IO }` is not a superset of `{ Test }`. Adding a test_assert
        // call inside an IO-only function should still be rejected — the effect
        // systems are orthogonal axes of the effect row.
        let decls = parse_surf(
            r#"
def leak() -> unit ! {IO} = test_assert(true, "sneak")
"#,
        )
        .expect("surf parse");
        let deep = desugar_program(&decls);
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        let errors =
            check_program(&checked).expect_err("IO-declared fn must not silently acquire Test");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect && error.message.contains("Test")
            }),
            "expected UnhandledEffect mentioning Test, got {errors:?}"
        );
    }

    // ----- Module-wrapped whole-program effect soundness -----
    //
    // A `.dp` MODULE wraps its decls in `(module ...)`. The whole-program
    // effect validators must descend into that wrapper so a module-wrapped
    // `chelis check` agrees with the flattened build/eval path. Without the
    // descent, a declared-pure function whose body performs Random/IO would be
    // accepted module-wrapped but rejected flattened.

    /// Type-check a module-WRAPPED Deep program (the shape `chelis check` runs
    /// on a `.dp` MODULE), preserving the `(module ...)` wrapper.
    fn typed_module(src: &str) -> CheckedProgram {
        let decls = parse_surf(src).expect("surf parse");
        let deep = desugar_program(&decls);
        chelis_types::check_typed_program(&deep).expect("type check")
    }

    #[test]
    fn module_wrapped_declared_pure_body_does_random_is_rejected() {
        // `entry` is declared pure (`! { }`) but its body calls `noisy`, which
        // performs Random. Wrapped in `(module ...)`, the declared-vs-inferred
        // validator must still fire after descending into the wrapper.
        let checked = typed_module(
            r#"module Frag.Effect
export (entry)
def noisy(x: tensor[8, f32]) -> tensor[8, f32] = dropout(x, 0.5)
def entry(x: tensor[8, f32]) -> tensor[8, f32] ! { } = noisy(x)
"#,
        );
        let errors = check_program(&checked)
            .expect_err("module-wrapped declared-pure body performing Random must be rejected");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect
                    && error.message.contains("entry")
                    && error.message.contains("Random")
            }),
            "expected UnhandledEffect on entry mentioning Random, got {errors:?}"
        );
    }

    #[test]
    fn module_wrapped_declared_pure_body_does_io_is_rejected() {
        // The IO counterpart: a declared-pure function whose body calls a
        // file-IO builtin must be rejected module-wrapped, the same as Random.
        let checked = typed_module(
            r#"module Frag.Io
export (entry)
def entry(path: string) -> string ! { } = read_file(path)
"#,
        );
        let errors = check_program(&checked)
            .expect_err("module-wrapped declared-pure body performing IO must be rejected");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect
                    && error.message.contains("entry")
                    && error.message.contains("IO")
            }),
            "expected UnhandledEffect on entry mentioning IO, got {errors:?}"
        );
    }

    #[test]
    fn module_wrapped_unhandled_random_value_root_is_rejected() {
        // A value-binding (non-fn) root that performs Random inside a module
        // wrapper must still be caught by the unhandled-random-roots validator.
        let checked = typed_module(
            r#"module Frag.Root
export (sampled)
sampled: tensor[8, f32] = dropout(to_tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]), 0.5)
"#,
        );
        let errors = check_program(&checked)
            .expect_err("module-wrapped unhandled Random value root must be rejected");
        assert!(
            errors.iter().any(|error| {
                error.kind == EffectErrorKind::UnhandledEffect && error.message.contains("Random")
            }),
            "expected unhandled Random effect, got {errors:?}"
        );
    }

    #[test]
    fn module_wrapped_pure_program_checks_clean() {
        // The negative-parity case: a pure module must still check clean after
        // the descent change (no effects -> no rejection).
        let checked = typed_module(
            r#"module Frag.Pure
export (entry)
def helper(x: f32) -> f32 = add(x, x)
def entry(x: f32) -> f32 = helper(mul(x, x))
"#,
        );
        check_program(&checked).expect("pure module-wrapped program must check clean");
    }

    #[test]
    fn module_wrapped_honest_random_signature_checks_clean() {
        // A module-wrapped function that honestly declares `! { Random }` and
        // handles the effect with `with seed(...)` must check clean: the
        // descent fix tightens the unsound-accept path only, not honest code.
        let checked = typed_module(
            r#"module Frag.Honest
export (entry)
def entry(x: tensor[8, f32]) -> tensor[8, f32] =
  with seed(7i64) { dropout(x, 0.5) }
"#,
        );
        check_program(&checked).expect("handled-Random module-wrapped program must check clean");
    }
}

#[cfg(test)]
mod decode_once_producer_tests {
    use super::*;

    /// chelis#731 Phase 3, found by the #887 consumption-boundary probe:
    /// `effect_set_expr` runs after the parser and the desugarer, so its
    /// heads are never stamped upstream. It spelled them `symbol("effects")`
    /// / `symbol("resource")`, which put raw vocabulary strings back into the
    /// tree - the same class as `ty_expr_to_deep` (chelis-ir), and invisible
    /// to the standing invariant because that is asserted on parsed and
    /// desugared trees, not on post-check synthesis.
    ///
    /// Both polarities: no raw vocabulary tag anywhere in the produced tree,
    /// AND the typed readers actually see the decoded tags.
    #[test]
    fn effect_set_expr_carries_no_raw_vocabulary_tag_strings() {
        let mut effects = EffectSet::new();
        effects.insert(Effect::Random);
        effects.insert(Effect::Io);
        effects.insert(Effect::Resource("gpu0".to_string()));

        let expr = effect_set_expr(&effects);
        assert_eq!(
            chelis_deep::validate::find_raw_vocabulary_tag(std::slice::from_ref(&expr)),
            None,
            "a synthesized effect-set node must not carry a raw vocabulary tag string"
        );
        assert_eq!(
            expr.tag(),
            Some(DeepTag::Effects),
            "the typed readers gate on `tag() == Some(DeepTag::Effects)`"
        );

        let Expr::List(list, _) = &expr else {
            panic!("effect_set_expr produces a list");
        };
        let resource = children(list)
            .iter()
            .find(|child| child.tag() == Some(DeepTag::Resource))
            .expect("the nested resource node must be stamped too");
        assert_eq!(resource.tag(), Some(DeepTag::Resource));
    }

    /// Negative parity: the effect NAMES are payload, not vocabulary, and
    /// must stay bare symbols. If they were ever stamped the readers below
    /// (`symbol_name`) would stop resolving them.
    #[test]
    fn effect_names_stay_bare_symbols() {
        let mut effects = EffectSet::new();
        effects.insert(Effect::Random);
        let expr = effect_set_expr(&effects);
        let Expr::List(list, _) = &expr else {
            panic!("effect_set_expr produces a list");
        };
        assert_eq!(
            children(list).first().and_then(symbol_name),
            Some("random"),
            "effect names are payload and must remain readable as bare symbols"
        );
    }
}
