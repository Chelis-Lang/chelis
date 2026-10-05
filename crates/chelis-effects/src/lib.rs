use chelis_deep::DeepTag;
use std::collections::{BTreeMap, BTreeSet};

#[cfg(test)]
use std::cell::RefCell;

use chelis_deep::annotations::{
    EffectMember, EffectSet as AstEffectSet, MetadataValue, ResourceEffect, Spanned,
};
use chelis_deep::ast::{Atom, Expr, Metadata};
use chelis_deep::{ExprCarrier, Span, decode_effect_kind};
use chelis_types::types::{Effect, EffectSet};
use chelis_types::{CheckedProgram, InferResult};
use chelis_vocab::EffectKind;
use chelis_vocab::EffectKindDecodeError;

pub mod realizability;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct EffectWorkProfile {
    infer_expr_visits: usize,
    annotation_expr_visits: usize,
    node_bridge_clone_nodes: usize,
    list_rewrite_clone_nodes: usize,
    top_level_body_clone_nodes: usize,
    metadata_rewrites: usize,
}

#[cfg(test)]
thread_local! {
    static EFFECT_WORK_PROFILE: RefCell<EffectWorkProfile> =
        RefCell::new(EffectWorkProfile::default());
}

#[cfg(test)]
fn record_effect_work(update: impl FnOnce(&mut EffectWorkProfile)) {
    EFFECT_WORK_PROFILE.with(|profile| update(&mut profile.borrow_mut()));
}

#[cfg(not(test))]
#[inline(always)]
fn record_effect_work(_update: impl FnOnce(&mut EffectWorkProfile)) {}

#[cfg(test)]
fn reset_effect_work_profile() {
    EFFECT_WORK_PROFILE.with(|profile| *profile.borrow_mut() = EffectWorkProfile::default());
}

#[cfg(test)]
fn take_effect_work_profile() -> EffectWorkProfile {
    EFFECT_WORK_PROFILE.with(|profile| std::mem::take(&mut *profile.borrow_mut()))
}

/// An effect diagnostic's kind.
///
/// Its published spelling is the governed `chelis_vocab::DiagnosticKind`
/// identity that `Diagnostic::from_effect_error` maps it to (chelis#886).
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

/// The inferred effect row of every top-level `def`, keyed by def name.
pub type DefEffectRows = BTreeMap<String, EffectSet>;

/// The one inferred-effect result for `program`, optionally extended with
/// an outer library scope.
///
/// Enforcement ([`check_program`], [`check_effects_with_context`]) and
/// reporting ([`def_effect_rows`], [`def_effect_rows_in_context`]) both
/// route through here, so a reported row is the row the discharge check
/// used. They differ only in what they do afterwards: enforcement
/// validates and annotates, reporting returns the rows.
///
/// `library` is the already-checked outer program a caller's imports
/// resolve against. Passing `None` infers over `program` alone, which is
/// correct only when `program` is the whole program: a def whose callee
/// lives in an omitted library would otherwise be inferred pure
/// (chelis#606).
fn infer_def_effects(
    library: Option<&CheckedProgram>,
    program: &CheckedProgram,
) -> (DefEffectRows, BTreeSet<String>) {
    let Some(library) = library else {
        return infer_program_effects(program.annotated_exprs());
    };
    // The library was already validated by its own check; re-inferring
    // here only surfaces the rows new code inherits when it calls in.
    let (library_effects, library_callables) = infer_program_effects(library.annotated_exprs());
    infer_program_effects_with_context(
        program.annotated_exprs(),
        &library_effects,
        &library_callables,
    )
}

/// Compute the inferred effect row for every top-level `def` in a
/// checked program, keyed by def name.
///
/// This runs the same iterative-fixed-point inference that
/// [`check_program`] uses internally, but exposes the per-def effect
/// rows directly instead of folding them into validation. It performs
/// NO validation — callers that need handler-arity /
/// declared-vs-inferred checks must still call [`check_program`].
///
/// The intended consumer is `chelis check --show-inferred --json`,
/// which joins these effect rows with the type-level signature
/// inference so a machine consumer (Hull) can reconstruct each
/// function's `(Type, EffectRow)` without re-parsing a printer.
///
/// `program` must be the whole program. A caller holding only a package
/// extension must use [`def_effect_rows_in_context`] and pass the
/// library the extension was checked against; see chelis#606.
pub fn def_effect_rows(program: &CheckedProgram) -> DefEffectRows {
    def_effect_rows_in_context(None, program)
}

/// The reporting counterpart to [`check_effects_with_context`]: the
/// inferred effect rows of `program`, resolved against `library`.
///
/// An imported callee's effects propagate to its caller exactly as a
/// same-package callee's do (spec/04-type-system.md §7.1: a function's
/// inferred effect set is the union of the effects of the operations in
/// its body, and stdlib helpers inherit effects through calls). Nothing
/// in that rule is package-relative, so the row a package boundary
/// reports must equal the row enforcement discharges.
pub fn def_effect_rows_in_context(
    library: Option<&CheckedProgram>,
    program: &CheckedProgram,
) -> DefEffectRows {
    let mut rows = infer_def_effects(library, program).0;
    let owned_definitions = top_level_def_bodies(program.annotated_exprs());
    rows.retain(|name, _| owned_definitions.contains_key(name));
    rows
}

pub fn check_program(program: &CheckedProgram) -> Result<CheckedProgram, Vec<EffectError>> {
    let (effects_by_def, top_level_callables) = infer_def_effects(None, program);
    let annotated_exprs: Vec<Expr> = program
        .annotated_exprs()
        .iter()
        .map(|expr| {
            annotate_effects(
                expr,
                &effects_by_def,
                &top_level_callables,
                &BTreeMap::new(),
            )
        })
        .collect();

    let mut errors = Vec::new();
    validate_handlers(&annotated_exprs, &mut errors);
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
/// Validation passes (handler arity, declared-vs-inferred) run ONLY on the new code's annotated expressions. Library
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
    // Iterative fixed-point inference for new-code defs, seeded with the
    // library's effect rows. Library callables stay reachable as call
    // targets; new-code defs of the same name shadow on the inner pass.
    // This is the same `infer_def_effects` result
    // `def_effect_rows_in_context` reports, so enforcement and reporting
    // cannot disagree (chelis#606).
    let (effects_by_def, top_level_callables) =
        infer_def_effects(Some(library_program), new_program);

    // Annotate ONLY new-code exprs, but with effect/callable maps that
    // include library entries so cross-context calls resolve.
    let annotated_exprs: Vec<Expr> = new_program
        .annotated_exprs()
        .iter()
        .map(|expr| {
            annotate_effects(
                expr,
                &effects_by_def,
                &top_level_callables,
                &BTreeMap::new(),
            )
        })
        .collect();

    let mut errors = Vec::new();
    validate_handlers(&annotated_exprs, &mut errors);
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
///   * resource-region pinning (host C admits only exact `cpu`; every other
///     selector requires a target capability that defines its semantics).
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
    validate_build_target_expressions(program.annotated_exprs(), target)
}

/// Check Resource regions in an emission scope selected from checked source.
///
/// This shares the whole-program validator's target classification. Callers
/// selecting an entry must include its transitive source dependencies before
/// lowering erases the handlers. This validation does not establish that an
/// arbitrary expression slice is well typed or is the correct emission scope.
pub fn validate_build_target_expressions(
    expressions: &[Expr],
    target: &str,
) -> Result<(), Vec<EffectError>> {
    let mut errors = Vec::new();
    for expr in expressions {
        validate_build_target_expr(expr, target, &mut errors);
    }
    if errors.is_empty() {
        Ok(())
    } else {
        Err(errors)
    }
}

fn infer_program_effects(exprs: &[Expr]) -> (BTreeMap<String, EffectSet>, BTreeSet<String>) {
    infer_program_effects_with_context(exprs, &BTreeMap::new(), &BTreeSet::new())
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
    library_effects: &BTreeMap<String, EffectSet>,
    library_callables: &BTreeSet<String>,
) -> (BTreeMap<String, EffectSet>, BTreeSet<String>) {
    let bodies = top_level_def_bodies(new_exprs);
    let new_callables = top_level_callable_names(&bodies);

    // Union of library + new callables. New-code names are present
    // because they are added below; both must be visible during inference.
    let mut top_level_callables: BTreeSet<String> = library_callables.clone();
    top_level_callables.extend(new_callables.iter().cloned());

    // Seed the effects map with the library's already-validated effect
    // rows. New-code defs are *not* in this map yet — the loop below will
    // populate them, shadowing library entries for any name the new code
    // re-defines.
    let mut effects: BTreeMap<String, EffectSet> = library_effects.clone();

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
                infer_expr_effects(body, &effects, &top_level_callables, &BTreeMap::new());
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
/// below compare declared-vs-inferred over a flat decl list, so without descent a module-wrapped `.dp` would hide every
/// nested def from them. This mirrors `top_level_decl_items` in chelis-types,
/// so the effect pass sees the same flattened decl set the type pass does and
/// the module-wrapped `chelis check` path agrees with the flattened
/// build/eval path. Descends nested wrappers to any depth.
fn flattened_top_level(exprs: &[Expr]) -> Vec<&Expr> {
    fn push<'a>(expr: &'a Expr, out: &mut Vec<&'a Expr>) {
        match expr.carrier() {
            ExprCarrier::DecodedNode(DeepTag::Module, _, children) => {
                // Module semantic children are `[name, declarations...]`.
                for child in children.iter().skip(1) {
                    push(child, out);
                }
            }
            ExprCarrier::DecodedNode(_, _, _)
            | ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => out.push(expr),
        }
    }
    let mut out = Vec::new();
    for expr in exprs {
        push(expr, &mut out);
    }
    out
}

fn top_level_def_bodies(exprs: &[Expr]) -> BTreeMap<String, &Expr> {
    let mut defs = BTreeMap::new();
    for expr in flattened_top_level(exprs) {
        match expr.carrier() {
            ExprCarrier::DecodedNode(DeepTag::Def, _, children) if children.len() >= 2 => {
                if let Some(name) = symbol_name(&children[0]) {
                    defs.insert(name.to_string(), &children[1]);
                }
            }
            ExprCarrier::DecodedNode(_, _, _)
            | ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => {}
        }
    }
    defs
}

/// A top-level function, or an alias whose body names another value: an
/// alias carries the latent effects of what it names, a builtin included
/// ([04-INF-9]), and an alias of plain data names nothing latent.
fn top_level_callable_names(bodies: &BTreeMap<String, &Expr>) -> BTreeSet<String> {
    bodies
        .iter()
        .filter(|(_, body)| matches!(body.tag(), Some(DeepTag::Fn | DeepTag::Var)))
        .map(|(name, _)| name.clone())
        .collect()
}

fn infer_expr_effects(
    expr: &Expr,
    top_level_effects: &BTreeMap<String, EffectSet>,
    top_level_callables: &BTreeSet<String>,
    locals: &BTreeMap<String, EffectSet>,
) -> EffectSet {
    record_effect_work(|profile| profile.infer_expr_visits += 1);
    match expr.carrier() {
        ExprCarrier::Atom(_) | ExprCarrier::MetadataMap(_) => EffectSet::new(),
        ExprCarrier::MetadataExpression(meta) => {
            infer_expr_effects(&meta.expr, top_level_effects, top_level_callables, locals)
        }
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            let handled_effect = (tag == DeepTag::HandleEffect)
                .then(|| decode_effect_kind(metadata).ok())
                .flatten();
            infer_tagged_effects(
                tag,
                children,
                handled_effect,
                top_level_effects,
                top_level_callables,
                locals,
            )
        }
        ExprCarrier::StructuralList(children) | ExprCarrier::UndecodableHead(_, _, children) => {
            infer_children_effects(children, top_level_effects, top_level_callables, locals)
        }
    }
}

fn infer_children_effects(
    children: &[Expr],
    top_level_effects: &BTreeMap<String, EffectSet>,
    top_level_callables: &BTreeSet<String>,
    locals: &BTreeMap<String, EffectSet>,
) -> EffectSet {
    children
        .iter()
        .map(|child| infer_expr_effects(child, top_level_effects, top_level_callables, locals))
        .fold(EffectSet::new(), |mut accumulated, effects| {
            accumulated.extend(&effects);
            accumulated
        })
}

fn infer_tagged_effects(
    tag: DeepTag,
    kids: &[Expr],
    handled_effect: Option<EffectKind>,
    top_level_effects: &BTreeMap<String, EffectSet>,
    top_level_callables: &BTreeSet<String>,
    locals: &BTreeMap<String, EffectSet>,
) -> EffectSet {
    match tag {
        DeepTag::Var => kids
            .first()
            .and_then(symbol_name)
            .and_then(|name| {
                locals.get(name).cloned().or_else(|| {
                    top_level_callables
                        .contains(name)
                        .then(|| top_level_effects.get(name).cloned())
                        .flatten()
                        // [04-INF-9]: a builtin named as a value performs
                        // its call's effect, as a lambda calling it would.
                        .or_else(|| {
                            chelis_types::builtin_call_effect(name).map(|effect| {
                                let mut effects = EffectSet::new();
                                effects.insert(effect);
                                effects
                            })
                        })
                })
            })
            .unwrap_or_default(),
        DeepTag::App => infer_app_effects(kids, top_level_effects, top_level_callables, locals),
        DeepTag::Fn => kids
            .get(1)
            .map(|body| infer_expr_effects(body, top_level_effects, top_level_callables, locals))
            .unwrap_or_default(),
        DeepTag::Let => infer_let_effects(kids, top_level_effects, top_level_callables, locals),
        DeepTag::Grad | DeepTag::Vmap => kids
            .first()
            .map(|child| infer_expr_effects(child, top_level_effects, top_level_callables, locals))
            .unwrap_or_default(),
        DeepTag::HandleEffect => infer_handle_effects(
            kids,
            handled_effect,
            top_level_effects,
            top_level_callables,
            locals,
        ),
        _ => infer_children_effects(kids, top_level_effects, top_level_callables, locals),
    }
}

fn infer_app_effects(
    kids: &[Expr],
    top_level_effects: &BTreeMap<String, EffectSet>,
    top_level_callables: &BTreeSet<String>,
    locals: &BTreeMap<String, EffectSet>,
) -> EffectSet {
    let mut effects = EffectSet::new();
    for kid in kids {
        effects.extend(&infer_expr_effects(
            kid,
            top_level_effects,
            top_level_callables,
            locals,
        ));
    }

    if let Some(effect) = kids
        .first()
        .and_then(var_name)
        .and_then(chelis_types::builtin_call_effect)
    {
        effects.insert(effect);
    }

    effects
}

fn infer_let_effects(
    kids: &[Expr],
    top_level_effects: &BTreeMap<String, EffectSet>,
    top_level_callables: &BTreeSet<String>,
    locals: &BTreeMap<String, EffectSet>,
) -> EffectSet {
    if kids.len() < 2 {
        return EffectSet::new();
    }

    let mut local_scope = locals.clone();
    let mut effects = EffectSet::new();

    match kids[0].carrier() {
        ExprCarrier::DecodedNode(DeepTag::Bind, _, bind_kids) => {
            let mut i = 0;
            while i + 1 < bind_kids.len() {
                let value = &bind_kids[i + 1];
                let value_effects =
                    infer_expr_effects(value, top_level_effects, top_level_callables, &local_scope);
                effects.extend(&value_effects);
                if let Some(name) = symbol_name(&bind_kids[i]) {
                    let binding_effects = if value.tag() == Some(DeepTag::Fn) {
                        value_effects
                    } else {
                        EffectSet::new()
                    };
                    local_scope.insert(name.to_string(), binding_effects);
                }
                i += 2;
            }
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => {
            effects.extend(&infer_expr_effects(
                &kids[0],
                top_level_effects,
                top_level_callables,
                &local_scope,
            ));
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
    kids: &[Expr],
    handled_effect: Option<EffectKind>,
    top_level_effects: &BTreeMap<String, EffectSet>,
    top_level_callables: &BTreeSet<String>,
    locals: &BTreeMap<String, EffectSet>,
) -> EffectSet {
    if kids.len() != 2 {
        return infer_children_effects(kids, top_level_effects, top_level_callables, locals);
    }
    let mut effects = infer_expr_effects(&kids[0], top_level_effects, top_level_callables, locals);
    let body_effects = infer_expr_effects(&kids[1], top_level_effects, top_level_callables, locals);
    match handled_effect {
        // A `resource` region places its body; it handles no inferred effect.
        Some(EffectKind::Resource) => {}
        // Validation reports the structural decode error. Inference leaves the
        // body's effects unhandled instead of substituting a known kind.
        None => {}
    }
    effects.extend(&body_effects);
    effects
}

fn annotate_effects(
    expr: &Expr,
    top_level_effects: &BTreeMap<String, EffectSet>,
    top_level_callables: &BTreeSet<String>,
    locals: &BTreeMap<String, EffectSet>,
) -> Expr {
    record_effect_work(|profile| profile.annotation_expr_visits += 1);
    match expr {
        Expr::Atom(_, _) => expr.clone(),
        Expr::Map(map, span) => Expr::Map(
            map.map_expressions(&mut |value, _| {
                annotate_effects(value, top_level_effects, top_level_callables, locals)
            })
            .expect("effect annotation preserves metadata payloads"),
            *span,
        ),
        Expr::MetaExpr(meta, span) => Expr::MetaExpr(
            chelis_deep::MetaExpr {
                expr: Box::new(annotate_effects(
                    &meta.expr,
                    top_level_effects,
                    top_level_callables,
                    locals,
                )),
                metadata: meta
                    .metadata
                    .map_expressions(&mut |value, _| {
                        annotate_effects(value, top_level_effects, top_level_callables, locals)
                    })
                    .expect("effect annotation preserves metadata payloads"),
            },
            *span,
        ),
        Expr::Node(node, span) => {
            let mut local_scope = locals.clone();
            let annotated_children =
                if node.tag() == DeepTag::Let && node.children_slice().len() >= 2 {
                    annotate_let_children(
                        node.children_slice(),
                        top_level_effects,
                        top_level_callables,
                        &mut local_scope,
                    )
                } else {
                    node.children_slice()
                        .iter()
                        .map(|child| {
                            annotate_effects(child, top_level_effects, top_level_callables, locals)
                        })
                        .collect()
                };
            let mut meta = node
                .meta()
                .map_expressions(&mut |value, _| {
                    annotate_effects(value, top_level_effects, top_level_callables, locals)
                })
                .expect("effect annotation preserves metadata payloads");
            if node.tag() == DeepTag::Fn {
                let effects =
                    infer_expr_effects(expr, top_level_effects, top_level_callables, locals);
                if !effects.is_empty() {
                    record_effect_work(|profile| profile.metadata_rewrites += 1);
                    meta.replace(MetadataValue::Effects(effect_set_metadata(&effects)));
                }
            }
            Expr::node(node.tag(), meta, annotated_children, *span)
        }
        Expr::BareList(elements, span) => Expr::BareList(
            elements
                .iter()
                .map(|child| {
                    annotate_effects(child, top_level_effects, top_level_callables, locals)
                })
                .collect(),
            *span,
        ),
        Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: data.head.clone(),
            meta: data
                .meta
                .map_expressions(&mut |value, _| {
                    annotate_effects(value, top_level_effects, top_level_callables, locals)
                })
                .expect("effect annotation preserves metadata payloads"),
            children: data
                .children
                .iter()
                .map(|child| {
                    annotate_effects(child, top_level_effects, top_level_callables, locals)
                })
                .collect(),
            span: data.span,
        })),
    }
}

fn annotate_let_children(
    kids: &[Expr],
    top_level_effects: &BTreeMap<String, EffectSet>,
    top_level_callables: &BTreeSet<String>,
    local_scope: &mut BTreeMap<String, EffectSet>,
) -> Vec<Expr> {
    let bind_kids = match kids[0].carrier() {
        ExprCarrier::DecodedNode(DeepTag::Bind, _, children) => children,
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => {
            return vec![
                annotate_effects(
                    &kids[0],
                    top_level_effects,
                    top_level_callables,
                    local_scope,
                ),
                annotate_effects(
                    &kids[1],
                    top_level_effects,
                    top_level_callables,
                    local_scope,
                ),
            ];
        }
    };

    let mut annotated_bind_kids = Vec::with_capacity(bind_kids.len());
    let mut index = 0;
    while index + 1 < bind_kids.len() {
        annotated_bind_kids.push(bind_kids[index].clone());
        let value = &bind_kids[index + 1];
        annotated_bind_kids.push(annotate_effects(
            value,
            top_level_effects,
            top_level_callables,
            local_scope,
        ));
        let value_effects =
            infer_expr_effects(value, top_level_effects, top_level_callables, local_scope);
        if let Some(name) = symbol_name(&bind_kids[index]) {
            let binding_effects = if value.tag() == Some(DeepTag::Fn) {
                value_effects
            } else {
                EffectSet::new()
            };
            local_scope.insert(name.to_string(), binding_effects);
        }
        index += 2;
    }
    annotated_bind_kids.extend(bind_kids[index..].iter().cloned());

    let annotated_bind = match &kids[0] {
        Expr::Node(bind_node, span) => Expr::node(
            bind_node.tag(),
            bind_node.meta().clone(),
            annotated_bind_kids,
            *span,
        ),
        _ => unreachable!("a decoded Bind carrier is an Expr::Node"),
    };

    vec![
        annotated_bind,
        annotate_effects(
            &kids[1],
            top_level_effects,
            top_level_callables,
            local_scope,
        ),
    ]
}

fn validate_handlers(exprs: &[Expr], errors: &mut Vec<EffectError>) {
    for expr in exprs {
        validate_handler_expr(expr, errors);
    }
}

fn validate_handler_expr(expr: &Expr, errors: &mut Vec<EffectError>) {
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            if tag == DeepTag::HandleEffect {
                validate_handler_kind(decode_effect_kind(metadata), children, errors);
            }
            metadata.visit_expressions(&mut |value, _| validate_handler_expr(value, errors));
            for kid in children {
                validate_handler_expr(kid, errors);
            }
        }
        ExprCarrier::MetadataMap(map) => {
            map.visit_expressions(&mut |value, _| validate_handler_expr(value, errors));
        }
        ExprCarrier::MetadataExpression(meta) => {
            validate_handler_expr(&meta.expr, errors);
            meta.metadata
                .visit_expressions(&mut |value, _| validate_handler_expr(value, errors));
        }
        ExprCarrier::Atom(_) => {}
        ExprCarrier::StructuralList(children) => {
            for child in children {
                validate_handler_expr(child, errors);
            }
        }
        ExprCarrier::UndecodableHead(_, _, children) => {
            for child in children {
                validate_handler_expr(child, errors);
            }
        }
    }
}

fn validate_handler_kind(
    effect_kind: Result<EffectKind, EffectKindDecodeError<'_>>,
    kids: &[Expr],
    errors: &mut Vec<EffectError>,
) {
    if kids.len() != 2 {
        errors.push(EffectError {
            kind: EffectErrorKind::InvalidHandler,
            message: format!(
                "`handle-effect` requires exactly two children (handler payload and body), got {}",
                kids.len()
            ),
            suggestions: vec!["Use `(handle-effect {effect: ...} <handler> <body>)`".to_string()],
        });
        return;
    }

    match effect_kind {
        Ok(EffectKind::Resource) if kids.first().and_then(string_literal).is_none() => {
            errors.push(EffectError {
                kind: EffectErrorKind::InvalidHandler,
                message: "with device(...) currently requires a string literal device".to_string(),
                suggestions: vec![
                    "Use `with device(\"gpu:0\") { ... }` with an explicit device literal"
                        .to_string(),
                ],
            });
        }
        Ok(EffectKind::Resource) => {}
        Err(error) => errors.push(EffectError {
            kind: EffectErrorKind::InvalidHandler,
            message: format!("{error} in `handle-effect`"),
            suggestions: vec![
                "Use the closed effect kind `resource`; randomness has no handler, \
                 a random primitive takes an explicit key"
                    .to_string(),
            ],
        }),
    }
}

/// Extract the effect set declared on a
/// `(defsig name [(binders...)] t-fn-with-eff-meta)` expression.
/// Returns `None` when there is no explicit effect annotation (i.e., inference-only mode).
fn declared_effects_from_defsig(expr: &Expr) -> Option<EffectSet> {
    // The type is last in both canonical defsig forms.
    let kids = match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Defsig, _, children) => children,
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => return None,
    };
    let t_fn = kids.last()?;
    let meta = match t_fn.carrier() {
        ExprCarrier::DecodedNode(DeepTag::TFn, metadata, _) => metadata,
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => return None,
    };
    let effects = meta.eff()?;
    let mut declared = EffectSet::new();
    for member in effects.values() {
        match member {
            EffectMember::Name(name) => match name.value().as_str() {
                "accum" => declared.insert(Effect::Accum),
                "io" => declared.insert(Effect::Io),
                "test" => declared.insert(Effect::Test),
                // Diff is tracked separately from the inferred effect row.
                _ => {}
            },
            EffectMember::Resource(resource) => {
                declared.insert(Effect::Resource(resource.name().value().clone()))
            }
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
    effects_by_def: &BTreeMap<String, EffectSet>,
    errors: &mut Vec<EffectError>,
) {
    let mut declared_by_name: BTreeMap<String, EffectSet> = BTreeMap::new();
    for expr in flattened_top_level(exprs) {
        match expr.carrier() {
            ExprCarrier::DecodedNode(DeepTag::Defsig, _, children) => {
                let Some(name) = children.first().and_then(symbol_name) else {
                    continue;
                };
                if let Some(declared) = declared_effects_from_defsig(expr) {
                    declared_by_name.insert(name.to_string(), declared);
                }
            }
            ExprCarrier::DecodedNode(_, _, _)
            | ExprCarrier::StructuralList(_)
            | ExprCarrier::UndecodableHead(_, _, _)
            | ExprCarrier::Atom(_)
            | ExprCarrier::MetadataMap(_)
            | ExprCarrier::MetadataExpression(_) => {}
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
    match expr.carrier() {
        ExprCarrier::DecodedNode(tag, metadata, children) => {
            if tag == DeepTag::HandleEffect {
                validate_build_target_handler(
                    decode_effect_kind(metadata),
                    children,
                    target,
                    errors,
                );
            }
            metadata.visit_expressions(&mut |value, _| {
                validate_build_target_expr(value, target, errors)
            });
            for kid in children {
                validate_build_target_expr(kid, target, errors);
            }
        }
        ExprCarrier::MetadataMap(map) => {
            map.visit_expressions(&mut |value, _| {
                validate_build_target_expr(value, target, errors)
            });
        }
        ExprCarrier::MetadataExpression(meta) => {
            validate_build_target_expr(&meta.expr, target, errors);
            meta.metadata.visit_expressions(&mut |value, _| {
                validate_build_target_expr(value, target, errors)
            });
        }
        ExprCarrier::Atom(_) => {}
        ExprCarrier::StructuralList(children) => {
            for child in children {
                validate_build_target_expr(child, target, errors);
            }
        }
        ExprCarrier::UndecodableHead(_, _, children) => {
            for child in children {
                validate_build_target_expr(child, target, errors);
            }
        }
    }
}

fn validate_build_target_handler(
    effect_kind: Result<EffectKind, EffectKindDecodeError<'_>>,
    kids: &[Expr],
    target: &str,
    errors: &mut Vec<EffectError>,
) {
    match effect_kind {
        Ok(EffectKind::Resource) => {
            if let Some(device) = kids.first().and_then(string_literal) {
                let ok = match target {
                    "c" => is_c_host_device_designator(device),
                    "hip" | "metal" => device.starts_with("gpu"),
                    _ => true,
                };
                if !ok {
                    errors.push(EffectError {
                        kind: EffectErrorKind::BuildTargetMismatch,
                        message: if target == "c" {
                            format!(
                                "`chelis build --target c` cannot satisfy resource region \
                                 `{device}`: host C accepts only exact `cpu`"
                            )
                        } else {
                            format!(
                                "`chelis build --target {target}` cannot satisfy resource region \
                                 `{device}`"
                            )
                        },
                        suggestions: match target {
                            "c" => vec![
                                "Use `with device(\"cpu\") { ... }` for host execution; \
                                 accelerator placement is a separate target capability \
                                 (chelis#2104)"
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
        // `validate_handlers` is the owning structural diagnostic pass. Do
        // not assign target semantics after decode fails.
        Err(_) => {}
    }
}

/// The host-C selector contract owned by spec/04 [04-EFF-2].
///
/// This is deliberately a positive host classification. Any new accelerator
/// spelling therefore fails closed instead of inheriting host execution merely
/// because it lacks a known prefix.
fn is_c_host_device_designator(device: &str) -> bool {
    device == "cpu"
}

/// Convert semantic effects to their dedicated AST annotation payload.
fn effect_set_metadata(effects: &EffectSet) -> AstEffectSet {
    let values = effects
        .iter()
        .map(|effect| match effect {
            Effect::Accum => EffectMember::Name(Spanned::new("accum".into(), zero_span())),
            Effect::Io => EffectMember::Name(Spanned::new("io".into(), zero_span())),
            Effect::Test => EffectMember::Name(Spanned::new("test".into(), zero_span())),
            Effect::Resource(device) => EffectMember::Resource(
                ResourceEffect::new(
                    Spanned::new(device.clone(), zero_span()),
                    Metadata::default(),
                    zero_span(),
                )
                .expect("resource effect annotations"),
            ),
        })
        .collect();
    AstEffectSet::new(Metadata::default(), values, zero_span())
}

fn var_name(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        ExprCarrier::DecodedNode(DeepTag::Var, _, children) => {
            children.first().and_then(symbol_name)
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
}

fn symbol_name(expr: &Expr) -> Option<&str> {
    match expr {
        Expr::Atom(Atom::Name(name), _) => Some(name.as_str()),
        _ => None,
    }
}

fn string_literal(expr: &Expr) -> Option<&str> {
    match expr.carrier() {
        ExprCarrier::Atom(Atom::Str(value)) => Some(value.as_str()),
        ExprCarrier::DecodedNode(DeepTag::Lit, _, children) => {
            children.first().and_then(|child| match child.carrier() {
                ExprCarrier::Atom(Atom::Str(value)) => Some(value.as_str()),
                ExprCarrier::DecodedNode(_, _, _)
                | ExprCarrier::StructuralList(_)
                | ExprCarrier::UndecodableHead(_, _, _)
                | ExprCarrier::Atom(_)
                | ExprCarrier::MetadataMap(_)
                | ExprCarrier::MetadataExpression(_) => None,
            })
        }
        ExprCarrier::DecodedNode(_, _, _)
        | ExprCarrier::StructuralList(_)
        | ExprCarrier::UndecodableHead(_, _, _)
        | ExprCarrier::Atom(_)
        | ExprCarrier::MetadataMap(_)
        | ExprCarrier::MetadataExpression(_) => None,
    }
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
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
        let checked = chelis_types::check_ir_program(&deep).expect("type check");
        check_program(&checked).expect("effect check")
    }

    fn issue_1205_source(operations: usize, flat: bool) -> String {
        let module = if flat { "Flat" } else { "Nested" };
        let mut lines = vec![
            format!("module FrontEndPerformance.{module}N{operations}"),
            "def bc(c: f32) -> tensor[8, f32] = \
             reshape(insert(to_tensor([c]), 0, 8i64), [8i64])"
                .to_string(),
        ];
        if flat {
            lines.push(
                "def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = \
                 if gte(i, 5i64) then s else {"
                    .to_string(),
            );
            let mut previous = "s".to_string();
            for index in 0..operations {
                lines.push(format!(
                    "  t{index} = mul(add({previous}, bc(cast(1.0, f32))), \
                     bc(cast(0.5, f32)))"
                ));
                previous = format!("t{index}");
            }
            lines.extend([format!("  st({previous}, add(i, 1i64))"), "}".to_string()]);
        } else {
            let mut body = "s".to_string();
            for _ in 0..operations {
                body = format!("mul(add({body}, bc(cast(1.0, f32))), bc(cast(0.5, f32)))");
            }
            lines.push(format!(
                "def st(s: tensor[8, f32], i: i64) -> tensor[8, f32] = \
                 if gte(i, 5i64) then s else st({body}, add(i, 1i64))"
            ));
        }
        lines.push("r = index(to_list(st(bc(cast(1.0, f32)), 0i64)), 0i64)".to_string());
        lines.join("\n") + "\n"
    }

    fn issue_1205_effect_profile(operations: usize, flat: bool) -> EffectWorkProfile {
        std::thread::Builder::new()
            .name(format!("issue-1205-effects-{operations}"))
            .stack_size(64 * 1024 * 1024)
            .spawn(move || {
                let source = issue_1205_source(operations, flat);
                let decls = parse_surf(&source).expect("#1205 surf fixture parses");
                let deep = desugar_program(&decls).expect("Surf fixture must desugar");
                let typed =
                    chelis_types::check_ir_program(&deep).expect("#1205 fixture type checks");
                reset_effect_work_profile();
                check_program(&typed).expect("#1205 fixture passes effect checking");
                take_effect_work_profile()
            })
            .expect("#1205 effect profile thread starts")
            .join()
            .expect("#1205 effect profile thread completes")
    }

    #[test]
    fn issue_1205_effect_clone_work_is_linear() {
        for flat in [false, true] {
            let mut previous = None;
            for operations in [20, 40, 80, 160] {
                let profile = issue_1205_effect_profile(operations, flat);
                let clone_work = profile.node_bridge_clone_nodes
                    + profile.list_rewrite_clone_nodes
                    + profile.top_level_body_clone_nodes;
                let total_work = clone_work
                    + profile.infer_expr_visits
                    + profile.annotation_expr_visits
                    + profile.metadata_rewrites;
                eprintln!(
                    "#1205 effects shape={} n={operations} profile={profile:?} clone_work={clone_work}",
                    if flat { "flat" } else { "nested" }
                );
                assert_eq!(
                    clone_work, 0,
                    "#1205 effects must not clone descendant-bearing expression trees"
                );
                if let Some(previous) = previous {
                    assert!(
                        total_work * 10 <= previous * 22,
                        "#1205 effect clone work must grow linearly: shape={} n={operations} \
                         previous={previous} current={total_work} profile={profile:?}",
                        if flat { "flat" } else { "nested" }
                    );
                }
                previous = Some(total_work);
            }
        }
    }

    #[test]
    fn effect_annotation_reconstruction_preserves_type_context() {
        let decls = parse_surf("def add_one(x: i32) -> i32 = add(x, 1)").expect("surf parse");
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
        let typed = chelis_types::check_ir_program(&deep).expect("type check");
        let expected_type_env = typed.type_env().clone();
        let expected_signatures = typed.signature_inference().clone();

        let reconstructed = check_program(&typed).expect("effect check");
        assert_eq!(reconstructed.type_env(), &expected_type_env);
        assert_eq!(reconstructed.signature_inference(), &expected_signatures);
    }

    #[test]
    fn stamped_effect_annotation_preserves_the_checked_node_carrier() {
        let deep = chelis_deep::parse_and_stamp("(def {} answer (fn {} (params {}) (lit {} 1)))")
            .expect("canonical Deep fixture stamps");
        let typed = chelis_types::check_typed_program(&deep).expect("type check");

        let reconstructed = check_program(&typed).expect("effect check");

        assert!(
            matches!(
                reconstructed.annotated_exprs().first(),
                Some(Expr::Node(..))
            ),
            "effect annotation must not normalize stamped checker output"
        );
    }

    #[test]
    fn stamped_declared_pure_function_rejects_inferred_io() {
        let deep = chelis_deep::parse_and_stamp(
            r#"(defsig {} entry
                 (t-fn {eff: (effects {})}
                   (t-prim {} string)
                   (t-prim {} string)))
               (def {} entry
                 (fn {}
                   (params {} (x {type: (t-prim {} string)}))
                   (app {} (var {} debug) (var {} x))))"#,
        )
        .expect("canonical Deep fixture stamps");
        let typed = chelis_types::check_typed_program(&deep).expect("type check");

        let errors = check_program(&typed).expect_err("declared-pure IO body must reject");

        assert!(
            errors
                .iter()
                .any(|error| { error.message.contains("entry") && error.message.contains("IO") }),
            "effect diagnostic must name the function and missing effect: {errors:?}"
        );
    }

    fn effect_error_messages(errors: Vec<EffectError>) -> Vec<String> {
        errors
            .into_iter()
            .map(|error| format!("{:?}: {}", error.kind, error.message))
            .collect()
    }

    #[test]
    fn effects_reader_classes_read_a_recursive_stamped_program() {
        let successor = chelis_deep::parse_and_stamp_file(
            r#"(module {} Test
                 (defsig {} entry
                   (t-fn {eff: (effects {})} (t-prim {} unit)))
                 (def {} entry
                   (fn {}
                     (params {})
                     (let {}
                       (bind {} local
                         (fn {}
                           (params {})
                           (app {} (var {} debug) (lit {} 1))))
                       (app {} (var {} local))))))"#,
        )
        .expect("recursive effects fixture stamps");

        let successor_bodies = top_level_def_bodies(&successor);
        assert_eq!(successor_bodies.keys().collect::<Vec<_>>(), ["entry"]);

        let (successor_effects, successor_callables) = infer_program_effects(&successor);
        assert!(
            successor_effects
                .get("entry")
                .is_some_and(|effects| effects.contains(&Effect::Io))
        );

        let mut successor_errors = Vec::new();
        validate_declared_vs_inferred(&successor, &successor_effects, &mut successor_errors);
        assert!(
            effect_error_messages(successor_errors)
                .iter()
                .any(|message| message.contains("`entry`")),
            "the declared-pure `entry` must be rejected for its local closure's IO"
        );

        let successor_annotated = successor
            .iter()
            .map(|expr| {
                annotate_effects(
                    expr,
                    &successor_effects,
                    &successor_callables,
                    &BTreeMap::new(),
                )
            })
            .collect::<Vec<_>>();
        assert!(
            chelis_deep::printer::print_canonical(&successor_annotated).contains("eff"),
            "the annotated closure carries its inferred effect row"
        );
    }

    #[test]
    fn guarded_var_and_literal_readers_read_decoded_nodes() {
        let span = Span::new(0, 0);
        let successor_var = Expr::node(
            DeepTag::Var,
            Metadata::default(),
            vec![Expr::Atom(Atom::Name("value".into()), span)],
            span,
        );
        let successor_lit = Expr::node(
            DeepTag::Lit,
            Metadata::default(),
            vec![Expr::Atom(Atom::Str("gpu:0".into()), span)],
            span,
        );
        assert_eq!(var_name(&successor_var), Some("value"));
        assert_eq!(string_literal(&successor_lit), Some("gpu:0"));

        let wrong_tag = Expr::node(
            DeepTag::Var,
            Metadata::default(),
            vec![Expr::Atom(Atom::Str("gpu:0".into()), span)],
            span,
        );
        assert_eq!(string_literal(&wrong_tag), None);
        assert_eq!(
            var_name(&Expr::BareList(
                vec![Expr::Atom(Atom::Name("value".into()), span)],
                span,
            )),
            None
        );
    }

    #[test]
    fn undecodable_head_metadata_is_not_traversed_by_handler_validation() {
        fn stamped_def_body(source: &str) -> Expr {
            let exprs = chelis_deep::parse_and_stamp(source).expect("fixture stamps");
            let ExprCarrier::DecodedNode(DeepTag::Def, _, children) = exprs[0].carrier() else {
                panic!("fixture is a def");
            };
            children[1].clone()
        }

        fn expression_metadata(expr: Expr) -> Metadata {
            Metadata::from(MetadataValue::PropertySeed(
                chelis_deep::annotations::RuntimeExpression::try_new(expr)
                    .expect("handler is a runtime expression"),
            ))
        }

        fn unknown_form(metadata: Metadata, span: Span) -> Expr {
            Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
                head: "future-wrapper".into(),
                meta: metadata,
                children: Vec::new(),
                span,
            }))
        }

        let span = Span::new(0, 0);
        let invalid_resource = stamped_def_body(
            "(def {} x
               (handle-effect {effect: resource}
                 (var {} device)
                 (lit {} 1)))",
        );
        let invalid_metadata = expression_metadata(invalid_resource);
        let mut unknown_errors = Vec::new();
        validate_handler_expr(&unknown_form(invalid_metadata, span), &mut unknown_errors);
        assert!(
            unknown_errors.is_empty(),
            "UnknownForm metadata was not traversed before the carrier migration: {unknown_errors:?}"
        );

        let gpu_resource = stamped_def_body(
            "(def {} x
               (handle-effect {effect: resource}
                 (lit {} \"gpu:0\")
                 (lit {} 1)))",
        );
        let resource_metadata = expression_metadata(gpu_resource);
        let mut unknown_target_errors = Vec::new();
        validate_build_target_expr(
            &unknown_form(resource_metadata, span),
            "c",
            &mut unknown_target_errors,
        );
        assert!(
            unknown_target_errors.is_empty(),
            "UnknownForm metadata was not traversed by target validation before migration: \
             {unknown_target_errors:?}"
        );
    }

    #[test]
    fn let_effect_reader_traverses_every_non_bind_slot_carrier() {
        let span = Span::new(0, 0);
        let name = |value: &str| Expr::Atom(Atom::Name(value.into()), span);
        let io_expr = || {
            Expr::node(
                DeepTag::App,
                Metadata::default(),
                vec![
                    Expr::node(DeepTag::Var, Metadata::default(), vec![name("debug")], span),
                    Expr::node(
                        DeepTag::Lit,
                        Metadata::default(),
                        vec![Expr::Atom(Atom::Int(1), span)],
                        span,
                    ),
                ],
                span,
            )
        };
        let pure_expr = || {
            Expr::node(
                DeepTag::Lit,
                Metadata::default(),
                vec![Expr::Atom(Atom::Int(0), span)],
                span,
            )
        };

        let carrier_cases = |child: Expr| {
            [
                ("structural-list", Expr::BareList(vec![child.clone()], span)),
                (
                    "undecodable-head",
                    Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
                        head: "future-bind".into(),
                        meta: Metadata::default(),
                        children: vec![child.clone()],
                        span,
                    })),
                ),
                (
                    "metadata-expression",
                    Expr::MetaExpr(
                        chelis_deep::MetaExpr {
                            metadata: Metadata::default(),
                            expr: Box::new(child.clone()),
                        },
                        span,
                    ),
                ),
                (
                    "decoded-non-bind",
                    Expr::node(DeepTag::Tuple, Metadata::default(), vec![child], span),
                ),
            ]
        };
        let effectful_cases = carrier_cases(io_expr());
        assert!(matches!(
            effectful_cases[0].1.carrier(),
            ExprCarrier::StructuralList(_)
        ));
        assert!(matches!(
            effectful_cases[1].1.carrier(),
            ExprCarrier::UndecodableHead(..)
        ));
        assert!(matches!(
            effectful_cases[2].1.carrier(),
            ExprCarrier::MetadataExpression(_)
        ));
        assert!(matches!(
            effectful_cases[3].1.carrier(),
            ExprCarrier::DecodedNode(DeepTag::Tuple, ..)
        ));

        let hidden = effectful_cases
            .iter()
            .filter_map(|(label, bind_slot)| {
                let effects = infer_let_effects(
                    &[bind_slot.clone(), pure_expr()],
                    &BTreeMap::new(),
                    &BTreeSet::new(),
                    &BTreeMap::new(),
                );
                (!effects.contains(&Effect::Io)).then_some(*label)
            })
            .collect::<Vec<_>>();
        assert!(
            hidden.is_empty(),
            "bind-slot carriers hide effectful content: {hidden:?}"
        );

        let pure_cases = carrier_cases(pure_expr());
        let invented = pure_cases
            .iter()
            .filter_map(|(label, bind_slot)| {
                let effects = infer_let_effects(
                    &[bind_slot.clone(), pure_expr()],
                    &BTreeMap::new(),
                    &BTreeSet::new(),
                    &BTreeMap::new(),
                );
                effects.contains(&Effect::Io).then_some(*label)
            })
            .collect::<Vec<_>>();
        assert!(
            invented.is_empty(),
            "bind-slot carrier traversal invents effects: {invented:?}"
        );
    }

    #[test]
    fn declared_effect_reader_reads_a_decoded_defsig() {
        let mut parsed = chelis_deep::parse_and_stamp(
            "(defsig {} entry \
               (t-fn {eff: (effects {} accum io)} \
                 (t-prim {} i32) \
                 (t-prim {} i32)))",
        )
        .expect("canonical defsig fixture stamps");
        let successor = parsed.remove(0);

        let successor_effects =
            declared_effects_from_defsig(&successor).expect("successor declaration");
        assert!(successor_effects.contains(&Effect::Accum));
        assert!(successor_effects.contains(&Effect::Io));
    }

    #[test]
    fn declared_effect_reader_declines_nondecoded_carriers() {
        let span = Span::new(0, 0);
        for expr in [
            Expr::BareList(vec![], span),
            Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
                head: "future-defsig".into(),
                meta: Metadata::default(),
                children: vec![],
                span,
            })),
        ] {
            assert_eq!(declared_effects_from_defsig(&expr), None);
        }
    }

    #[test]
    fn declared_effect_reader_has_no_local_optional_carrier_adapter() {
        let source = include_str!("lib.rs");
        let definition = ["fn stamped_", "parts"].concat();
        assert!(
            !source.contains(&definition),
            "E5b requires declared-effect reads to disposition ExprCarrier directly"
        );
    }

    #[test]
    fn stamped_declared_pure_function_rejects_io_local_closure() {
        let deep = chelis_deep::parse_and_stamp(
            r#"(defsig {} entry
                 (t-fn {eff: (effects {})}
                   (t-prim {} string)
                   (t-prim {} string)))
               (def {} entry
                 (fn {}
                   (params {} (x {type: (t-prim {} string)}))
                   (let {}
                     (bind {} step
                       (fn {}
                         (params {} (y {type: (t-prim {} string)}))
                         (app {} (var {} debug) (var {} y))))
                     (app {} (var {} step) (var {} x)))))"#,
        )
        .expect("canonical Deep fixture stamps");
        let typed = chelis_types::check_typed_program(&deep).expect("type check");

        let errors = check_program(&typed).expect_err("declared-pure local IO closure must reject");

        assert!(
            errors
                .iter()
                .any(|error| { error.message.contains("entry") && error.message.contains("IO") }),
            "effect diagnostic must name the function and local closure effect: {errors:?}"
        );
    }

    #[test]
    fn literal_resource_handler_crosses_the_type_effect_boundary() {
        let deep = parse_str(
            r#"(def {} value
                   (handle-effect {effect: resource}
                     (lit {type: (t-prim {} string)} "cpu")
                     (lit {type: (t-prim {} i32)} 1)))"#,
        )
        .expect("Deep handler fixture parses");
        let typed = chelis_types::check_ir_program(&deep).expect("type boundary accepts");
        check_program(&typed).expect("effects boundary accepts literal handler");
    }

    #[test]
    fn nonliteral_resource_handler_is_rejected_once_by_the_effect_owner() {
        let deep = parse_str(
            "(def {} value (handle-effect {effect: resource} \
             (var {} computed_handler) (lit {type: (t-prim {} i32)} 1)))",
        )
        .expect("Deep handler fixture parses");
        let typed = chelis_types::check_ir_program(&deep)
            .expect("handler payload is owned by the effects gate");
        let errors = check_program(&typed).expect_err("nonliteral handler must reject");
        assert_eq!(errors.len(), 1, "one effects owner diagnostic: {errors:?}");
        assert_eq!(errors[0].kind, EffectErrorKind::InvalidHandler);
        assert!(
            errors[0]
                .message
                .contains("requires a string literal device"),
            "{errors:?}"
        );
    }

    #[test]
    fn keyed_dropout_introduces_no_effect() {
        // [05-RNG-1]: a draw is a pure function of its key, so a keyed
        // `dropout` carries an empty effect row (the `Random` effect was
        // retired with the counter stream, #2413).
        let exprs = parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} dropout)
               (app {} (var {} key_from_seed) (lit {type: (t-prim {} i64)} 7))
               (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        )
        .unwrap();
        let (inferred, _) = infer_program_effects(&exprs);
        assert!(
            inferred.get("y").is_some_and(|effects| effects.is_empty()),
            "keyed dropout must infer an empty effect row, got {:?}",
            inferred.get("y")
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
def pure_add(x: i64, y: i64) -> i64 = add(x, y)
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
    fn keyed_dropout_root_checks_clean() {
        // The retired unhandled-`Random` root check has no successor: a keyed
        // draw at a value root needs no handler.
        let exprs = parse_str(
            "(def {} x (lit {type: (t-tensor {} (d-lit {} 8) (t-prim {} f32))} 0))
             (def {} y (app {} (var {} dropout)
               (app {} (var {} key_from_seed) (lit {type: (t-prim {} i64)} 7))
               (var {} x) (lit {type: (t-prim {} f32)} 0.5)))",
        )
        .unwrap();
        let checked = chelis_types::check_ir_program(&exprs).unwrap();
        check_program(&checked).expect("a keyed draw root needs no handler");
    }

    #[test]
    fn build_target_rejects_gpu_region_for_c() {
        let program = checked(
            "(def {} x
               (handle-effect {effect: resource}
                 (lit {type: (t-prim {} string)} \"gpu:0\")
                 (lit {type: (t-prim {} i32)} 1)))",
        );
        let errors = validate_build_target(&program, "c").unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.kind == EffectErrorKind::BuildTargetMismatch)
        );
    }

    #[test]
    fn c_target_accepts_only_explicit_host_resource_designators() {
        let program = surf_checked("def main() -> i32 = with device(\"cpu\") { 1 }\n");
        validate_build_target(&program, "c").expect("exact cpu host region");

        for device in [
            "cpu:0",
            "cpu:author-device",
            "cpu:socket_9",
            "cpu:HOST_2",
            "cuda:0",
            "metal",
            "rocm",
            "xpu:1",
            "Gpu:0",
            "gpu:0",
            "host",
            "",
            "cpu:",
            "cpu:two words",
            "cpu:/0",
        ] {
            let program = surf_checked(&format!(
                "def main() -> i32 = with device(\"{device}\") {{ 1 }}\n"
            ));
            let errors = validate_build_target(&program, "c").expect_err(device);
            assert_eq!(errors.len(), 1, "{device}: {errors:?}");
            assert_eq!(
                errors[0].kind,
                EffectErrorKind::BuildTargetMismatch,
                "{device}"
            );
            assert_eq!(
                errors[0].message,
                format!(
                    "`chelis build --target c` cannot satisfy resource region `{device}`: \
                     host C accepts only exact `cpu`"
                ),
                "{device}"
            );
            assert_eq!(
                errors[0].suggestions,
                vec![
                    "Use `with device(\"cpu\") { ... }` for host execution; accelerator \
                     placement is a separate target capability (chelis#2104)"
                        .to_string()
                ],
                "{device}"
            );
        }
    }

    #[test]
    fn c_target_checks_each_nested_resource_region() {
        for source in [
            r#"def main() -> i32 = with device("cpu") { with device("cuda:0") { 1 } }"#,
            r#"def main() -> i32 = with device("metal") { with device("cpu") { 1 } }"#,
            r#"def main() -> i32 = with device("cpu") { with device("cpu:author-device") { 1 } }"#,
            r#"def main() -> i32 = with device("cpu:socket_9") { with device("cpu") { 1 } }"#,
        ] {
            let program = surf_checked(source);
            let errors = validate_build_target(&program, "c").expect_err(source);
            assert_eq!(errors.len(), 1, "{source}: {errors:?}");
            assert_eq!(errors[0].kind, EffectErrorKind::BuildTargetMismatch);
        }

        let accepted =
            surf_checked(r#"def main() -> i32 = with device("cpu") { with device("cpu") { 1 } }"#);
        validate_build_target(&accepted, "c").expect("nested host regions");

        let two_rejected = surf_checked(
            r#"def main() -> i32 = with device("cpu:author-device") { with device("cpu:socket_9") { 1 } }"#,
        );
        let errors = validate_build_target(&two_rejected, "c").expect_err("two rejected regions");
        assert_eq!(errors.len(), 2, "{errors:?}");
        assert_eq!(
            errors
                .iter()
                .map(|error| error.message.as_str())
                .collect::<Vec<_>>(),
            [
                "`chelis build --target c` cannot satisfy resource region \
                 `cpu:author-device`: host C accepts only exact `cpu`",
                "`chelis build --target c` cannot satisfy resource region \
                 `cpu:socket_9`: host C accepts only exact `cpu`",
            ]
        );
    }

    #[test]
    fn checked_program_does_not_emit_internal_type_override_metadata() {
        let program = checked(
            "(def {} f
               (fn {} (params {} (x {type: (t-prim {} string)}))
                 (app {} (var {} debug) (var {} x))))",
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
def emit(x: i64) -> i64 = debug(add(x, cast(1, i64)))
xs: List[i64] = [cast(1, i64), cast(2, i64)]
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
xs: List[i64] = [cast(1, i64), cast(2, i64)]
total = fold(fn (acc: i64, x: i64) -> debug(add(acc, x)), cast(0, i64), xs)
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
xs: List[i64] = [cast(1, i64), cast(2, i64)]
totals = scan(fn (acc: i64, x: i64) -> debug(add(acc, x)), cast(0, i64), xs)
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
    fn partition_propagates_io_effect_from_callback() {
        let program = surf_checked(
            r#"
def keep(x: i64) -> bool = gt(debug(x), cast(1, i64))
xs: List[i64] = [cast(1, i64), cast(2, i64)]
buckets = partition(keep, xs)
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("buckets")
                .is_some_and(|effects| effects.contains(&Effect::Io)),
            "expected IO effect on partition result, got {:?}",
            inferred.get("buckets")
        );
    }

    #[test]
    fn flat_map_propagates_io_effect_from_callback() {
        let program = surf_checked(
            r#"
xs: List[i64] = [cast(1, i64), cast(2, i64)]
ys = flat_map(fn (x: i64) -> debug([x, add(x, cast(10, i64))]), xs)
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

    /// [04-INF-9] (chelis#3149): a builtin named as a value performs its
    /// call's effect, as a lambda that calls it does, so an alias cannot shed
    /// `IO` or `Test`. Before the fix `reader` and `caller` were pure.
    #[test]
    fn a_builtin_named_as_a_value_performs_its_call_effect() {
        let program = surf_checked(
            r#"
reader = read_file
def caller(path: string) -> string = reader(path)
def local_alias(path: string) -> string = {
  op = read_file
  op(path)
}
checker = test_assert
def via_lambda(path: string) -> string = {
  op = fn (p: string) -> read_file(p)
  op(path)
}
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        for (name, effect) in [
            ("reader", Effect::Io),
            ("caller", Effect::Io),
            ("local_alias", Effect::Io),
            ("via_lambda", Effect::Io),
            ("checker", Effect::Test),
        ] {
            assert!(
                inferred
                    .get(name)
                    .is_some_and(|effects| effects.contains(&effect)),
                "expected {effect:?} on `{name}`, got {:?}",
                inferred.get(name)
            );
        }
    }

    #[test]
    fn file_io_builtins_infer_io_but_mmap_reads_stay_pure() {
        let program = surf_checked(
            r#"
mapped = mmap_file("dataset.txt")
prefix = mmap_read(mapped, cast(0, i64), cast(4, i64))
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
pure_value = add(cast(1, i64), cast(2, i64))
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

    #[test]
    fn clock_reads_infer_io_but_pure_binding_stays_pure() {
        // [05-OP-75]: each clock read is an IO source; an adjacent pure
        // binding stays effect-free.
        let program = surf_checked(
            r#"
wall = clock_wall_read()
mono = clock_monotonic_read()
pure_value = add(1i64, 2i64)
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        for root in ["wall", "mono"] {
            assert!(
                inferred
                    .get(root)
                    .is_some_and(|effects| effects.contains(&Effect::Io)),
                "expected IO effect on clock root {root}, got {:?}",
                inferred.get(root)
            );
        }
        assert!(
            inferred
                .get("pure_value")
                .is_none_or(|effects| effects.is_empty()),
            "expected pure arithmetic binding to stay effect-free, got {:?}",
            inferred.get("pure_value")
        );
    }

    #[test]
    fn declared_pure_function_cannot_read_a_clock() {
        // [05-OP-75] and spec/04 section 7: a function whose declared effects
        // omit IO cannot read either clock, directly or through a caller.
        // Declaring IO admits the same bodies.
        for clock in ["clock_wall_read", "clock_monotonic_read"] {
            for (declared, admitted) in [("{}", false), ("{Test}", false), ("{IO}", true)] {
                let decls = parse_surf(&format!(
                    "def read() -> (i64, i64) ! {declared} = {clock}()\n\
                     def outer() -> (i64, i64) ! {declared} = read()\n"
                ))
                .expect("surf parse");
                let deep = desugar_program(&decls).expect("Surf fixture must desugar");
                let checked = chelis_types::check_ir_program(&deep).expect("type check");
                let verdict = check_program(&checked);
                if admitted {
                    assert!(verdict.is_ok(), "{clock} under ! {declared}: {verdict:?}");
                    continue;
                }
                let errors = verdict.expect_err("a clock read needs a declared IO effect");
                for function in ["read", "outer"] {
                    assert!(
                        errors.iter().any(|error| {
                            error.kind == EffectErrorKind::UnhandledEffect
                                && error.message.contains(&format!("`{function}`"))
                                && error.message.contains("IO")
                        }),
                        "expected UnhandledEffect on {function} mentioning IO for {clock} \
                         under ! {declared}, got {errors:?}"
                    );
                }
            }
        }
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
    fn with_device_does_not_handle_test_effect() {
        // A `resource` handler handles no inferred effect: `with device(...)`
        // must NOT remove Test.
        let program = surf_checked(
            r#"
def sealed() -> unit =
  with device("cpu") { test_assert(true, "inside-handler") }
"#,
        );
        let (inferred, _) = infer_program_effects(program.annotated_exprs());
        assert!(
            inferred
                .get("sealed")
                .is_some_and(|effects| effects.contains(&Effect::Test)),
            "with device(...) must not swallow the Test effect, got {:?}",
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
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
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
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
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
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
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
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
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
    // descent, a declared-pure function whose body performs IO would be
    // accepted module-wrapped but rejected flattened.

    /// Type-check a module-WRAPPED Deep program (the shape `chelis check` runs
    /// on a `.dp` MODULE), preserving the `(module ...)` wrapper.
    fn typed_module(src: &str) -> CheckedProgram {
        let decls = parse_surf(src).expect("surf parse");
        let deep = desugar_program(&decls).expect("Surf fixture must desugar");
        chelis_types::check_typed_program(&deep).expect("type check")
    }

    #[test]
    fn module_wrapped_declared_pure_body_does_io_is_rejected() {
        // A declared-pure function whose body calls a file-IO builtin must be
        // rejected module-wrapped, the same as flattened.
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
    fn module_wrapped_honest_io_signature_checks_clean() {
        // A module-wrapped function that honestly declares `! { IO }` must
        // check clean: the descent fix tightens the unsound-accept path only,
        // not honest code.
        let checked = typed_module(
            r#"module Frag.Honest
export (entry)
def entry(path: string) -> string ! { IO } = read_file(path)
"#,
        );
        check_program(&checked).expect("honest IO module-wrapped program must check clean");
    }
}

#[cfg(test)]
mod decode_once_producer_tests {
    use super::*;

    #[test]
    fn synthesized_effects_preserve_typed_members_and_decoded_wire_tags() {
        let mut effects = EffectSet::new();
        effects.insert(Effect::Accum);
        effects.insert(Effect::Io);
        effects.insert(Effect::Resource("gpu0".into()));
        let metadata = Metadata::from(MetadataValue::Effects(effect_set_metadata(&effects)));
        let mut payload = None;
        metadata.visit_syntax(&mut |_, value| payload = Some(value.clone()));
        let expr = payload.unwrap();
        assert_eq!(
            chelis_deep::validate::find_raw_vocabulary_tag(std::slice::from_ref(&expr)),
            None
        );
        assert_eq!(expr.tag(), Some(DeepTag::Effects));
        let Expr::Node(node, _) = expr else {
            panic!("structural effects node")
        };
        let resource = node
            .children_slice()
            .iter()
            .find(|child| child.tag() == Some(DeepTag::Resource))
            .unwrap();
        assert_eq!(resource.tag(), Some(DeepTag::Resource));
    }

    #[test]
    fn effect_names_are_payload_not_vocabulary_tags() {
        let mut effects = EffectSet::new();
        effects.insert(Effect::Accum);
        let payload = effect_set_metadata(&effects);
        assert!(matches!(payload.values(), [EffectMember::Name(name)] if name.value() == "accum"));
    }

    #[test]
    fn effect_annotation_reaches_registered_expressions_and_preserves_data() {
        let source = "(def {property_seed: (fn {} (params {}) (app {} (var {} debug) (lit {} 1))), custom: (fn {} (params {}) (app {} (var {} debug) (lit {} 1))), source: (original (fn {} (params {}) (app {} (var {} debug) (lit {} 1))))} f (lit {} 1))";
        let parsed = chelis_deep::parser::parse_str(source).unwrap();
        let Expr::Node(node, _) = &parsed[0] else {
            panic!("stamped declaration")
        };
        let unknown = Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
            head: "custom".into(),
            meta: node.meta().clone(),
            children: vec![],
            span: parsed[0].span(),
        }));
        for original in [parsed[0].clone(), unknown] {
            let rewritten = annotate_effects(
                &original,
                &BTreeMap::new(),
                &BTreeSet::new(),
                &BTreeMap::new(),
            );
            let metadata = match &rewritten {
                Expr::Node(node, _) => node.meta(),
                Expr::UnknownForm(data) => &data.meta,
                _ => unreachable!(),
            };
            assert_eq!(metadata.extensions(), node.meta().extensions());
            let closure = metadata.property_seed().unwrap().expression();
            let Expr::Node(closure, _) = closure else {
                panic!("closure")
            };
            assert!(
                closure
                    .meta()
                    .values()
                    .any(|v| matches!(v, MetadataValue::Effects(_))),
                "registered expression closure needs inferred effects"
            );
            assert_eq!(
                metadata.source(),
                node.meta().source(),
                "preserved source is data"
            );
        }
    }
}
