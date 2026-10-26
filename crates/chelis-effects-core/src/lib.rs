//! Portable effect analysis for explicit Deep expression slices.
//!
//! Results contain data, not checked-program authority.
//! The caller supplies allocation and sufficient stack space.
//! Host adapters retain checked reconstruction and target admission.
//! Host adapters supply the HM-annotated tree from type inference.
//!
//! Default features are empty. The `serde` feature enables portable model transport.
//! The `test-instrumentation` feature exposes explicit per-analysis counters without host state.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

use alloc::boxed::Box;
use alloc::collections::{BTreeMap, BTreeSet};
use alloc::format;
use alloc::string::{String, ToString};
use alloc::vec;
use alloc::vec::Vec;

use chelis_deep::DeepTag;
use chelis_deep::annotations::{
    EffectMember, EffectSet as AstEffectSet, MetadataValue, ResourceEffect, Spanned,
};
use chelis_deep::ast::{Atom, Expr, Metadata};
use chelis_deep::{ExprCarrier, Span, decode_effect_kind, string_literal};
use chelis_type_core::{Effect, EffectSet};
use chelis_unord::ScopedMap;
use chelis_vocab::EffectKind;
use chelis_vocab::EffectKindDecodeError;

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

/// Infer rows for definitions owned by `expressions`.
///
/// The optional library supplies an immutable outer scope.
/// This function does not validate handlers or declared upper bounds.
pub fn infer_effect_rows(expressions: &[Expr], library: Option<&[Expr]>) -> DefEffectRows {
    let mut rows = Analysis::default()
        .infer_def_effects(expressions, library)
        .0;
    let owned_definitions = top_level_def_bodies(expressions);
    rows.retain(|name, _| owned_definitions.contains_key(name));
    rows
}

/// Annotate and validate only `expressions` against the optional library.
///
/// A successful result does not establish type correctness or source admission.
/// The host must retain its checked annotation-update boundary.
pub fn analyze_effects(
    expressions: &[Expr],
    library: Option<&[Expr]>,
) -> Result<Vec<Expr>, Vec<EffectError>> {
    Analysis::default().analyze(expressions, library)
}

/// Explicit counters for one analysis invocation.
///
/// These counters require no ambient state or host services.
#[cfg(feature = "test-instrumentation")]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EffectWorkProfile {
    pub infer_expr_visits: usize,
    pub annotation_expr_visits: usize,
    pub node_bridge_clone_nodes: usize,
    pub list_rewrite_clone_nodes: usize,
    pub top_level_body_clone_nodes: usize,
    pub metadata_rewrites: usize,
}

/// Analyze with explicit work counters for full-host cost tests.
#[cfg(feature = "test-instrumentation")]
pub fn analyze_effects_with_profile(
    expressions: &[Expr],
    library: Option<&[Expr]>,
) -> (Result<Vec<Expr>, Vec<EffectError>>, EffectWorkProfile) {
    let mut analysis = Analysis::default();
    let result = analysis.analyze(expressions, library);
    (result, analysis.profile)
}

#[derive(Default)]
struct Analysis {
    #[cfg(feature = "test-instrumentation")]
    profile: EffectWorkProfile,
}

impl Analysis {
    fn infer_def_effects(
        &mut self,
        expressions: &[Expr],
        library: Option<&[Expr]>,
    ) -> (DefEffectRows, BTreeSet<String>) {
        let Some(library) = library else {
            return self.infer_program_effects(expressions);
        };
        let (library_effects, library_callables) = self.infer_program_effects(library);
        self.infer_program_effects_with_context(expressions, &library_effects, &library_callables)
    }

    fn analyze(
        &mut self,
        expressions: &[Expr],
        library: Option<&[Expr]>,
    ) -> Result<Vec<Expr>, Vec<EffectError>> {
        let (effects_by_def, top_level_callables) = self.infer_def_effects(expressions, library);
        let annotated_exprs = expressions
            .iter()
            .map(|expr| {
                self.annotate_effects(
                    expr,
                    &effects_by_def,
                    &top_level_callables,
                    &mut Locals::new(),
                )
            })
            .collect::<Vec<_>>();
        let mut errors = Vec::new();
        validate_handlers(&annotated_exprs, &mut errors);
        validate_declared_vs_inferred(&annotated_exprs, &effects_by_def, &mut errors);
        if errors.is_empty() {
            Ok(annotated_exprs)
        } else {
            Err(errors)
        }
    }

    fn infer_program_effects(
        &mut self,
        exprs: &[Expr],
    ) -> (BTreeMap<String, EffectSet>, BTreeSet<String>) {
        self.infer_program_effects_with_context(exprs, &BTreeMap::new(), &BTreeSet::new())
    }

    /// Iterative-fixed-point effect inference that lets new-code defs see an
    /// outer library scope. The library's `effects_by_def` and callable-name
    /// set are merged into the seed maps so `(var libname)` references in new
    /// code resolve to the library's effect row.
    ///
    /// New-code defs shadow library defs of the same name on the inner pass.
    /// The function does NOT mutate the library inputs; merging is local.
    fn infer_program_effects_with_context(
        &mut self,
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
                let inferred = self.infer_expr_effects(
                    body,
                    &effects,
                    &top_level_callables,
                    &mut Locals::new(),
                );
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

    fn infer_expr_effects(
        &mut self,
        expr: &Expr,
        top_level_effects: &BTreeMap<String, EffectSet>,
        top_level_callables: &BTreeSet<String>,
        locals: &mut Locals,
    ) -> EffectSet {
        #[cfg(feature = "test-instrumentation")]
        {
            self.profile.infer_expr_visits += 1;
        }
        match expr.carrier() {
            ExprCarrier::Atom(_) | ExprCarrier::MetadataMap(_) => EffectSet::new(),
            ExprCarrier::MetadataExpression(meta) => {
                self.infer_expr_effects(&meta.expr, top_level_effects, top_level_callables, locals)
            }
            ExprCarrier::DecodedNode(tag, metadata, children) => {
                let handled_effect = (tag == DeepTag::HandleEffect)
                    .then(|| decode_effect_kind(metadata).ok())
                    .flatten();
                self.infer_tagged_effects(
                    tag,
                    children,
                    handled_effect,
                    top_level_effects,
                    top_level_callables,
                    locals,
                )
            }
            ExprCarrier::StructuralList(children)
            | ExprCarrier::UndecodableHead(_, _, children) => self.infer_children_effects(
                children,
                top_level_effects,
                top_level_callables,
                locals,
            ),
        }
    }

    fn infer_children_effects(
        &mut self,
        children: &[Expr],
        top_level_effects: &BTreeMap<String, EffectSet>,
        top_level_callables: &BTreeSet<String>,
        locals: &mut Locals,
    ) -> EffectSet {
        children
            .iter()
            .map(|child| {
                self.infer_expr_effects(child, top_level_effects, top_level_callables, locals)
            })
            .fold(EffectSet::new(), |mut accumulated, effects| {
                accumulated.extend(&effects);
                accumulated
            })
    }

    fn infer_tagged_effects(
        &mut self,
        tag: DeepTag,
        kids: &[Expr],
        handled_effect: Option<EffectKind>,
        top_level_effects: &BTreeMap<String, EffectSet>,
        top_level_callables: &BTreeSet<String>,
        locals: &mut Locals,
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
                                builtin_call_effect(name).map(|effect| {
                                    let mut effects = EffectSet::new();
                                    effects.insert(effect);
                                    effects
                                })
                            })
                    })
                })
                .unwrap_or_default(),
            DeepTag::App => {
                self.infer_app_effects(kids, top_level_effects, top_level_callables, locals)
            }
            DeepTag::Fn => locals.scope(|scope| {
                bind_lexical_names(tag, kids, scope);
                kids.get(1)
                    .map(|body| {
                        self.infer_expr_effects(body, top_level_effects, top_level_callables, scope)
                    })
                    .unwrap_or_default()
            }),
            DeepTag::Arm => locals.scope(|scope| {
                bind_lexical_names(tag, kids, scope);
                self.infer_children_effects(kids, top_level_effects, top_level_callables, scope)
            }),
            DeepTag::Let => {
                self.infer_let_effects(kids, top_level_effects, top_level_callables, locals)
            }
            DeepTag::Grad | DeepTag::Vmap => kids
                .first()
                .map(|child| {
                    self.infer_expr_effects(child, top_level_effects, top_level_callables, locals)
                })
                .unwrap_or_default(),
            DeepTag::HandleEffect => self.infer_handle_effects(
                kids,
                handled_effect,
                top_level_effects,
                top_level_callables,
                locals,
            ),
            _ => self.infer_children_effects(kids, top_level_effects, top_level_callables, locals),
        }
    }

    fn infer_app_effects(
        &mut self,
        kids: &[Expr],
        top_level_effects: &BTreeMap<String, EffectSet>,
        top_level_callables: &BTreeSet<String>,
        locals: &mut Locals,
    ) -> EffectSet {
        let mut effects = EffectSet::new();
        for kid in kids {
            effects.extend(&self.infer_expr_effects(
                kid,
                top_level_effects,
                top_level_callables,
                locals,
            ));
        }

        effects
    }

    fn infer_let_effects(
        &mut self,
        kids: &[Expr],
        top_level_effects: &BTreeMap<String, EffectSet>,
        top_level_callables: &BTreeSet<String>,
        locals: &mut Locals,
    ) -> EffectSet {
        if kids.len() < 2 {
            return EffectSet::new();
        }

        locals.scope(|local_scope| {
            let mut effects = EffectSet::new();

            match kids[0].carrier() {
                ExprCarrier::DecodedNode(DeepTag::Bind, _, bind_kids) => {
                    let mut i = 0;
                    while i + 1 < bind_kids.len() {
                        let value = &bind_kids[i + 1];
                        let value_effects = self.infer_expr_effects(
                            value,
                            top_level_effects,
                            top_level_callables,
                            local_scope,
                        );
                        effects.extend(&value_effects);
                        let binding_effects = if value.tag() == Some(DeepTag::Fn) {
                            value_effects
                        } else {
                            EffectSet::new()
                        };
                        for name in bind_target_names(&bind_kids[i]) {
                            local_scope.insert(name, binding_effects.clone());
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
                    effects.extend(&self.infer_expr_effects(
                        &kids[0],
                        top_level_effects,
                        top_level_callables,
                        local_scope,
                    ));
                }
            }

            effects.extend(&self.infer_expr_effects(
                &kids[1],
                top_level_effects,
                top_level_callables,
                local_scope,
            ));
            effects
        })
    }

    fn infer_handle_effects(
        &mut self,
        kids: &[Expr],
        handled_effect: Option<EffectKind>,
        top_level_effects: &BTreeMap<String, EffectSet>,
        top_level_callables: &BTreeSet<String>,
        locals: &mut Locals,
    ) -> EffectSet {
        if kids.len() != 2 {
            return self.infer_children_effects(
                kids,
                top_level_effects,
                top_level_callables,
                locals,
            );
        }
        let mut effects =
            self.infer_expr_effects(&kids[0], top_level_effects, top_level_callables, locals);
        let body_effects =
            self.infer_expr_effects(&kids[1], top_level_effects, top_level_callables, locals);
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
        &mut self,
        expr: &Expr,
        top_level_effects: &BTreeMap<String, EffectSet>,
        top_level_callables: &BTreeSet<String>,
        locals: &mut Locals,
    ) -> Expr {
        #[cfg(feature = "test-instrumentation")]
        {
            self.profile.annotation_expr_visits += 1;
        }
        match expr {
            Expr::Atom(_, _) => expr.clone(),
            Expr::Map(map, span) => Expr::Map(
                map.map_expressions(&mut |value, _| {
                    self.annotate_effects(value, top_level_effects, top_level_callables, locals)
                })
                .expect("effect annotation preserves metadata payloads"),
                *span,
            ),
            Expr::MetaExpr(meta, span) => Expr::MetaExpr(
                chelis_deep::MetaExpr {
                    expr: Box::new(self.annotate_effects(
                        &meta.expr,
                        top_level_effects,
                        top_level_callables,
                        locals,
                    )),
                    metadata: meta
                        .metadata
                        .map_expressions(&mut |value, _| {
                            self.annotate_effects(
                                value,
                                top_level_effects,
                                top_level_callables,
                                locals,
                            )
                        })
                        .expect("effect annotation preserves metadata payloads"),
                },
                *span,
            ),
            Expr::Node(node, span) => {
                let annotated_children = locals.scope(|scope| {
                    if node.tag() == DeepTag::Let && node.children_slice().len() >= 2 {
                        self.annotate_let_children(
                            node.children_slice(),
                            top_level_effects,
                            top_level_callables,
                            scope,
                        )
                    } else {
                        bind_lexical_names(node.tag(), node.children_slice(), scope);
                        node.children_slice()
                            .iter()
                            .map(|child| {
                                self.annotate_effects(
                                    child,
                                    top_level_effects,
                                    top_level_callables,
                                    scope,
                                )
                            })
                            .collect()
                    }
                });
                let mut meta = node
                    .meta()
                    .map_expressions(&mut |value, _| {
                        self.annotate_effects(value, top_level_effects, top_level_callables, locals)
                    })
                    .expect("effect annotation preserves metadata payloads");
                if node.tag() == DeepTag::Fn {
                    let effects = self.infer_expr_effects(
                        expr,
                        top_level_effects,
                        top_level_callables,
                        locals,
                    );
                    if !effects.is_empty() {
                        #[cfg(feature = "test-instrumentation")]
                        {
                            self.profile.metadata_rewrites += 1;
                        }
                        meta.replace(MetadataValue::Effects(effect_set_metadata(&effects)));
                    }
                }
                Expr::node(node.tag(), meta, annotated_children, *span)
            }
            Expr::BareList(elements, span) => Expr::BareList(
                elements
                    .iter()
                    .map(|child| {
                        self.annotate_effects(child, top_level_effects, top_level_callables, locals)
                    })
                    .collect(),
                *span,
            ),
            Expr::UnknownForm(data) => Expr::UnknownForm(Box::new(chelis_deep::UnknownFormData {
                head: data.head.clone(),
                meta: data
                    .meta
                    .map_expressions(&mut |value, _| {
                        self.annotate_effects(value, top_level_effects, top_level_callables, locals)
                    })
                    .expect("effect annotation preserves metadata payloads"),
                children: data
                    .children
                    .iter()
                    .map(|child| {
                        self.annotate_effects(child, top_level_effects, top_level_callables, locals)
                    })
                    .collect(),
                span: data.span,
            })),
        }
    }

    fn annotate_let_children(
        &mut self,
        kids: &[Expr],
        top_level_effects: &BTreeMap<String, EffectSet>,
        top_level_callables: &BTreeSet<String>,
        local_scope: &mut Locals,
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
                    self.annotate_effects(
                        &kids[0],
                        top_level_effects,
                        top_level_callables,
                        local_scope,
                    ),
                    self.annotate_effects(
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
            annotated_bind_kids.push(self.annotate_effects(
                value,
                top_level_effects,
                top_level_callables,
                local_scope,
            ));
            let value_effects =
                self.infer_expr_effects(value, top_level_effects, top_level_callables, local_scope);
            let binding_effects = if value.tag() == Some(DeepTag::Fn) {
                value_effects
            } else {
                EffectSet::new()
            };
            for name in bind_target_names(&bind_kids[index]) {
                local_scope.insert(name, binding_effects.clone());
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
            self.annotate_effects(
                &kids[1],
                top_level_effects,
                top_level_callables,
                local_scope,
            ),
        ]
    }
}

/// The effect a call to this builtin performs, if any. The effect checker
/// charges it at every call and wherever the builtin is named as a value, as
/// it charges an anonymous function that calls it, so an alias cannot shed
/// it. A random draw is a pure function of its key ([05-RNG-1]): `dropout`
/// and `uniform_like` perform none.
pub fn builtin_call_effect(name: &str) -> Option<Effect> {
    match name {
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
        | "clock_wall_read"
        | "clock_monotonic_read" => Some(Effect::Io),
        "test_assert" | "test_assert_eq" | "test_assert_close_tensor" | "test_assert_eq_tensor" => {
            Some(Effect::Test)
        }
        _ => None,
    }
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

/// The local names in scope where the walk is, each with the latent effects
/// of the value it binds. A binder's names last for one
/// [`ScopedMap::scope`], so entering a binder copies nothing, and a chain of
/// `k` nested `let`s costs `k` rather than `k` squared (chelis#3740).
type Locals = ScopedMap<String, EffectSet>;

/// Bind, in the scope `locals` has open, the names this node binds for its
/// children: a function's parameters and a match arm's pattern variables.
/// The names come from the type checker's own binder reading, so a lexical
/// binder that reuses a builtin's name is the binder here exactly as it is
/// there, and the builtin's call effect does not apply to it ([04-INF-9]).
fn bind_lexical_names(tag: DeepTag, kids: &[Expr], locals: &mut Locals) {
    let names = match (tag, kids.first()) {
        (DeepTag::Fn, Some(params)) => chelis_deep::lexical_param_names(params),
        (DeepTag::Arm, Some(pattern)) => chelis_deep::lexical_pattern_names(pattern),
        _ => Vec::new(),
    };
    for name in names {
        locals.insert(name, EffectSet::new());
    }
}

/// The names a `let` binding target binds: a plain name, or a destructuring
/// pattern's variables.
fn bind_target_names(target: &Expr) -> Vec<String> {
    match symbol_name(target) {
        Some(name) => vec![name.to_string()],
        None => chelis_deep::lexical_pattern_names(target),
    }
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

#[cfg(test)]
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

fn zero_span() -> Span {
    Span::new(0, 0)
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod decode_once_producer_tests;
