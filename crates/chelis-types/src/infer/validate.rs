//! IR validation and type conversion.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::shape_honesty::*;
use super::*;

#[cfg(test)]
thread_local! {
    static CANCEL_BEFORE_DECLARATION_VALIDATION: std::cell::Cell<bool> =
        const { std::cell::Cell::new(false) };
}

#[cfg(test)]
struct DeclarationValidationCancellationHook;

#[cfg(test)]
impl Drop for DeclarationValidationCancellationHook {
    fn drop(&mut self) {
        CANCEL_BEFORE_DECLARATION_VALIDATION.with(|armed| armed.set(false));
    }
}

#[cfg(test)]
fn cancel_before_declaration_validation_for_test() -> DeclarationValidationCancellationHook {
    CANCEL_BEFORE_DECLARATION_VALIDATION.with(|armed| {
        assert!(
            !armed.replace(true),
            "the declaration-validation cancellation hook is already armed"
        );
    });
    DeclarationValidationCancellationHook
}

#[cfg(test)]
fn trip_declaration_validation_cancellation_for_test() {
    CANCEL_BEFORE_DECLARATION_VALIDATION.with(|armed| {
        if armed.replace(false) {
            crate::cancel::current_cancel_token()
                .expect("the declaration-validation hook requires an installed token")
                .cancel();
        }
    });
}

/// The ordered semantic pass protocol shared by every public checker entry.
///
/// PP9 / [04-TOT-5] keeps language-required checks here and leaves backend
/// capability refusals to their owning lowering stages. Callers supply the
/// declaration-type view appropriate to their context; the program carrier is
/// preserved and every reader in this protocol must consume it structurally.
pub(super) fn validate_semantic_program(
    exprs: &[deep::Expr],
    type_env: &IrTypeEnv,
    top_level_references: &TopLevelReferenceGraph,
    errors: &mut DiagnosticSink<'_>,
) {
    top_level_references.report_initialization_errors(errors);
    validate_core_transform_fragment(exprs, errors);
    validate_vmap_extent_dependencies(exprs, type_env, errors);
    let mut static_env = UnordMap::new();
    let shape_env = shape_type_env(type_env);
    let declared_signatures = collect_declared_sig_metadata(top_level_decl_items(exprs));
    // chelis#930: per-top-level-declaration cancellation, same grain as
    // inference. Without it this validator is one uninterruptible step whose
    // cost grows with the program, and interrupt latency is bounded by the
    // longest such step. The caller's `cancellation_gate` rejects the
    // truncated walk.
    #[cfg(test)]
    trip_declaration_validation_cancellation_for_test();
    let cancel = crate::cancel::current_cancel_token();
    for expr in top_level_decl_items(exprs) {
        if cancel.as_ref().is_some_and(CancelToken::is_cancelled) {
            break;
        }
        validate_ir_expr(
            expr,
            &shape_env,
            &mut static_env,
            &declared_signatures,
            errors,
        );
    }
}

/// Launch-core transforms are deliberately a smaller acceptance surface than
/// the timeless transform language contract. A named target must not alias an
/// unshadowed top-level function declaration; local wrapper closures remain
/// on their separately tested path. `grad` keeps its existing
/// direct-inline-lambda path. `vmap` admits inline lambdas only when every
/// parameter is explicitly typed: otherwise the inference order can bind an
/// untyped parameter to the unsliced operand. This keeps the checker from
/// certifying a program whose evaluator could select a different callable or
/// whose inline-vmap parameter could have the wrong rank (#1887, #1952,
/// #1954).
///
/// The walk is lexical rather than type-directed.  A local `loss` with the
/// same function type as a top-level `loss` is exactly the #1954 hazard, so
/// looking only at `Type::Fn` would recreate the silent global fallback.  The
/// normative transformation semantics remain in spec/06; the release
/// supported-fragment document records this temporary admission fence.
fn validate_core_transform_fragment(exprs: &[deep::Expr], errors: &mut DiagnosticSink<'_>) {
    let top_level_functions = collect_top_level_function_names(exprs);
    let module_function_aliases = collect_top_level_function_aliases(exprs, &top_level_functions);
    let lexical_scope = CoreTransformScope::default();
    for expr in top_level_decl_items(exprs) {
        walk_core_transform_targets(
            expr,
            &top_level_functions,
            &module_function_aliases,
            &lexical_scope,
            errors,
        );
    }
}

#[derive(Clone, Default)]
struct CoreTransformScope {
    local_names: UnordSet<String>,
    direct_function_aliases: UnordSet<String>,
}

impl CoreTransformScope {
    fn bind_local(&mut self, name: String, aliases_top_level_function: bool) {
        self.direct_function_aliases.remove(&name);
        if aliases_top_level_function {
            self.direct_function_aliases.insert(name.clone());
        }
        self.local_names.insert(name);
    }
}

fn collect_top_level_function_names(exprs: &[deep::Expr]) -> UnordSet<String> {
    let mut names = UnordSet::new();
    for expr in top_level_decl_items(exprs) {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let Some(name) = children.first().and_then(symbol_name) else {
            continue;
        };
        if children
            .get(1)
            .and_then(stamped_parts)
            .is_some_and(|(tag, _, _)| tag == DeepTag::Fn)
        {
            names.insert(name.to_string());
        }
    }
    names
}

fn collect_top_level_function_aliases(
    exprs: &[deep::Expr],
    top_level_functions: &UnordSet<String>,
) -> UnordSet<String> {
    let mut aliases = UnordSet::new();
    for expr in top_level_decl_items(exprs) {
        let Some((DeepTag::Def, _, children)) = stamped_parts(expr) else {
            continue;
        };
        let (Some(name), Some(value)) = (children.first().and_then(symbol_name), children.get(1))
        else {
            continue;
        };
        if direct_module_function_alias(value, top_level_functions, &aliases) {
            aliases.insert(name.to_string());
        }
    }
    aliases
}

fn walk_core_transform_targets(
    expr: &deep::Expr,
    top_level_functions: &UnordSet<String>,
    module_function_aliases: &UnordSet<String>,
    lexical_scope: &CoreTransformScope,
    errors: &mut DiagnosticSink<'_>,
) {
    stack_guard!("walk_core_transform_targets", expr);
    if let deep::Expr::MetaExpr(meta, _) = expr {
        walk_core_transform_targets(
            &meta.expr,
            top_level_functions,
            module_function_aliases,
            lexical_scope,
            errors,
        );
        return;
    }
    let Some((tag, _, children)) = stamped_parts(expr) else {
        return;
    };
    match tag {
        DeepTag::Def => {
            if let Some(body) = children.get(1) {
                walk_core_transform_targets(
                    body,
                    top_level_functions,
                    module_function_aliases,
                    lexical_scope,
                    errors,
                );
            }
        }
        DeepTag::Fn => {
            let mut scoped = lexical_scope.clone();
            if let Some(params) = children.first()
                && let Some((DeepTag::Params, _, params)) = stamped_parts(params)
            {
                for param in params {
                    if let Some(name) = param_name_for_refs(param) {
                        scoped.bind_local(name, false);
                    }
                }
            }
            if let Some(body) = children.get(1) {
                walk_core_transform_targets(
                    body,
                    top_level_functions,
                    module_function_aliases,
                    &scoped,
                    errors,
                );
            }
        }
        DeepTag::Let => {
            let Some(binding) = children.first() else {
                return;
            };
            let Some((DeepTag::Bind, _, bind_children)) = stamped_parts(binding) else {
                for child in children {
                    walk_core_transform_targets(
                        child,
                        top_level_functions,
                        module_function_aliases,
                        lexical_scope,
                        errors,
                    );
                }
                return;
            };
            if let Some(value) = bind_children.get(1) {
                walk_core_transform_targets(
                    value,
                    top_level_functions,
                    module_function_aliases,
                    lexical_scope,
                    errors,
                );
            }
            let mut scoped = lexical_scope.clone();
            if let Some(name) = bind_children.first().and_then(symbol_name) {
                let aliases_top_level_function = bind_children.get(1).is_some_and(|value| {
                    direct_unshadowed_top_level_function(
                        value,
                        top_level_functions,
                        module_function_aliases,
                        lexical_scope,
                    )
                });
                scoped.bind_local(name.to_string(), aliases_top_level_function);
            }
            if let Some(body) = children.get(1) {
                walk_core_transform_targets(
                    body,
                    top_level_functions,
                    module_function_aliases,
                    &scoped,
                    errors,
                );
            }
        }
        DeepTag::Match => {
            if let Some(scrutinee) = children.first() {
                walk_core_transform_targets(
                    scrutinee,
                    top_level_functions,
                    module_function_aliases,
                    lexical_scope,
                    errors,
                );
            }
            for arm in children.iter().skip(1) {
                let Some((DeepTag::Arm, _, arm_children)) = stamped_parts(arm) else {
                    walk_core_transform_targets(
                        arm,
                        top_level_functions,
                        module_function_aliases,
                        lexical_scope,
                        errors,
                    );
                    continue;
                };
                let mut scoped = lexical_scope.clone();
                if let Some(pattern) = arm_children.first() {
                    for name in pattern_names_for_signature(pattern).to_sorted() {
                        scoped.bind_local(name.clone(), false);
                    }
                }
                for child in arm_children.iter().skip(1) {
                    walk_core_transform_targets(
                        child,
                        top_level_functions,
                        module_function_aliases,
                        &scoped,
                        errors,
                    );
                }
            }
        }
        DeepTag::Grad | DeepTag::Vmap => {
            validate_core_transform_target(
                tag,
                children.first(),
                top_level_functions,
                module_function_aliases,
                lexical_scope,
                errors,
            );
            for child in children {
                walk_core_transform_targets(
                    child,
                    top_level_functions,
                    module_function_aliases,
                    lexical_scope,
                    errors,
                );
            }
        }
        _ => {
            for child in children {
                walk_core_transform_targets(
                    child,
                    top_level_functions,
                    module_function_aliases,
                    lexical_scope,
                    errors,
                );
            }
        }
    }
}

fn direct_module_function_alias(
    expr: &deep::Expr,
    top_level_functions: &UnordSet<String>,
    module_function_aliases: &UnordSet<String>,
) -> bool {
    stamped_parts(expr).is_some_and(|(tag, _, children)| {
        tag == DeepTag::Var
            && children.first().and_then(symbol_name).is_some_and(|name| {
                top_level_functions.contains(name) || module_function_aliases.contains(name)
            })
    })
}

fn direct_unshadowed_top_level_function(
    expr: &deep::Expr,
    top_level_functions: &UnordSet<String>,
    module_function_aliases: &UnordSet<String>,
    lexical_scope: &CoreTransformScope,
) -> bool {
    stamped_parts(expr).is_some_and(|(tag, _, children)| {
        tag == DeepTag::Var
            && children.first().and_then(symbol_name).is_some_and(|name| {
                (top_level_functions.contains(name) && !lexical_scope.local_names.contains(name))
                    || (module_function_aliases.contains(name)
                        && !lexical_scope.local_names.contains(name))
                    || lexical_scope.direct_function_aliases.contains(name)
            })
    })
}

/// An inline `vmap` target is safe on the release fragment when its parameter
/// type is explicit. `infer_vmap` can then transform that type by inserting the
/// mapped axis before the eventual application unifies it with the operand.
/// Without an annotation, the lambda is inferred first and can instead bind to
/// the unsliced operand (#1887).
fn inline_vmap_has_untyped_parameter(target: Option<&deep::Expr>) -> bool {
    let Some((DeepTag::Fn, _, children)) = target.and_then(stamped_parts) else {
        return false;
    };
    let Some((DeepTag::Params, _, params)) = children.first().and_then(stamped_parts) else {
        return false;
    };
    params
        .iter()
        .any(|param| !matches!(param_name_and_inline_type(param), Some((_, Some(_)))))
}

fn validate_core_transform_target(
    tag: DeepTag,
    target: Option<&deep::Expr>,
    top_level_functions: &UnordSet<String>,
    module_function_aliases: &UnordSet<String>,
    lexical_scope: &CoreTransformScope,
    errors: &mut DiagnosticSink<'_>,
) {
    let target_parts = target.and_then(stamped_parts);
    let name = target_parts.and_then(|(target_tag, _, children)| {
        (target_tag == DeepTag::Var)
            .then(|| children.first().and_then(symbol_name))
            .flatten()
    });
    let shadows_top_level = name.is_some_and(|name| {
        top_level_functions.contains(name) && lexical_scope.local_names.contains(name)
    });
    let aliases_top_level_function = name.is_some_and(|name| {
        lexical_scope.direct_function_aliases.contains(name)
            || module_function_aliases.contains(name)
    });

    let requires_fence = match tag {
        // Existing `grad(fn (...) -> ...)` execution is a distinct, covered
        // path. The P1 hazards are aliases and a local binder choosing a
        // same-named global declaration, both represented as `var`.
        DeepTag::Grad => shadows_top_level || aliases_top_level_function,
        // #1887 is specifically an inline lambda whose parameter was inferred
        // from the unsliced operand. Explicit parameter annotations provide
        // the pre-transform function type, so they remain supported.
        // Aliases get the same direct-name fence as `grad`.
        DeepTag::Vmap => {
            inline_vmap_has_untyped_parameter(target)
                || shadows_top_level
                || aliases_top_level_function
        }
        _ => unreachable!("only transform tags call this validator"),
    };
    if !requires_fence {
        return;
    }

    let transform = match tag {
        DeepTag::Grad => "grad",
        DeepTag::Vmap => "vmap",
        _ => unreachable!("only transform tags call this validator"),
    };
    errors.push(CheckError::new(
        CheckErrorKind::TypeMismatch,
        format!(
            "the core transform fragment rejects this `{transform}` target: local aliases \
             and shadowing bindings must be direct, unshadowed top-level function \
             declarations, and inline `vmap` parameters must be explicitly typed \
             (chelis#1887, #1952, #1954)"
        ),
        vec![
            format!("Define a top-level function and write `{transform}(that_function)`."),
            "For an inline `vmap` lambda, give every parameter an explicit type.".to_string(),
            "The rejected callable form is outside the Chelis 0.19 core fragment.".to_string(),
        ],
    ));
}

/// Yield each top-level declaration, flattening through a `(module {} name ...)`
/// wrapper if present. Deep sources produced by Surf `module X` desugaring
/// have every def/defsig/deftype inside this wrapper; without flattening,
/// top-level walkers see a single `(module ...)` and miss everything inside.
/// Extract the name of a top-level decl (`def`, `defsig`, `deftype`,
/// `typealias`) for profile instrumentation. Returns `None` for shapes
/// that don't have a leading symbol.
/// Walk the program's Deep AST and reject any `(t-tensor ... (t-prim {} P))`
/// whose precision P is not supported by the Phase 0f tensor backend
/// (currently: f16, bf16, f64, f8e4m3, string).
///
/// This runs after HM inference so it catches user-written tensor type
/// ascriptions, defsig tensor types, parameter type annotations, literal
/// type metadata, and any cast target that produces a tensor with an
/// unsupported element precision.
pub(super) fn validate_tensor_precisions_in_program(
    exprs: &[deep::Expr],
    errors: &mut impl DiagnosticOutput,
) {
    let mut seen: UnordSet<(String, String)> = UnordSet::new();
    // Descend through `(module {} name ...)` wrappers so per-def dedup
    // keeps each def's tensor types in their own key space (otherwise
    // every def lives under def_context="" and errors collapse).
    for expr in top_level_decl_items(exprs) {
        let def_name = match expr {
            deep::Expr::List(list, _)
                if matches!(
                    get_tag(list),
                    Some(DeepTag::Def)
                        | Some(DeepTag::Defsig)
                        | Some(DeepTag::Deftype)
                        | Some(DeepTag::Typealias)
                ) =>
            {
                children(list)
                    .first()
                    .and_then(symbol_name)
                    .unwrap_or("")
                    .to_string()
            }
            _ => String::new(),
        };
        walk_for_tensor_precision(expr, errors, &mut seen, &def_name);
    }
}

pub(super) fn walk_for_tensor_precision(
    expr: &deep::Expr,
    errors: &mut impl DiagnosticOutput,
    seen: &mut UnordSet<(String, String)>,
    def_context: &str,
) {
    // Bail before this walker's own unbounded recursion exhausts the
    // native stack (gdb confirmed this is a real SIGSEGV site on deep `app`
    // trees, distinct from `infer_expr`). The bail records into
    // `STACK_EXHAUSTED`; the check entry boundary turns that into a single
    // located failure, so we just stop recursing here. See
    // `STACK_RED_ZONE_BYTES`.
    stack_guard!(
        "validate_tensor_precisions (walk_for_tensor_precision)",
        expr
    );
    match expr.carrier() {
        deep::ExprCarrier::DecodedNode(tag, metadata, kids) => {
            // Check t-tensor nodes at this level.
            if tag == DeepTag::TTensor {
                // chelis#1125 PP7 / [04-TOT-5]: read the trailing `t-prim`
                // through the shared carrier-total accessor. The former
                // `Node::to_list` bridge copied children verbatim, so its
                // trailing `t-prim` remained a `Node` and a List-only
                // destructure silently skipped the check (PP7 finding 3).
                if let Some(last) = kids.last()
                    && let Some((DeepTag::TPrim, _, prec_kids)) = stamped_parts(last)
                    && let Some(name) = prec_kids.first().and_then(symbol_name)
                {
                    // The shared Deep resolver owns every §1.1.1 reserved
                    // spelling, including the internal `Prim::F8e4m3` row.
                    // This legacy walker retains only its distinct job:
                    // rejecting an otherwise unknown `t-prim` precision from
                    // value-position metadata. The reserved-name exclusions
                    // keep those spellings from acquiring a second owner.
                    if Prim::parse_name(name).is_none()
                        && !crate::deep_type::is_retired_integer_dtype_name(name)
                        && !is_unsigned_dtype_name(name)
                        && !is_deferred_dtype_name(name)
                        && seen.insert((def_context.to_string(), name.to_string()))
                    {
                        let active_set = "f32, f64, bf16, f16, bool, i8, i16, i32, i64";
                        // WS-A5 RT-3a F3: an identifier in a `t-prim`
                        // precision slot that is neither a known active
                        // primitive nor a §1.1.1 deferred dtype name
                        // (unsigned alias or reserved name) is an
                        // unbound name. Inside a sig the desugarer emits
                        // such an identifier as `t-var`, so reaching this
                        // arm with `t-prim` proves the name appears in a
                        // value-position annotation (let binding, def
                        // param without a surrounding sig that quantified
                        // it) where the closed primitive set must apply.
                        // Without this guard the name silently collapses
                        // to a witnessed resolution failure at the centralized
                        // Deep type boundary's
                        // `Prim::parse_name` fall-through and the
                        // permissive unify rule absorbs the mismatch.
                        errors.push(CheckError::new(
                            CheckErrorKind::UnsupportedTensorPrecision,
                            format!(
                                "tensor element precision `{name}` is not a recognized \
                                 primitive (active set: {active_set}); inside a sig an \
                                 unbound lowercase name introduces a precision tvar per \
                                 spec/04-type-system.md §5.8, but in this position the \
                                 closed primitive set applies",
                            ),
                            vec![format!(
                                "Use one of {active_set}, or move the annotation into a \
                                 `sig` declaration that quantifies `{name}` as a precision \
                                 type variable",
                            )],
                        ));
                    }
                }
            }

            // `cast` is inferred by infer_cast which already emits a clearer
            // site-local error for bad precisions. Skip the walker's recursion
            // inside a cast so we don't duplicate the diagnostic.
            if tag == DeepTag::Cast {
                return;
            }

            // Metadata may carry `type:` ascriptions containing tensor types.
            metadata.visit_syntax(&mut |_, value| {
                walk_for_tensor_precision(value, errors, seen, def_context);
            });

            for child in kids {
                walk_for_tensor_precision(child, errors, seen, def_context);
            }
        }
        deep::ExprCarrier::MetadataMap(metadata) => {
            metadata.visit_syntax(&mut |_, value| {
                walk_for_tensor_precision(value, errors, seen, def_context);
            });
        }
        deep::ExprCarrier::MetadataExpression(metadata_expr) => {
            metadata_expr.metadata.visit_syntax(&mut |_, value| {
                walk_for_tensor_precision(value, errors, seen, def_context);
            });
            walk_for_tensor_precision(&metadata_expr.expr, errors, seen, def_context);
        }
        deep::ExprCarrier::Atom(_) => {}
        deep::ExprCarrier::StructuralList(elements) => {
            for child in elements {
                walk_for_tensor_precision(child, errors, seen, def_context);
            }
        }
        deep::ExprCarrier::UndecodableHead(_, metadata, children) => {
            metadata.visit_syntax(&mut |_, value| {
                walk_for_tensor_precision(value, errors, seen, def_context);
            });
            for child in children {
                walk_for_tensor_precision(child, errors, seen, def_context);
            }
        }
        deep::ExprCarrier::MalformedLegacyList(list) => {
            for element in &list.elements {
                walk_for_tensor_precision(element, errors, seen, def_context);
            }
        }
    }
}

/// Build a name → declared-type-expr map for a def's body params, by
/// pairing each param's name with the corresponding sig parameter slot.
pub(super) fn build_def_param_scope(
    expr: &deep::Expr,
    sigs: &IrTypeEnv,
) -> BTreeMap<String, deep::Expr> {
    let mut scope = BTreeMap::new();
    // chelis#1107: preserve stamped and ordinary carriers alike. A List-only
    // destructure loses every stamped definition's parameter scope.
    let Some((tag, _, kids)) = stamped_parts(expr) else {
        return scope;
    };
    if tag != DeepTag::Def {
        return scope;
    }
    let Some(name) = kids.first().and_then(symbol_name) else {
        return scope;
    };
    let Some(body_expr) = kids.get(1) else {
        return scope;
    };
    let Some((param_names, _)) = extract_fn_params_and_body(body_expr) else {
        return scope;
    };
    // Prefer the def's surrounding sig if any; fall back to inline
    // param-type annotations on the params themselves (def shape:
    // `def f(x: tensor[3, f32]) = ...`).
    if let Some(sig) = sigs.get(name)
        && let Some((sig_params, _)) = parse_t_fn_parts(sig)
    {
        for (pname, sig_param) in param_names.iter().zip(sig_params.iter()) {
            scope.insert(pname.clone(), sig_param.clone());
        }
        return scope;
    }
    // Fall back: inline param-type annotations.
    let Some((_, _, fn_kids)) = stamped_parts(body_expr) else {
        return scope;
    };
    let Some(params_expr) = fn_kids.first() else {
        return scope;
    };
    let Some((params_tag, _, params)) = stamped_parts(params_expr) else {
        return scope;
    };
    if params_tag != DeepTag::Params {
        return scope;
    }
    for param in params {
        if let Some((pname, Some(ty_expr))) = param_name_and_inline_type(param) {
            scope.insert(pname, ty_expr);
        }
    }
    scope
}

/// The `[name, {type: T}]` element sequence shared by the two carriers an
/// inline-annotated param can arrive in: a tagless `Expr::List` on the
/// serialized-IR ingress and an `Expr::BareList` on the stamped one.
fn inline_param_parts(elements: &[deep::Expr]) -> Option<(String, Option<deep::Expr>)> {
    let Some(deep::Expr::Atom(deep::Atom::Name(name), _)) = elements.first() else {
        return None;
    };
    let ty = match elements.get(1) {
        Some(deep::Expr::Map(meta, _)) => meta.ty().map(|v| v.expression().clone()),
        _ => None,
    };
    Some((name.clone(), ty))
}

/// Extract `(name {type: T} ...)` shape's name+type from a single param
/// expression. Returns `None` for plain `(name {})` shape (no inline
/// type) — the surrounding sig fills those in.
pub(super) fn param_name_and_inline_type(
    param: &deep::Expr,
) -> Option<(String, Option<deep::Expr>)> {
    match param {
        deep::Expr::Atom(deep::Atom::Name(name), _) => Some((name.clone(), None)),
        deep::Expr::List(list, _) => inline_param_parts(&list.elements),
        // chelis#1125 PP7 finding 1 / [04-TOT-5]: an inline-annotated param
        // `(x {type: T})` is a TAGLESS list, so the stamp pass produces an
        // `Expr::BareList`, not an `Expr::Node`. `build_def_param_scope` was
        // migrated to `stamped_parts` by chelis#1126 and stayed inert anyway,
        // because `stamped_parts` reads `Node` and `List` and returns `None`
        // for `BareList` and this last step was `List`-only. With no param
        // scope, the WS-A8 cross-row pass could not resolve a body `(var x)`,
        // so a def with inline param types and no `defsig` skipped the
        // chelis#724 capability rejection on the stamped ingress.
        deep::Expr::BareList(elements, _) => inline_param_parts(elements),
        deep::Expr::MetaExpr(meta, _) => {
            let deep::Expr::Atom(deep::Atom::Name(name), _) = meta.expr.as_ref() else {
                return None;
            };
            let ty = meta.metadata.ty().map(|v| v.expression().clone());
            Some((name.clone(), ty))
        }
        _ => None,
    }
}

pub(super) fn extract_fn_params_and_body(expr: &deep::Expr) -> Option<(Vec<String>, deep::Expr)> {
    let (DeepTag::Fn, _, kids) = stamped_parts(expr)? else {
        return None;
    };
    let params_expr = kids.first()?;
    let body_expr = kids.get(1)?;
    let params = match params_expr {
        deep::Expr::Node(node, _) if node.tag() == DeepTag::Params => node.children_slice(),
        deep::Expr::List(params_list, _) if get_tag(params_list) == Some(DeepTag::Params) => {
            children(params_list)
        }
        deep::Expr::BareList(elements, _) => elements.as_slice(),
        _ => return None,
    };
    let mut names = Vec::new();
    for param in params {
        if let Some(name) = param_name_for_refs(param) {
            names.push(name);
        }
    }
    Some((names, body_expr.clone()))
}

pub(super) fn parse_t_fn_parts(expr: &deep::Expr) -> Option<(Vec<deep::Expr>, deep::Expr)> {
    // chelis#1107: carrier-preserving read. Every reader in this WS-A8 support
    // cluster took the stamped carrier only after `collect_defsig_exprs` was
    // fixed to see it; a `List`-only destructure here would have left the pass
    // half-live.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag != DeepTag::TFn {
        return None;
    }
    let (ret, args) = kids.split_last()?;
    Some((args.iter().map(|e| (*e).clone()).collect(), (*ret).clone()))
}

pub(super) fn validate_ir_expr(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
    static_env: &mut UnordMap<String, StaticValue>,
    declared_signatures: &UnordMap<String, DeclaredSigMetadata>,
    errors: &mut DiagnosticSink<'_>,
) -> StaticValue {
    stack_guard!("validate_ir_expr", expr, StaticValue::Unknown);
    match expr {
        deep::Expr::List(list, _) => {
            if get_tag(list) == Some(DeepTag::Module) {
                for elem in list.elements.iter().skip(3) {
                    validate_ir_expr(elem, type_env, static_env, declared_signatures, errors);
                }
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some(DeepTag::Def) {
                let kids = children(list);
                let Some(name) = kids.first().and_then(symbol_name) else {
                    return StaticValue::Unknown;
                };
                let Some(value_expr) = kids.get(1) else {
                    return StaticValue::Unknown;
                };
                let signature_env = declared_signatures.get(name).map(|signature| {
                    extend_ir_env_with_declared_fn_params(
                        value_expr,
                        &signature.param_types,
                        type_env,
                    )
                });
                let value = validate_ir_expr(
                    value_expr,
                    signature_env.as_ref().unwrap_or(type_env),
                    static_env,
                    declared_signatures,
                    errors,
                );
                static_env.insert(name.to_string(), value);
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some(DeepTag::Fn) {
                let scoped_env = extend_ir_env_with_fn_params(list, type_env);
                let mut scoped_static_env = static_env.clone();
                bind_fn_params_unknown(list, &mut scoped_static_env);
                for elem in &list.elements {
                    validate_ir_expr(
                        elem,
                        &scoped_env,
                        &mut scoped_static_env,
                        declared_signatures,
                        errors,
                    );
                }
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some(DeepTag::Arm) {
                let kids = children(list);
                let mut scoped_static_env = static_env.clone();
                if let Some(pattern) = kids.first() {
                    for name in chelis_deep::pattern_binder_names(pattern) {
                        scoped_static_env.insert(name, StaticValue::Unknown);
                    }
                }
                // The pattern is binding syntax. Only the optional guard and
                // body execute in the arm's lexical scope.
                for elem in kids.iter().skip(1) {
                    validate_ir_expr(
                        elem,
                        type_env,
                        &mut scoped_static_env,
                        declared_signatures,
                        errors,
                    );
                }
                return StaticValue::Unknown;
            }
            if get_tag(list) == Some(DeepTag::Let) {
                let kids = children(list);
                let mut scoped_static_env = static_env.clone();
                // Clone the type env on let-scope entry so each binding's
                // derivable IR-shape-sensitive type (e.g. conv's output
                // dims) can extend the env visible to the let body. Without
                // this the validator cannot resolve `(var y)` for a let-
                // bound `y = conv(...)` and silently rejects the next
                // shape-sensitive call that consumes `y` (RT-205 F5).
                let mut scoped_type_env = type_env.clone();
                if let Some(bind_expr) = kids.first()
                    && let Some((DeepTag::Bind, _, bind_children)) = stamped_parts(bind_expr)
                {
                    let mut index = 0;
                    while index + 1 < bind_children.len() {
                        if let Some(name) = symbol_name(&bind_children[index]) {
                            let value_expr = &bind_children[index + 1];
                            let value = validate_ir_expr(
                                value_expr,
                                &scoped_type_env,
                                &mut scoped_static_env,
                                declared_signatures,
                                errors,
                            );
                            scoped_static_env.insert(name.to_string(), value);
                            record_let_binding_shape_fact(
                                name,
                                value_expr,
                                &mut scoped_type_env,
                                &scoped_static_env,
                            );
                        }
                        index += 2;
                    }
                }
                if let Some(body) = kids.get(1) {
                    return validate_ir_expr(
                        body,
                        &scoped_type_env,
                        &mut scoped_static_env,
                        declared_signatures,
                        errors,
                    );
                }
                return StaticValue::Unknown;
            }
            if let Some(tag) = get_tag(list) {
                // `par` (sequential v1, spec/03-deep-syntax.md §2.3) and `jit`
                // (compilation trigger, §2.7) are spec-blessed pass-through
                // forms at Phase 0 evaluation. The validator used to reject
                // both; the rejection is removed because lowering handles them
                // (see `lower_par` and the `jit` lowering arm).
                if tag == DeepTag::App
                    && let Some(func_name) = active_ir_builtin_name(list, static_env)
                    && is_ir_shape_sensitive_builtin(func_name)
                {
                    validate_ir_builtin_semantic_requirements(list, func_name, type_env, errors);
                }
            }

            if get_tag(list) == Some(DeepTag::Var)
                && let Some(name) = children(list).first().and_then(symbol_name)
            {
                return if name == "Nil" {
                    StaticValue::List(Vec::new())
                } else {
                    static_env
                        .get(name)
                        .cloned()
                        .unwrap_or(StaticValue::Unknown)
                };
            }
            if get_tag(list) == Some(DeepTag::Lit) {
                return literal_static_value(expr);
            }
            if get_tag(list) == Some(DeepTag::Cast) {
                let kids = children(list);
                return kids
                    .first()
                    .map(|inner| {
                        validate_ir_expr(inner, type_env, static_env, declared_signatures, errors)
                    })
                    .unwrap_or(StaticValue::Unknown);
            }
            if get_tag(list) == Some(DeepTag::App) {
                let kids = children(list);
                let func_name = kids
                    .first()
                    .and_then(app_builtin_name)
                    .filter(|name| compiler_name_is_active(name, static_env));
                let arg_values = kids
                    .iter()
                    .skip(1)
                    .map(|arg| {
                        validate_ir_expr(arg, type_env, static_env, declared_signatures, errors)
                    })
                    .collect::<Vec<_>>();
                if func_name == Some("Cons") && arg_values.len() == 2 {
                    if let StaticValue::List(mut tail) = arg_values[1].clone() {
                        tail.insert(0, arg_values[0].clone());
                        return StaticValue::List(tail);
                    }
                    return StaticValue::Unknown;
                }
                if let Some(name) = func_name {
                    return validate_static_builtin_application(name, &arg_values, expr, errors);
                }
                return StaticValue::Unknown;
            }

            for elem in &list.elements {
                validate_ir_expr(elem, type_env, static_env, declared_signatures, errors);
            }
            StaticValue::Unknown
        }
        deep::Expr::Map(map, _) => {
            map.visit_syntax(&mut |_, value| {
                validate_ir_expr(value, type_env, static_env, declared_signatures, errors);
            });
            StaticValue::Unknown
        }
        deep::Expr::MetaExpr(meta, _) => {
            meta.metadata.visit_syntax(&mut |_, value| {
                validate_ir_expr(value, type_env, static_env, declared_signatures, errors);
            });
            validate_ir_expr(
                &meta.expr,
                type_env,
                static_env,
                declared_signatures,
                errors,
            )
        }
        deep::Expr::Atom(_, _) => literal_static_value(expr),
        // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
        deep::Expr::Node(node, span) => {
            let bridged = deep::Expr::List(node.to_list(*span), *span);
            validate_ir_expr(&bridged, type_env, static_env, declared_signatures, errors)
        }
        deep::Expr::BareList(elems, _) => {
            let mut last = StaticValue::Unknown;
            for child in elems {
                last = validate_ir_expr(child, type_env, static_env, declared_signatures, errors);
            }
            last
        }
        deep::Expr::UnknownForm(data) => {
            for child in &data.children {
                validate_ir_expr(child, type_env, static_env, declared_signatures, errors);
            }
            StaticValue::Unknown
        }
    }
}

/// Encode a checker-owned type as canonical Deep type metadata.
///
/// Lowering consumers use this boundary when they need the checker's
/// alias-resolved ADT field types without reparsing authored declarations.
pub fn type_to_deep_expr(ty: &Type) -> deep::Expr {
    type_to_deep_expr_with(ty, stamped_node_expr)
}

pub(super) fn type_to_legacy_deep_expr(ty: &Type) -> deep::Expr {
    type_to_deep_expr_with(ty, node_expr)
}

type NodeBuilder = fn(DeepTag, Vec<deep::Expr>) -> deep::Expr;

fn type_to_deep_expr_with(ty: &Type, make_node: NodeBuilder) -> deep::Expr {
    match ty {
        Type::Prim(prim) => make_node(DeepTag::TPrim, vec![symbol_expr(prim.name())]),
        Type::Fn(args, ret) => {
            let mut children: Vec<deep::Expr> = args
                .iter()
                .map(|arg| type_to_deep_expr_with(arg, make_node))
                .collect();
            children.push(type_to_deep_expr_with(ret, make_node));
            make_node(DeepTag::TFn, children)
        }
        Type::Ref(inner) => make_node(
            DeepTag::TRef,
            vec![type_to_deep_expr_with(inner, make_node)],
        ),
        Type::Tensor(dims, prec) => {
            let mut children: Vec<deep::Expr> = dims
                .iter()
                .map(|dim| dim_to_deep_expr_with(dim, make_node))
                .collect();
            children.push(match prec {
                TensorPrec::Concrete(p) => type_to_deep_expr_with(&Type::Prim(*p), make_node),
                TensorPrec::Var(v) => {
                    make_node(DeepTag::TVar, vec![symbol_expr(&format!("t{}", v.0))])
                }
            });
            make_node(DeepTag::TTensor, children)
        }
        Type::Adt(name, args) => {
            let mut children = vec![symbol_expr(name)];
            children.extend(
                args.iter()
                    .map(|arg| type_to_deep_expr_with(arg, make_node)),
            );
            make_node(DeepTag::TAdt, children)
        }
        Type::KindedAdt(name, args) => {
            let mut children = vec![symbol_expr(name)];
            children.extend(args.iter().map(|argument| match argument {
                NominalArg::Type(ty) => type_to_deep_expr_with(ty, make_node),
                NominalArg::Dimension(dim) => dim_to_deep_expr_with(dim, make_node),
            }));
            make_node(DeepTag::TAdt, children)
        }
        Type::Var(var) => make_node(DeepTag::TVar, vec![symbol_expr(&format!("t{}", var.0))]),
        Type::Tuple(types) => make_node(
            DeepTag::TTuple,
            types
                .iter()
                .map(|ty| type_to_deep_expr_with(ty, make_node))
                .collect(),
        ),
        Type::Unit => make_node(DeepTag::TUnit, vec![]),
        Type::Error(_) => make_node(DeepTag::TVar, vec![symbol_expr("_")]),
    }
}

fn dim_to_deep_expr_with(dim: &Dim, make_node: NodeBuilder) -> deep::Expr {
    match dim {
        Dim::Name(name) => make_node(DeepTag::DName, vec![symbol_expr(name)]),
        Dim::Var(var) => make_node(DeepTag::DVar, vec![symbol_expr(&format!("d{}", var.0))]),
        Dim::Lit(value) => make_node(
            DeepTag::DLit,
            vec![deep::Expr::Atom(deep::Atom::Int(*value), zero_span())],
        ),
        Dim::Wildcard => make_node(DeepTag::DName, vec![symbol_expr("*")]),
        Dim::Rank(rank) => make_node(DeepTag::DRank, vec![symbol_expr(&format!("r{}", rank.0))]),
    }
}

pub(super) fn node_expr(tag: DeepTag, children: Vec<deep::Expr>) -> deep::Expr {
    let mut elements = vec![
        deep::Expr::Atom(deep::Atom::Tag(tag), zero_span()),
        deep::Expr::Map(deep::Metadata::default(), zero_span()),
    ];
    elements.extend(children);
    deep::Expr::List(deep::List { elements }, zero_span())
}

pub(super) fn stamped_node_expr(tag: DeepTag, children: Vec<deep::Expr>) -> deep::Expr {
    deep::Expr::node(tag, deep::Metadata::default(), children, zero_span())
}

pub(super) fn symbol_expr(name: &str) -> deep::Expr {
    deep::Expr::Atom(deep::Atom::Name(name.to_string()), zero_span())
}

pub(super) fn zero_span() -> Span {
    Span::new(0, 0)
}

pub(super) fn span_of_expr(expr: &deep::Expr) -> Span {
    match expr {
        deep::Expr::Atom(_, span)
        | deep::Expr::List(_, span)
        | deep::Expr::Map(_, span)
        | deep::Expr::MetaExpr(_, span)
        | deep::Expr::Node(_, span)
        | deep::Expr::BareList(_, span) => *span,
        deep::Expr::UnknownForm(data) => data.span,
    }
}

pub(super) fn span_of_list(list: &deep::List) -> Span {
    list.elements
        .first()
        .map(span_of_expr)
        .unwrap_or_else(zero_span)
}

pub(super) fn ir_builtin_name(list: &deep::List) -> Option<&str> {
    ir_builtin_name_of_expr(list.elements.get(2)?)
}

/// Preserve ordinary lexical precedence when this post-inference validator
/// selects a compiler-owned route. `static_env` is already the validator's
/// scoped binding environment: function parameters, sequential let binders,
/// and pattern binders are inserted before their bodies are visited.
pub(super) fn active_ir_builtin_name<'a>(
    list: &'a deep::List,
    static_env: &UnordMap<String, StaticValue>,
) -> Option<&'a str> {
    ir_builtin_name(list).filter(|name| compiler_name_is_active(name, static_env))
}

pub(super) fn compiler_name_is_active(
    name: &str,
    static_env: &UnordMap<String, StaticValue>,
) -> bool {
    !builtins::BUILTIN_NAMES.contains(&name) || !static_env.contains_key(name)
}

/// The builtin callee name of an `app`'s callee child, on either carrier.
///
/// chelis#1107 amendment: `validate_ir_expr` bridges a stamped `Expr::Node`
/// one level (`Node::to_list`), so the callee child it hands on is still an
/// `Expr::Node`. The previous `List`-only read returned `None` for every
/// stamped callee, which silently disabled the shape-sensitivity and
/// output-type derivation below on the stamped carrier.
pub(super) fn ir_builtin_name_of_expr(func_expr: &deep::Expr) -> Option<&str> {
    let (DeepTag::Var, _, kids) = stamped_parts(func_expr)? else {
        return None;
    };
    match kids.first() {
        Some(deep::Expr::Atom(deep::Atom::Name(name), _)) => Some(name.as_str()),
        _ => None,
    }
}

/// Borrow `expr` as a `deep::List`, materializing a one-level bridge for a
/// stamped `Expr::Node` into `storage`.
///
/// chelis#1107 amendment: the `derive_*` output-type family threads
/// `&deep::List` through several helpers. Rather than change all of their
/// signatures, the entry points bridge once here; every leaf reader they call
/// (`tensor_precision_expr`, `ir_builtin_name`, …) is carrier-agnostic, so one
/// level is enough.
pub(super) fn as_list<'a>(
    expr: &'a deep::Expr,
    storage: &'a mut Option<deep::List>,
) -> Option<&'a deep::List> {
    match expr {
        deep::Expr::List(list, _) => Some(list),
        deep::Expr::Node(node, span) => Some(storage.insert(node.to_list(*span))),
        _ => None,
    }
}

pub(super) fn is_ir_shape_sensitive_builtin(name: &str) -> bool {
    matches!(
        name,
        "matmul"
            | "softmax"
            | "mean"
            | "layer_norm"
            | "conv"
            | "sum"
            | "count"
            | "max_reduce"
            | "min_reduce"
            | "prod_reduce"
            | "argmax_reduce"
            | "argmin_reduce"
            | "reshape"
            | "permute"
            | "expand"
            | "insert"
            | "pad"
            | "shrink"
            | "stride"
    )
}

pub(super) fn expr_type_expr(expr: &deep::Expr, type_env: &IrTypeEnv) -> Option<deep::Expr> {
    stack_guard!("expr_type_expr", expr, None);
    match expr {
        deep::Expr::Node(node, _) => {
            if let Some(ty) = node.meta().ty() {
                return Some(ty.expression().clone());
            }
            if node.tag() == DeepTag::Var
                && let Some(name) = node.children_slice().first().and_then(symbol_name)
            {
                return type_env.get(name).cloned();
            }
            None
        }
        deep::Expr::List(list, _) => {
            if let Some(meta) = get_meta(list)
                && let Some(ty) = meta.ty()
            {
                return Some(ty.expression().clone());
            }
            if get_tag(list) == Some(DeepTag::Var)
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

pub(super) fn extend_ir_env_with_fn_params(
    fn_list: &deep::List,
    type_env: &ShapeTypeEnv,
) -> ShapeTypeEnv {
    let mut scoped = type_env.clone();
    let Some(params_expr) = children(fn_list).first() else {
        return scoped;
    };
    // chelis#1107 amendment: carrier-preserving read. `validate_ir_expr`
    // bridges only the `fn` node, so `(params {} ...)` arrives as `Expr::Node`.
    let Some((DeepTag::Params, _, param_entries)) = stamped_parts(params_expr) else {
        return scoped;
    };
    for param in param_entries {
        // An inline-annotated entry `(x {type: T})` is symbol-headed, so the
        // stamp pass carries it as `Expr::BareList`, never a `Node` -- this
        // one needs its own arm rather than `stamped_parts`.
        // chelis#1107 amendment (justified-safe, not routed): this match
        // handles every carrier a params entry can take -- `List` (legacy) and
        // `BareList` (stamped, symbol-headed) -- so there is no fall-through.
        let (name, meta) = match param {
            deep::Expr::List(param_list, _) => {
                let Some(name) = param_list.elements.first().and_then(symbol_name) else {
                    continue;
                };
                let Some(meta) = get_meta(param_list) else {
                    continue;
                };
                (name, meta)
            }
            deep::Expr::BareList(elems, _) => {
                let Some(name) = elems.first().and_then(symbol_name) else {
                    continue;
                };
                let Some(deep::Expr::Map(meta, _)) = elems.get(1) else {
                    continue;
                };
                (name, meta)
            }
            _ => continue,
        };
        let Some(ty) = meta.ty() else {
            continue;
        };
        // An inline parameter whose generated `defsig` owns the real type
        // carries a whole-slot hole here. The enclosing `Def` arm has already
        // installed that signature type in this scope; replacing it with `_`
        // would erase concrete shape evidence before validation. A real
        // independently authored annotation still overrides as before.
        if is_wildcard_tvar_expr(ty.expression()) {
            continue;
        }
        scoped.insert(name.to_string(), ty.expression().clone());
    }
    scoped
}

pub(super) fn validate_ir_builtin_semantic_requirements(
    list: &deep::List,
    func_name: &str,
    type_env: &ShapeTypeEnv,
    errors: &mut DiagnosticSink<'_>,
) {
    if func_name == "conv" {
        validate_conv_semantic_requirements(list, type_env, errors);
    }
}

/// Build a `CheckError` for a validator-arm diagnostic that
/// references a specific call site. Appends the call site's `:span`
/// metadata identifier (if present) to the message so JSON consumers
/// can locate the offending expression in the source.
///
/// All shape-sensitive validator errors flow through this helper so
/// they uniformly get DimensionMismatch-grade severity and span
/// suffixes, matching the inference-layer DimensionMismatch surface
/// that JSON tooling already understands (RT-205 F6).
pub(super) fn validator_error(
    kind: CheckErrorKind,
    call_site: &deep::List,
    message: String,
    suggestions: Vec<String>,
) -> CheckError {
    let suffixed = match validator_span_suffix(call_site) {
        Some(span) => format!("{message} {span}"),
        None => message,
    };
    CheckError::new(kind, suffixed, suggestions)
}

/// Render the call site's source span as a parenthesized suffix
/// (e.g. ` (at surf:144..165)`). Returns `None` when the call site
/// carries no `:span` metadata so the unmodified message is used.
pub(super) fn validator_span_suffix(call_site: &deep::List) -> Option<String> {
    let meta = get_meta(call_site)?;
    meta.span_id().map(|v| format!("(at {})", v.value()))
}

/// Extract the `:span` metadata string from a list node, if present.
/// Used to propagate external span identifiers into check diagnostics.
pub(super) fn list_span_id(list: &deep::List) -> Option<&str> {
    let meta = get_meta(list)?;
    meta.span_id().map(|v| v.value())
}

/// Parse the start byte offset from a span identifier string.
/// Handles the `"surf:<start>..<end>"` format emitted by the desugar step
/// and bare `"<start>..<end>"` ranges.
pub(super) fn parse_span_offset(span_id: &str) -> Option<usize> {
    // Format: "surf:10..25" or "10..25" or "octant:line:7" (opaque)
    let numeric_part = span_id
        .rfind(':')
        .map(|i| &span_id[i + 1..])
        .unwrap_or(span_id);
    // Try to parse "start..end"
    numeric_part
        .split_once("..")
        .and_then(|(start, _)| start.parse::<usize>().ok())
}

/// [05-OP-51] check-time obligations for canonical convolution.
///
/// Literal-provable stride, padding, kernel, fit, and arithmetic failures are
/// type errors. Symbolic tensor extents and runtime stride/padding metadata are
/// legal language inputs and retain runtime guards; backend capability limits
/// are not checker signatures.
pub(super) fn validate_conv_semantic_requirements(
    list: &deep::List,
    type_env: &ShapeTypeEnv,
    errors: &mut DiagnosticSink<'_>,
) {
    let [_, _, _, input, kernel, strides, padding] = list.elements.as_slice() else {
        return;
    };
    let input_dims =
        arg_tensor_type_expr(input, type_env).and_then(|ty| tensor_dims_from_type_expr(&ty));
    let kernel_dims =
        arg_tensor_type_expr(kernel, type_env).and_then(|ty| tensor_dims_from_type_expr(&ty));
    let spatial_rank = input_dims
        .as_ref()
        .or(kernel_dims.as_ref())
        .and_then(|dims| dims.len().checked_sub(2))
        .filter(|rank| *rank > 0)
        .or_else(|| {
            collect_shape_list_elements(strides)
                .map(|entries| entries.len())
                .or_else(|| collect_shape_list_elements(padding).map(|entries| entries.len()))
        });
    let Some(spatial_rank) = spatial_rank else {
        return;
    };
    let params = match conv_parameters(strides, padding, spatial_rank) {
        Ok(Some(params)) => params,
        Ok(None) => return,
        Err(message) => {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                message,
                vec![],
            ));
            return;
        }
    };
    let (Some(input_dims), Some(kernel_dims)) = (input_dims, kernel_dims) else {
        return;
    };
    if input_dims.len() < 3 || input_dims.len() != kernel_dims.len() {
        return;
    }
    for (axis, ((input, kernel), (stride, low, high))) in input_dims[2..]
        .iter()
        .zip(&kernel_dims[2..])
        .zip(params)
        .enumerate()
    {
        if matches!(kernel, DeepDimKind::Lit(value) if *value <= 0) {
            errors.push(validator_error(
                CheckErrorKind::DimensionMismatch,
                list,
                format!("conv requires positive kernel extents at spatial axis {axis}"),
                vec![],
            ));
            return;
        }
        let (DeepDimKind::Lit(input), DeepDimKind::Lit(kernel)) = (input, kernel) else {
            continue;
        };
        if conv_output_extent(*input, *kernel, stride, low, high).is_none() {
            errors.push(validator_error(CheckErrorKind::DimensionMismatch, list,
                match input.checked_add(low).and_then(|n| n.checked_add(high)) {
                    Some(padded) if *kernel > padded => format!("IR builtin `conv` output spatial axis {axis}: kernel must fit the padded input"),
                    _ => format!("IR builtin `conv` output spatial axis {axis} arithmetic overflows i64"),
                }, vec![]));
            return;
        }
    }
}

/// [05-OP-51]: positive kernel, nonnegative padding, positive stride, and
/// checked i64 arithmetic. A kernel that does not fit is never an empty
/// result or a truncating-division special case.
pub(super) fn conv_output_extent(
    input: i64,
    kernel: i64,
    stride: i64,
    low: i64,
    high: i64,
) -> Option<i64> {
    if input < 0 || kernel <= 0 || stride <= 0 || low < 0 || high < 0 {
        return None;
    }
    let padded = input.checked_add(low)?.checked_add(high)?;
    if kernel > padded {
        return None;
    }
    padded
        .checked_sub(kernel)?
        .checked_div(stride)?
        .checked_add(1)
}

/// If `expr` is a recognizable shape-sensitive IR builtin call whose
/// output tensor type can be derived from its argument types and
/// literal scalar args, return that type as a Deep `(t-tensor ...)`
/// expression. Used to extend the validator's per-let-scope type env
/// so downstream uses of a let-bound name resolve to a concrete
/// tensor type (RT-205 F5).
///
/// In addition to `conv` direct calls, this also handles
/// shape-PRESERVING unary and binary point-wise ops (relu, tanh,
/// add, mul, etc.) so the canonical CNN layer pattern
/// `y = relu(conv(...))` chains correctly into a downstream
/// `conv(&y, ...)` (RT-205 round-2 F2). Reductions and most movement
/// ops are intentionally NOT handled here; they need separate per-op
/// derivation because they change rank or shape. `stride`, `expand`, and
/// `insert` are deliberately absent: the rank-only facts they used to derive
/// were a second rank model, and rank agreement is unification's
/// (`spec/04-type-system.md` section 4.7.2; PP5 D6 (d) of
/// `spec/design/checker_totality.md`).
///
/// Returns `None` when the call shape is unrecognized, the args are
/// non-concrete, or the derived output would be ill-formed (in which
/// case the validator's own arm will report the diagnostic).
pub(super) fn derive_ir_builtin_output_type(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
    static_env: &UnordMap<String, StaticValue>,
) -> Option<deep::Expr> {
    // chelis#1107 amendment: carrier-preserving entry. The `derive_*` helpers
    // below take `&deep::List`, so bridge a stamped Node once here.
    let mut bridge = None;
    let list = as_list(expr, &mut bridge)?;
    if get_tag(list) != Some(DeepTag::App) {
        return None;
    }
    let func_name = active_ir_builtin_name(list, static_env)?;
    match func_name {
        "conv" => derive_conv_output_type(list, type_env),
        // softmax takes a (tensor, axis) tuple but its output shape
        // equals the input tensor's shape, but it is intentionally not in the
        // rank-polymorphism Identity class because its axis is positional.
        "softmax" => derive_unary_shape_passthrough(list, type_env, static_env),
        // The central shape registry owns every shape-identity builtin. This
        // resolver must consume that registry directly: a second manual
        // allowlist omitted floor_div/mod/clamp/where/bitwise identities and
        // let inline or let-bound rank evidence disappear (chelis#668).
        // `max_elem` and `min_elem` are direct Tier-1 identities; the same
        // registry-owned path preserves their inferred shapes without
        // restoring a second spelling list here.
        _ if crate::shape_class(func_name) == crate::ShapeClass::Identity => {
            derive_identity_shape_passthrough(list, type_env, static_env)
        }
        _ => None,
    }
}

/// Derive the output tensor type of a shape-preserving unary
/// point-wise call: it equals the type of the single argument.
/// Recurses through nested apps so e.g. `relu(conv(...))`
/// resolves to conv's derived output type, peeking through any
/// borrow wrapper as usual (RT-205 round-2 F2).
pub(super) fn derive_unary_shape_passthrough(
    list: &deep::List,
    type_env: &ShapeTypeEnv,
    static_env: &UnordMap<String, StaticValue>,
) -> Option<deep::Expr> {
    let arg = list.elements.get(3)?;
    resolve_let_value_tensor_type(arg, type_env, static_env)
}

/// Derive the output tensor type of any centrally classified identity op from
/// its first tensor argument whose type is structurally resolvable. Identity
/// operations may be unary, binary, or carry scalar parameters (`clamp`,
/// `uniform_like`), so arity-specific allowlists are both unnecessary and a
/// source of registry drift. Rank agreement, broadcasting, and dtype
/// promotion are ordinary inference's; this helper carries an exact shape to
/// the exact-shape validators (`conv` chaining, RT-205) and nothing else.
pub(super) fn derive_identity_shape_passthrough(
    list: &deep::List,
    type_env: &ShapeTypeEnv,
    static_env: &UnordMap<String, StaticValue>,
) -> Option<deep::Expr> {
    list.elements
        .iter()
        .skip(3)
        .find_map(|argument| resolve_let_value_tensor_type(argument, type_env, static_env))
}

/// Resolve the tensor type an identity op publishes for its own consumers.
fn resolve_let_value_tensor_type(
    expr: &deep::Expr,
    type_env: &ShapeTypeEnv,
    static_env: &UnordMap<String, StaticValue>,
) -> Option<deep::Expr> {
    // Peek through borrow before recursing in case a wrapper op
    // appears under an `&` borrow (uncommon but cheap).
    let inner = peel_borrow(expr);
    derive_ir_builtin_output_type(inner, type_env, static_env)
        .or_else(|| expr_shape_type_fact(inner, type_env))
}

/// Derive all convolution spatial extents, preserving the batch dimension's
/// original symbolic identity when it is not a literal.
pub(super) fn derive_conv_output_type(
    list: &deep::List,
    type_env: &ShapeTypeEnv,
) -> Option<deep::Expr> {
    let input_ty = arg_tensor_type_expr(list.elements.get(3)?, type_env)?;
    let kernel_ty = arg_tensor_type_expr(list.elements.get(4)?, type_env)?;
    let input_dims = tensor_dims_from_type_expr(&input_ty)?;
    let kernel_dims = tensor_dims_from_type_expr(&kernel_ty)?;
    if input_dims.len() < 3 || input_dims.len() != kernel_dims.len() {
        return None;
    }
    let params = conv_parameters(
        list.elements.get(5)?,
        list.elements.get(6)?,
        input_dims.len() - 2,
    )
    .ok()??;
    let mut trailing = vec![match kernel_dims[0] {
        DeepDimKind::Lit(n) => n,
        _ => return None,
    }];
    for ((input, kernel), (stride, low, high)) in
        input_dims[2..].iter().zip(&kernel_dims[2..]).zip(params)
    {
        let (DeepDimKind::Lit(input), DeepDimKind::Lit(kernel)) = (input, kernel) else {
            return None;
        };
        trailing.push(conv_output_extent(*input, *kernel, stride, low, high)?);
    }
    Some(build_tensor_type_expr_with_batch(
        tensor_dim_exprs_from_type_expr(&input_ty)?.first()?.clone(),
        &trailing,
        tensor_precision_expr(&input_ty)?,
    ))
}

/// Return the raw Deep `Expr` for each dimension in a `(t-tensor {} dim1
/// dim2 ... prec)`. Unlike `tensor_dims_from_type_expr`, which returns a
/// `DeepDimKind` flattening, this preserves the original
/// `(d-name {} batch)` / `(d-var {} ...)` / `(d-lit {} N)` sub-expression
/// so the caller can carry it forward verbatim when synthesizing a
/// derived tensor type (RT-205 round-3 F-C, symbolic batch propagation).
pub(super) fn tensor_dim_exprs_from_type_expr(expr: &deep::Expr) -> Option<Vec<deep::Expr>> {
    let list = match expr {
        deep::Expr::List(list, _) => list,
        _ => return None,
    };
    if get_tag(list) == Some(DeepTag::TRef) {
        return children(list)
            .first()
            .and_then(tensor_dim_exprs_from_type_expr);
    }
    if get_tag(list) != Some(DeepTag::TTensor) {
        return None;
    }
    let kids = children(list);
    if kids.is_empty() {
        return None;
    }
    Some(kids[..kids.len().saturating_sub(1)].to_vec())
}

/// Extract the precision sub-expression (last child) of a
/// `(t-tensor {} dim1 dim2 ... precision)` expression. Returns the
/// raw Deep `Expr` so it can be re-used unchanged when synthesizing
/// a derived tensor type.
pub(super) fn tensor_precision_expr(ty: &deep::Expr) -> Option<deep::Expr> {
    // chelis#1107 amendment: carrier-preserving read.
    let (tag, _, kids) = stamped_parts(ty)?;
    if tag == DeepTag::TRef {
        return kids.first().and_then(tensor_precision_expr);
    }
    if tag != DeepTag::TTensor {
        return None;
    }
    kids.last().cloned()
}

/// Build a synthetic `(t-tensor {} <batch-dim-expr> (d-lit {} d1)
/// (d-lit {} d2) ... prec)`, placing a verbatim Deep expression at
/// axis 0 (the batch dim) and integer literals for the remaining
/// axes. Used to preserve symbolic batch (`(d-name {} batch)`) when
/// deriving a chained conv's output type (RT-205 round-3 F-C).
/// Spans are zeroed because the derived type is synthetic; downstream
/// lookups care only about the structural shape.
pub(super) fn build_tensor_type_expr_with_batch(
    batch_dim: deep::Expr,
    other_dims: &[i64],
    prec: deep::Expr,
) -> deep::Expr {
    let zero = zero_span();
    let empty_meta = || deep::Metadata::default();
    let make_d_lit = |v: i64| {
        deep::Expr::List(
            deep::List {
                elements: vec![
                    deep::Expr::Atom(deep::Atom::Tag(DeepTag::DLit), zero),
                    deep::Expr::Map(empty_meta(), zero),
                    deep::Expr::Atom(deep::Atom::Int(v), zero),
                ],
            },
            zero,
        )
    };
    let mut elements = vec![
        deep::Expr::Atom(deep::Atom::Tag(DeepTag::TTensor), zero),
        deep::Expr::Map(empty_meta(), zero),
    ];
    elements.push(batch_dim);
    for &d in other_dims {
        elements.push(make_d_lit(d));
    }
    elements.push(prec);
    deep::Expr::List(deep::List { elements }, zero)
}

// ── Helpers ──────────────────────────────────────────────────────

/// chelis#731 Phase 2 ([04-TOT-2] / spec/design/checker_totality.md §C4.1):
/// the ALWAYS-ON totality invariant. After a check completes with an EMPTY
/// error vector, the typed result must contain no error-typed node. With the
/// §C3 `ErrorWitness` token a silent `Type::Error` is unconstructible (every
/// `Type::Error` is minted by `report`, which pushes a diagnostic, or
/// `propagate`, which is downstream of one), so `errors.is_empty()`
/// structurally implies no fresh error was produced. This pass stays on as
/// the standing tripwire that verifies the claim -- it is a pushed internal
/// error, never a panic (the checker is reachable-input territory).
///
/// The shared finalizer checks both authoritative surfaces: every runtime
/// expression/pattern/function node in annotated Deep must carry the stamp
/// produced by its owning inference epoch, and the signature-inference table
/// must contain no structural `Type::Error`. The signature table remains the
/// value-level backstop; the annotated tree now has no re-inference gap.
pub(super) fn annotated_totality_invariant_traces(exprs: &[deep::Expr]) -> Vec<String> {
    fn walk(expr: &deep::Expr, traces: &mut Vec<String>) {
        match expr {
            deep::Expr::Atom(_, _) | deep::Expr::Map(_, _) => {}
            deep::Expr::MetaExpr(meta, _) => walk(&meta.expr, traces),
            deep::Expr::List(list, _) => {
                let tag = get_tag(list);
                let requires_stamp = tag.is_some_and(|tag| {
                    tag == DeepTag::Fn
                        || matches!(tag, DeepTag::PatVar | DeepTag::PatAs)
                        || should_attach_type_metadata(tag)
                });
                if requires_stamp && !get_meta(list).is_some_and(|meta| meta.ty().is_some()) {
                    traces.push(format!(
                        "annotated `{}` node is missing its type stamp",
                        tag.map(DeepTag::as_str).unwrap_or("<untagged-list>")
                    ));
                }

                let kids = children(list);
                for (index, child) in kids.iter().enumerate() {
                    // Decode-once: `child_stamp_role` is total over
                    // `DeepTag`, so the version-skew arm is
                    // unrepresentable; untagged structural lists take the
                    // recursive walk.
                    match tag.map(|tag| child_stamp_role(tag, index, kids.len())) {
                        Some(
                            ChildStampRole::RuntimeExpr | ChildStampRole::ExplicitInferenceBypass,
                        )
                        | None => walk(child, traces),
                        Some(
                            ChildStampRole::Syntax
                            | ChildStampRole::Selector
                            | ChildStampRole::EffectHandler
                            | ChildStampRole::Binder
                            | ChildStampRole::Type,
                        ) => {}
                    }
                }
            }
            // Bridge: reconstruct List so existing tag-dispatch logic runs unchanged (#908)
            deep::Expr::Node(node, span) => {
                let bridged = deep::Expr::List(node.to_list(*span), *span);
                walk(&bridged, traces);
            }
            deep::Expr::BareList(elems, _) => {
                for child in elems {
                    walk(child, traces);
                }
            }
            deep::Expr::UnknownForm(data) => {
                for child in &data.children {
                    walk(child, traces);
                }
            }
        }
    }

    let mut traces = Vec::new();
    for expr in exprs {
        walk(expr, &mut traces);
    }
    traces
}

pub(super) fn totality_invariant_traces(sig: &SignatureInferenceMetadata) -> Vec<String> {
    let mut out = Vec::new();
    for (name, f) in &sig.functions {
        if type_carries_error(&f.checked_signature) {
            out.push(format!(
                "function `{name}` checked_signature carries Type::Error"
            ));
        }
        if type_carries_error(&f.display_signature) {
            out.push(format!(
                "function `{name}` display_signature carries Type::Error"
            ));
        }
        for param in &f.params {
            if type_carries_error(&param.checked_type) || type_carries_error(&param.display_type) {
                out.push(format!(
                    "function `{name}` parameter #{} carries Type::Error",
                    param.index
                ));
            }
        }
    }
    out
}

/// True if `ty` is, or structurally contains, a `Type::Error`.
pub(super) fn type_carries_error(ty: &Type) -> bool {
    match ty {
        Type::Error(_) => true,
        Type::Fn(args, ret) => args.iter().any(type_carries_error) || type_carries_error(ret),
        Type::Ref(inner) => type_carries_error(inner),
        Type::Adt(_, args) => args.iter().any(type_carries_error),
        Type::KindedAdt(_, args) => args
            .iter()
            .any(|argument| argument.as_type().is_some_and(type_carries_error)),
        Type::Tuple(elems) => elems.iter().any(type_carries_error),
        Type::Prim(_) | Type::Tensor(_, _) | Type::Var(_) | Type::Unit => false,
    }
}

/// Build the internal diagnostic for a totality-invariant violation
/// ([04-TOT-2]). Only reachable if the §C3 witness token were bypassed (a
/// deserialization mint or a genuine bug); it names the offending nodes so a
/// regression is localizable rather than a bare "internal error".
pub(super) fn totality_violation_error(traces: &[String]) -> CheckError {
    CheckError::new(
        CheckErrorKind::Other,
        format!(
            "internal: [04-TOT-2] totality invariant violated -- the check reported \
             success (empty error vector) but the typed result carries {} silent \
             Type::Error verdict(s): {} (spec/design/checker_totality.md \u{00a7}C4.1; \
             chelis#731 Phase 2)",
            traces.len(),
            traces.join("; ")
        ),
        vec![],
    )
}

#[cfg(test)]
mod cancellation_tests {
    use super::*;

    #[derive(Debug)]
    struct Observation {
        reported_success: bool,
        messages: Vec<String>,
    }

    type Entry = fn(&[deep::Expr]) -> Observation;

    fn validation_only_failure_fixture() -> Vec<deep::Expr> {
        let declarations = chelis_surf::parser::parse_str(
            "def f(x: tensor[1, 1, 4, f32], k: tensor[1, 1, 2, f32]) = \
             conv(x, k, [0i64], [(0i64, 0i64)])\n",
        )
        .expect("the zero-stride Surf fixture must parse");
        chelis_surf::desugar::desugar_program(&declarations)
    }

    fn observe_check(result: Result<CheckedProgram, InferResult>) -> Observation {
        match result {
            Ok(_) => Observation {
                reported_success: true,
                messages: Vec::new(),
            },
            Err(result) => Observation {
                reported_success: false,
                messages: result
                    .errors
                    .into_iter()
                    .map(|error| error.message)
                    .collect(),
            },
        }
    }

    fn observe_infer(result: InferResult) -> Observation {
        Observation {
            reported_success: result.errors.is_empty(),
            messages: result
                .errors
                .into_iter()
                .map(|error| error.message)
                .collect(),
        }
    }

    fn run_with_validator_cancellation(
        exprs: &[deep::Expr],
        run: impl FnOnce(&[deep::Expr]) -> Observation,
    ) -> Observation {
        let token = crate::cancel::CancelToken::new();
        let _cancel_guard = crate::cancel::install_cancel_token(token);
        let _hook_guard = cancel_before_declaration_validation_for_test();
        run(exprs)
    }

    #[test]
    fn cancellation_inside_shared_semantic_validation_is_hard_at_every_public_entry() {
        let exprs = validation_only_failure_fixture();
        let entries: [(&str, Entry); 4] = [
            ("check_ir_program", |exprs| {
                observe_check(crate::check_ir_program(exprs))
            }),
            ("check_typed_program", |exprs| {
                observe_check(crate::check_typed_program(exprs))
            }),
            ("infer_ir_program", |exprs| {
                observe_infer(crate::infer_ir_program(exprs))
            }),
            ("infer_program", |exprs| {
                observe_infer(crate::infer_program(exprs))
            }),
        ];

        for (entry, run) in entries {
            let observation = run_with_validator_cancellation(&exprs, run);
            assert!(
                !observation.reported_success,
                "{entry} returned without a cancellation diagnostic: {observation:?}"
            );
            assert_eq!(
                observation.messages,
                [crate::cancel::EVAL_CANCELLED_MSG.to_string()],
                "{entry} must report exactly one cancellation diagnostic in order"
            );
        }
    }
}
