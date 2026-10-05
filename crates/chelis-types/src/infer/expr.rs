//! Expression dispatch and common expression helpers.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;
use crate::unsupported::{SpanRef, Stage, Unsupported, UnsupportedKind};

/// Exact ownership record for a root expression consumer that resolves its
/// own `type:` metadata. `infer_let` uses this instead of inferring ownership
/// from a generic `Type::Error`: an unrelated RHS failure must not suppress an
/// independently malformed let ascription.
pub(super) enum OwnedTypeMetadataResolution {
    Resolved(Type),
    Failed(ErrorWitness),
}

/// Keep the comparatively large typed rejection construction out of
/// `infer_expr_with_type_metadata_ownership`'s recursive stack frame. The
/// stack-depth guards are a hard checker boundary, so a fence that makes every
/// unrelated recursive expression consume more stack would be a regression.
fn report_par_fence(node: &DeepNode, source_span: &Span, errors: &mut DiagnosticSink<'_>) {
    // Surf desugaring carries its authored location only as an opaque
    // `span_id`; the structural Deep span remains the default 0/0. Do not
    // publish that sentinel as a measured source range. Native Deep nodes do
    // carry a non-empty structural range, which remains valid independently
    // of any opaque identity attached to the node.
    let (offset, len) = if source_span.len == 0 {
        (None, None)
    } else {
        (Some(source_span.offset), Some(source_span.len))
    };
    let unsupported = Unsupported::new(
        UnsupportedKind::Construct("`par` expression".to_string()),
        "every evaluation and build target",
        Stage::Checker,
        crate::unimplemented_rejection!(
            2503,
            "`par` is not fully implemented in the evaluator or in compiled code; \
             use `do { ... }` when sequential evaluation is intended"
        ),
    )
    .with_span(SpanRef {
        offset,
        len,
        span_id: node_span_id(node).map(str::to_owned),
    })
    .with_supported_alternative("use `do { ... }` when sequential evaluation is intended");
    errors.push(CheckError::from_unsupported(unsupported));
}

/// The whole of `copy`'s inference, in one place.
///
/// Both `DeepTag::Copy` arms were byte-identical apart from how they spelled
/// the list, so they were two chances to fix a bug once (chelis#1489).
#[allow(clippy::too_many_arguments)]
fn infer_copy(
    expr: &deep::Expr,
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = node.children_slice();
    if let Some(inner) = kids.first() {
        let inner_ty = infer_expr(inner, env, vg, subst, adt_reg, errors, product);
        let resolved = subst.apply(&inner_ty);
        match copy_result_from_source(&resolved) {
            Some(settled) => settled,
            None => match resolved {
                // chelis#1489: the operand may simply not be
                // resolved YET. Suspend the decision on it, and
                // unification calls `copy_result_from_source`
                // with the settled type at the instant that
                // variable is bound. A variable that is never
                // bound is still rejected by the per-def pass,
                // so this defers the decision rather than
                // dropping it.
                //
                // The result is a FRESH variable, not the
                // operand's, and discharge unifies the shared
                // decision's answer into it. Returning the
                // operand's variable made a deferred `copy(&t)`
                // type as `&t` where an eager one is `t` -- the
                // inference-order sensitivity this issue exists to
                // delete, reintroduced by its own fix.
                Type::Var(tv) => {
                    let result = vg.fresh_type();
                    subst.record_deferred_tensor_operand(
                        tv,
                        DeferredOperandGate::Copy {
                            result: Box::new(result.clone()),
                            location: TypeDiagnosticLocation::from_expr(expr),
                        },
                    );
                    result
                }
                _ => report(
                    errors,
                    at_check_site(
                        expr,
                        CheckError::with_types(
                            CheckErrorKind::TypeMismatch,
                            format!("copy argument 1: expected tensor, got {resolved}"),
                            "tensor".to_string(),
                            resolved.to_string(),
                            vec!["Wrap only tensor values in copy".to_string()],
                        ),
                    ),
                ),
            },
        }
    } else {
        malformed_form(node, "copy", "one wrapped expression", errors)
    }
}

/// The type `&x` has once `x`'s type is settled: a reference stays itself,
/// a tensor or tensor-carrying value (or an error) is borrowed, and anything
/// else is no borrow (`None`), which the caller reports. The eager `borrow`
/// arm and the discharge of a cast suspended on `&v` (chelis#3101) both
/// decide through it, so the two cannot disagree.
pub(crate) fn settled_borrow_type(resolved: Type) -> Option<Type> {
    match resolved {
        Type::Ref(_) => Some(resolved),
        Type::Tensor(_, _)
        | Type::Adt(_, _)
        | Type::KindedAdt(_, _)
        | Type::Tuple(_)
        | Type::Error(_) => Some(Type::Ref(Box::new(resolved))),
        _ => None,
    }
}

/// The source-side decision `copy` makes, factored out so the eager arm and
/// the suspended constraint's discharge run *the same code* (chelis#1489).
///
/// Returns `None` when the operand is not something `copy` accepts, leaving
/// the caller to report — the eager arm and discharge word that rejection
/// differently, and only the decision is shared.
///
/// This exists because the decision previously lived in three places: two
/// identical eager arms and a re-derivation in the pass that decided the
/// deferrals. They agreed, but nothing made them agree, and every defect on
/// chelis#1489 has been some path disagreeing with another. A comment even
/// claimed this function existed before it did, which is worse than the
/// duplication: a reviewer reading it would stop looking.
pub(crate) fn copy_result_from_source(resolved: &Type) -> Option<Type> {
    match resolved {
        Type::Tensor(_, _) | Type::Error(_) => Some(resolved.clone()),
        // `copy(&t)` yields `t`, not `&t`.
        Type::Ref(inner) if matches!(inner.as_ref(), Type::Tensor(_, _)) => {
            Some(inner.as_ref().clone())
        }
        _ => None,
    }
}

/// Named type/dimension/rank variables in source annotations are closed by
/// default. Only the lexical environment cloned for a matching `defsig` body
/// exposes an explicit binder set.
pub(super) fn annotation_binder_mode(env: &Env) -> BinderMode<'_> {
    env.type_resolution_binders()
        .map(BinderMode::Lexical)
        .unwrap_or(BinderMode::ClosedInput)
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_expr(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    infer_expr_with_type_metadata_ownership(
        expr, env, vg, subst, adt_reg, errors, product, None, None, None,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_expr_with_declaration_diagnostic_owner(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    declaration_diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
) -> Type {
    infer_expr_with_type_metadata_ownership(
        expr,
        env,
        vg,
        subst,
        adt_reg,
        errors,
        product,
        None,
        None,
        declaration_diagnostic_owner,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_expr_with_expected(
    expr: &deep::Expr,
    expected: &Type,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    infer_expr_with_type_metadata_ownership(
        expr,
        env,
        vg,
        subst,
        adt_reg,
        errors,
        product,
        Some(expected),
        None,
        None,
    )
}

/// A Surf literal ascription keeps the literal intact inside a typed block.
/// Retain the opaque-forgery diagnostic at that checking boundary as well as
/// at the direct Deep literal-metadata boundary.
pub(super) fn check_opaque_literal_ascription(
    expr: &deep::Expr,
    declared: &Type,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> bool {
    let Some((DeepTag::Block, _, [value])) = stamped_parts(expr) else {
        return false;
    };
    if !matches!(stamped_parts(value), Some((DeepTag::Lit, _, _))) {
        return false;
    }
    match declared {
        Type::Adt(name, _) | Type::KindedAdt(name, _) => crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::LitForge,
            name,
            adt_reg,
            errors,
        ),
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_expr_with_type_metadata_ownership(
    expr: &deep::Expr,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    expected_result: Option<&Type>,
    type_metadata_resolution: Option<&mut Option<OwnedTypeMetadataResolution>>,
    declaration_diagnostic_owner: Option<&DeclarationDiagnosticOwner>,
) -> Type {
    // Bail before a deeply-nested `app` tree exhausts the native stack and
    // aborts the process (this is the gdb-pinned real-pricer crash site):
    // when the budget is gone we record the located bail in `STACK_EXHAUSTED`
    // (so the check entry boundary fails hard) and return `Type::Error`,
    // letting the stack unwind normally (no panic, no `catch_unwind`). See
    // `STACK_RED_ZONE_BYTES`. The WI-1 follow-up grows the native stack ONCE at
    // each public check entry (`with_grown_stack`), so on a legitimately deep
    // but finite program every recursive pass -- this one and the validate /
    // annotate / clone / drop passes -- runs inside the grown segment and this
    // guard does not fire; it remains the safety net for input deeper than the
    // grown segment can hold (covered-or-rejected).
    stack_guard!("infer_expr", expr, vg.fresh_type());

    product.total_nodes += 1;
    let type_metadata_owned_by_caller = type_metadata_resolution.is_some();

    let result = match expr {
        deep::Expr::Atom(atom, _) => infer_atom(atom, errors),
        deep::Expr::Map(_, _) => Type::Unit,
        deep::Expr::MetaExpr(meta, _) => {
            infer_expr(&meta.expr, env, vg, subst, adt_reg, errors, product)
        }
        deep::Expr::Node(node, source_span) => {
            // chelis#731 Phase 3 (checker_totality.md §C4.2): dispatch on
            // the typed closed vocabulary. A vocabulary node has the single
            // spelling `Expr::Node` whatever its ingress (chelis#1125). The
            // match is exhaustive with no `_` arm, so a 63rd `DeepTag`
            // variant fails to compile until this dispatch chooses its
            // disposition.
            match node.tag() {
                DeepTag::Var => infer_var(node, env, vg, subst, adt_reg, errors, product),
                DeepTag::Lit => infer_lit(node, env, vg, adt_reg, errors, type_metadata_resolution),
                DeepTag::App => infer_app(
                    expr,
                    node,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    product,
                    expected_result,
                ),
                DeepTag::Fn => infer_fn(
                    node,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    product,
                    declaration_diagnostic_owner,
                ),
                DeepTag::Let => infer_let(node, env, vg, subst, adt_reg, errors, product),
                DeepTag::If => infer_if(expr, node, env, vg, subst, adt_reg, errors, product),
                DeepTag::Match => infer_match(node, env, vg, subst, adt_reg, errors, product),
                DeepTag::Tuple => infer_tuple(node, env, vg, subst, adt_reg, errors, product),
                DeepTag::TupleGet => {
                    infer_tuple_get(node, env, vg, subst, adt_reg, errors, product)
                }
                DeepTag::Record => infer_record(node, env, vg, subst, adt_reg, errors, product),
                DeepTag::Access => infer_access(node, env, vg, subst, adt_reg, errors, product),
                DeepTag::RecordUpdate => {
                    infer_record_update(node, env, vg, subst, adt_reg, errors, product)
                }
                DeepTag::Cast => infer_cast(expr, node, env, vg, subst, adt_reg, errors, product),
                DeepTag::Grad => infer_grad(expr, node, env, vg, subst, adt_reg, errors, product),
                DeepTag::Vmap => infer_vmap(node, env, vg, subst, adt_reg, errors, product),
                DeepTag::Def => infer_def(node, env, vg, subst, adt_reg, errors, product),
                DeepTag::Defsig => {
                    // Already handled in first pass
                    Type::Unit
                }
                DeepTag::Deftype | DeepTag::Typealias => {
                    // Already handled in first pass
                    Type::Unit
                }
                // chelis#731 [04-TOT-1]: top-level declaration and module
                // tags carry no expression type. They reach `infer_expr` when
                // a program is checked item-by-item; their real processing is
                // the declaration passes (imports/exports/module framing,
                // dimension declarations), so their expression-level
                // disposition is an explicit `Unit` no-op. Before the loud
                // wildcard they fell through to a silent `Type::Error` that
                // happened to be harmless because the value is discarded; the
                // explicit no-op keeps that behavior while satisfying the
                // every-tag-has-a-disposition contract (never the loud arm,
                // which would wrongly flag a well-formed `(export ...)`).
                DeepTag::Module
                | DeepTag::Import
                | DeepTag::ImportAll
                | DeepTag::Export
                | DeepTag::Defdim => Type::Unit,
                DeepTag::Block => {
                    // chelis#859: sequenced expressions; the value (and
                    // type) is the last child's (spec/03 §2.3). Every child
                    // is checked in order so non-last children keep their
                    // own diagnostics. A childless block has no value and
                    // is malformed ([04-TOT-3]); lowering raises on the
                    // same shape.
                    let kids = node.children_slice();
                    if kids.is_empty() {
                        malformed_form(node, "block", "at least one child expression", errors)
                    } else {
                        let mut last_ty = Type::Unit;
                        for kid in kids {
                            last_ty = infer_expr(kid, env, vg, subst, adt_reg, errors, product);
                        }
                        last_ty
                    }
                }
                DeepTag::Par => {
                    // chelis#2503: every source ingress is fenced until the
                    // evaluator and compiled lanes preserve the same `par`
                    // effects. Keep checking children so this fence does not
                    // hide their independent diagnostics.
                    let kids = node.children_slice();
                    let mut last_ty = Type::Unit;
                    for kid in kids {
                        last_ty = infer_expr(kid, env, vg, subst, adt_reg, errors, product);
                    }
                    report_par_fence(node, source_span, errors);
                    last_ty
                }
                DeepTag::Jit => {
                    // jit: compilation trigger; semantically a no-op at eval
                    // (spec/03-deep-syntax.md §2.7). Type is the type of the
                    // wrapped expression.
                    let kids = node.children_slice();
                    if let Some(inner) = kids.first() {
                        infer_expr(inner, env, vg, subst, adt_reg, errors, product)
                    } else {
                        malformed_form(node, "jit", "one wrapped expression", errors)
                    }
                }
                DeepTag::Realize => {
                    let kids = node.children_slice();
                    if let Some(inner) = kids.first() {
                        infer_expr(inner, env, vg, subst, adt_reg, errors, product)
                    } else {
                        malformed_form(node, "realize", "one wrapped expression", errors)
                    }
                }
                DeepTag::Copy => infer_copy(expr, node, env, vg, subst, adt_reg, errors, product),
                DeepTag::Borrow => {
                    let kids = node.children_slice();
                    if let Some(inner) = kids.first() {
                        let inner_ty = infer_expr(inner, env, vg, subst, adt_reg, errors, product);
                        let resolved = subst.apply(&inner_ty);
                        match resolved {
                            // Issue #256: when the borrow inner is still an
                            // unresolved type variable (e.g. the output of a
                            // polymorphic-return call whose dim variables
                            // have not yet been pinned at this point in
                            // left-to-right inference), defer the
                            // tensor-or-carrier classification to subsequent
                            // unification. Wrapping as `Type::Ref(Type::Var)`
                            // lets the surrounding flow's expected argument
                            // type (e.g. a sig parameter `&tensor[..]`) pin
                            // the variable through unification. If the
                            // variable never gets pinned to a tensor or
                            // tensor-carrying type, the later unification
                            // failure surfaces the same diagnostic via the
                            // mismatched call site -- there is no silent
                            // accept. The linearity checker's
                            // `expr_is_owned_or_borrow_linear` still rejects
                            // a stamped `(t-var ...)` if no pinning happens.
                            //
                            // Soundness ledger (issue #256 round 2): record
                            // the inner type variable so the inference driver
                            // can re-check it against the *final*
                            // substitution after the def body completes. The
                            // deferral is sound only when the variable is
                            // eventually pinned to a tensor or tensor carrier;
                            // a fully-polymorphic consumer (e.g.
                            // `consume_any[a](t: a)`) never pins it, and a
                            // genuinely-non-tensor value would otherwise slip
                            // past every gate. See
                            // `validate_deferred_borrow_vars`.
                            Type::Var(tv) => {
                                subst.record_deferred_borrow_var(tv);
                                Type::Ref(Box::new(Type::Var(tv)))
                            }
                            settled => match settled_borrow_type(settled.clone()) {
                                Some(borrowed) => borrowed,
                                None => report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        format!(
                                            "borrow requires tensor or tensor-carrying input, got {settled}"
                                        ),
                                        vec!["Use `&x` only with tensor values".to_string()],
                                    ),
                                ),
                            },
                        }
                    } else {
                        malformed_form(node, "borrow", "one wrapped expression", errors)
                    }
                }
                DeepTag::HandleEffect => {
                    infer_handle_effect(node, env, vg, subst, adt_reg, errors, product)
                }
                // chelis#731 Phase 3 [04-TOT-1]: in-vocabulary tags with no
                // expression-position inference case: declaration internals,
                // patterns, type/dimension syntax, metaprogramming forms,
                // and structural helpers checked by their owning enclosing
                // form (the `child_stamp_role` ownership table), so reaching
                // expression dispatch means the node sits outside its owning
                // parent. (`block` graduated to a real case per chelis#859.)
                // The disposition is an explicit loud rejection, never a
                // silent `Type::Error` that would exempt the subtree (the
                // chelis#709 class defect). Before Phase 3 these fell through
                // the unknown-tag wildcard, whose message wrongly claimed
                // they were outside the vocabulary.
                undispatched @ (DeepTag::Variant
                | DeepTag::Field
                | DeepTag::Arm
                | DeepTag::PatVar
                | DeepTag::PatLit
                | DeepTag::PatCtor
                | DeepTag::PatTuple
                | DeepTag::PatRecord
                | DeepTag::PatWild
                | DeepTag::PatAs
                | DeepTag::TPrim
                | DeepTag::TFn
                | DeepTag::TTensor
                | DeepTag::TRef
                | DeepTag::TAdt
                | DeepTag::TVar
                | DeepTag::TUnit
                | DeepTag::TTuple
                | DeepTag::DName
                | DeepTag::DVar
                | DeepTag::DLit
                | DeepTag::DRank
                | DeepTag::Quote
                | DeepTag::Unquote
                | DeepTag::Splice
                | DeepTag::Params
                | DeepTag::Bind
                | DeepTag::Kv
                | DeepTag::Effects
                | DeepTag::Resource) => {
                    let named = undispatched.as_str();
                    report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::UnknownForm,
                            format!(
                                "Deep tag `{named}` has no expression-position checker \
                             disposition (helper/pattern/type syntax outside its owning \
                             form, or an expression form with no implemented case; \
                             spec/03-deep-syntax.md; chelis#731 [04-TOT-1])"
                            ),
                            vec![],
                        ),
                    )
                }
            }
        }
        // `BareList` is an untagged structural list — no vocabulary head.
        //
        // chelis#1085 / [04-TOT-1]: this arm used to infer each child and
        // return the LAST child's type — `Type::Unit` for the empty list —
        // and count the node as typed. So `(def {} f ())` type-checked
        // clean through `check_typed_program`, which walks stamped Deep
        // directly, while `check_ir_program` rejected the same program
        // loudly because it normalized `BareList` into a tagless legacy
        // list first. Same program, two verdicts, and the permissive one
        // scored a form with no honest type as checked (the chelis#873-shape
        // fail-open).
        //
        // A headless list has no expression-position type in any position,
        // so the disposition is the sibling arms' loud rejection. Per
        // `chelis_deep::role::child_stamp_role`, the stamp pass produces a
        // `BareList` at a RuntimeExpr or form-expecting slot only when the
        // list is EMPTY (a non-empty list there decodes to `Node` or
        // `UnknownForm`), and non-empty bare lists occupy only
        // Syntax/Binder/Selector/Type slots, whose owning form consumes
        // them structurally rather than through `infer_expr` — a `(params
        // {} (x {type: ...}))` entry is read by `infer_fn`, never inferred
        // as an expression. Reaching expression dispatch therefore means
        // malformed input, or a structural helper sitting outside its
        // owning form. Both are rejections.
        //
        // The one legitimate empty `()` in expression position — a match
        // arm's absent guard — is screened by `infer_match` before it
        // reaches here, so this arm does not reject it.
        deep::Expr::BareList(elems, _span) => {
            let shape = if elems.is_empty() {
                "an empty list `()`".to_string()
            } else {
                format!("an untagged list with {} element(s)", elems.len())
            };
            report(
                errors,
                CheckError::new(
                    CheckErrorKind::UnknownForm,
                    format!(
                        "{shape} in expression position has no checker disposition: a \
                         headless list is not a canonical `(tag {{}} ...)` Deep expression \
                         (spec/03-deep-syntax.md; chelis#1085 [04-TOT-1])"
                    ),
                    vec![
                        "write the canonical Deep form for the value you mean, such as \
                         `(lit {} 0)` or `(var {} name)`"
                            .to_string(),
                    ],
                ),
            )
        }
        // `UnknownForm` is a list whose head didn't decode into the vocabulary.
        deep::Expr::UnknownForm(data) => report(
            errors,
            CheckError::new(
                CheckErrorKind::UnknownForm,
                format!(
                    "unknown Deep tag `{}` has no checker disposition (not in the 62-tag \
                     closed vocabulary of spec/03-deep-syntax.md; chelis#731 [04-TOT-1])",
                    data.head
                ),
                vec![],
            ),
        ),
    };

    let result = if !type_metadata_owned_by_caller
        && !matches!(stamped_parts(expr), Some((DeepTag::Lit, _, _)))
        && let Some((_, meta, _)) = stamped_parts(expr)
        && let Some(authored_type) = meta.ty()
    {
        let declared = match resolve_deep_type_with_diagnostic_owner(
            authored_type.expression(),
            vg,
            adt_reg,
            TypeUseSite::Annotation,
            annotation_binder_mode(env),
            env.type_resolution_diagnostic_owner(),
            errors,
        ) {
            Ok(declared) => declared,
            Err(witness) => propagate(&witness),
        };
        if check_opaque_literal_ascription(expr, &declared, adt_reg, errors) {
            declared
        } else if product.defer_result_type_constraint(&result, &declared, subst) {
            result
        } else {
            if let Err(error) = unify(&result, &declared, subst) {
                let expected = subst.apply(&declared);
                let got = subst.apply(&result);
                let mut diagnostic = CheckError::with_types(
                    check_error_kind_from_type_error_kind(&error.kind),
                    format!(
                        "expression ascription does not match value: expected {expected}, got {got}"
                    ),
                    expected.to_string(),
                    got.to_string(),
                    vec![format!(
                        "Declared expression type is {declared}; inferred value type is {result}"
                    )],
                );
                if let Some(location) =
                    TypeDiagnosticLocation::from_expr(authored_type.expression())
                {
                    diagnostic = location.attach(diagnostic);
                }
                if let Some(location) = TypeDiagnosticLocation::from_expr(expr) {
                    diagnostic = location.attach(diagnostic);
                }
                errors.push(diagnostic);
            }
            declared
        }
    } else {
        result
    };

    if !matches!(result, Type::Error(_)) {
        product.typed_nodes += 1;
    }
    product.record_canonical(expr, result.clone());

    result
}

/// chelis#709 / spec/design/checker_totality.md §C1.5: the `handle-effect`
/// checker case. Before this, `handle-effect` fell through `infer_expr`'s
/// wildcard to a silent `Type::Error`, so `with device` bodies were not
/// type-checked at all and the enclosing `def`'s declared return type went
/// unenforced.
///
/// The typing SHAPE mirrors LaCaDiLE's T-Handle rule (Jeff's 2026-07 note on
/// chelis#709): type the body in the enclosing context and return the BODY's
/// type, so the enclosing signature is enforced against it. The handler is
/// checked per effect kind through the closed [`EffectKind`] set, with a loud
/// `MalformedForm` for a missing, malformed or unknown kind.
///
/// `resource` (`with device`) is the one handler kind: the device is a string
/// literal, checked by the shared effects gate (`chelis-effects`
/// `validate_handler_expr`, spec/02 §P5a); the device-name vocabulary is not
/// validated here (target knowledge, chelis#735). Randomness has no handler:
/// a random primitive takes an explicit key (spec/04 §7.1).
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_handle_effect(
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = node.children_slice();
    // Structural arity: spec/03-deep-syntax.md gives `handle-effect` EXACTLY
    // two children -- `(handle-effect {effect: K} <handler> <body>)`. Any other
    // count is malformed Deep; reject loudly ([04-TOT-3]). A THIRD child in
    // particular carries an ill-typed subtree the checker would never visit
    // (chelis#731 red team) -- the chelis#710 silent-exemption class -- so
    // tolerating `>= 2` would be a fitness-honesty hole.
    if kids.len() != 2 {
        return malformed_form(
            node,
            "handle-effect",
            "exactly two children (a handler expression and a body)",
            errors,
        );
    }
    let body = &kids[1];

    // The effects gate owns the handler's literal admission. Do not infer
    // arbitrary handler expressions here: their internal diagnostics would
    // preempt the owning handler rejection.
    //
    // chelis#730 Phase 2 (section C4.4; the pinned §I1 interlock): the kind
    // is parsed once into the closed [`EffectKind`] set - the same enum the
    // lowering lanes now use - and dispatched with an exhaustive `match`
    // over `EffectKind` (no wildcard arm). Adding a kind is a compile error
    // here until this checker case handles it. Decode failures preserve the
    // missing/malformed/unknown distinction from the shared Deep adapter.
    match decode_effect_kind(node.meta()) {
        Ok(EffectKind::Resource) => {
            // Device literal-ness is the effects gate's job; the device-name
            // vocabulary is not validated here (target knowledge, chelis#735).
        }
        Err(error) => {
            // §C1.5 / §I1: an unknown effect kind is malformed. Both the IR
            // and host lowering lanes now raise a branded `unsupported:`
            // diagnostic for an unknown kind too (chelis#730 Phase 1 rows
            // 9/20), but the checker is the earliest competent stage and
            // rejects it first here ([05-UNS-2]).
            errors.push(CheckError::new(
                CheckErrorKind::MalformedForm,
                format!(
                    "{error} in `handle-effect`: the checker \
                     recognizes only `resource` (with device) \
                     (spec/03-deep-syntax.md; chelis#730, chelis#731)"
                ),
                vec![],
            ));
        }
    }

    // T-Handle: return the BODY's type in the enclosing context. This is the
    // chelis#709 fix -- the enclosing `def` signature is now enforced against
    // the body, so an i64 body in an `-> f32` def, or a tensor body from a
    // scalar-typed fn, is a type error caught before any backend sees it.
    infer_expr(body, env, vg, subst, adt_reg, errors, product)
}

/// Type a bare atom in expression position.
///
/// `deep::Atom` unions two different kinds of thing: value literals
/// (`Int`/`Float`/`Bool`/`Str`, which denote runtime values and have types)
/// and structural tokens (`Symbol`/`Keyword`, which are the names forms are
/// built out of and never denote a value). Structural positions consume the
/// latter directly via `symbol_name` (a `var`'s name, a `record`'s
/// constructor head, a `kv` key) and never route them here, so an atom that
/// reaches this function as a `Symbol` or `Keyword` is in expression position
/// and has no type to give.
///
/// chelis#710 form 4 / chelis#873: those two variants used to return
/// `Type::Unit`, which is a verdict with no record. When nothing downstream
/// forced that `Unit` to meet a concrete type (a `def` with no `defsig`, a
/// signature returning `t-unit`, an unused `let` binding) `chelis check`
/// scored the program a perfect 1.0 while `chelis build` refused to lower it,
/// so the score lied about the one output spec/04-type-system.md §10
/// designates as agent feedback. The rule already existed one lane down
/// in `chelis_ir::lower::lower_atom`, whose diagnostic states the law this
/// site was breaking: unhandled or malformed forms cannot become Unit or
/// another value ([05-UNS-1]; chelis#730). Reporting here says the same thing
/// at the lane that is the oracle, and re-types the site `Type::Error` so it
/// sits back under the Phase 2 `ErrorWitness` guard (chelis#731 §C3).
pub(super) fn infer_atom(atom: &deep::Atom, errors: &mut DiagnosticSink<'_>) -> Type {
    match atom {
        // D1 (WS-A0 RT-1 fixup): per spec/04-type-system.md §5.3 the
        // lexer parses unsuffixed integer tokens at i64 so that
        // out-of-range literals can be diagnosed before the i32
        // narrowing. The bare-atom path is the value-only fallback for
        // Deep code that bypasses the desugarer's `(lit {type: i32}
        // N)` wrapping; the same range check is enforced more visibly
        // at `infer_lit` where the type metadata is in scope.
        // Out-of-range here would silently wrap to a negative i32 if
        // we let it default unchecked — exactly what §5.3 forbids.
        // The sink is now threaded (chelis#873), so a future range
        // check on this path has a failure channel; `infer_lit` remains
        // the more visible diagnostic site because the type metadata is
        // in scope there.
        deep::Atom::Int(_) => Type::Prim(Prim::Int32),
        deep::Atom::Float(_) => Type::Prim(Prim::F32),
        deep::Atom::Bool(_) => Type::Prim(Prim::Bool),
        deep::Atom::Str(_) => Type::Prim(Prim::String),
        deep::Atom::Name(name) => report(
            errors,
            CheckError::new(
                CheckErrorKind::MalformedForm,
                format!(
                    "a bare symbol atom `{name}` in expression position cannot be typed or \
                     lowered to the executable IR (spec/design/loud_unsupported.md section \
                     C1.4; chelis#710 form 4)"
                ),
                vec![format!("to reference a binding, write `(var {{}} {name})`")],
            ),
        ),
    }
}

/// [04-LIN-9] and spec/04 section 1.1: a builtin named as a value rather
/// than called (`map(len, xs)`, a key builtin in a tuple or an `if` arm) is
/// judged by the key allow-list at the parameters of the function type it
/// is instantiated at, as a call is judged operand by operand in linearity.
/// Every type variable of a parameter the builtin does not admit a key at
/// (`crate::key_admission::builtin_key_operand`, over all its sibling cases)
/// is key-free, so instantiating it at a key-carrying type is refused and
/// names the builtin. A user binding that shadows a builtin name is a
/// generic whose variables generalization already made key-free.
fn forbid_keys_a_builtin_value_does_not_admit(name: &str, ty: &Type, subst: &Subst) {
    let Some(decl) = crate::builtins::builtin_decl(name) else {
        return;
    };
    let Type::Fn(params, _) = ty else {
        return;
    };
    let cases: Vec<crate::builtins::BuiltinSiblingCaseId> = decl
        .capability
        .sibling_cases
        .iter()
        .map(|case| case.case)
        .collect();
    for (index, param) in params.iter().enumerate() {
        if crate::key_admission::builtin_key_operand(name, &cases, index).is_ok() {
            continue;
        }
        for var in crate::env::free_tvars(param) {
            subst.forbid_key_instantiation(
                var,
                crate::unify::GenericParameter {
                    generic: Some(name.to_string()),
                    binder: None,
                    value: false,
                },
            );
        }
    }
}

/// [04-INF-9]: a builtin named anywhere other than as the callee of an
/// application must carry its whole operation contract on the value
/// (`builtins::builtin_value_contract_carried`). One that does not is
/// applicable only by name, whatever value position it reaches: a binding,
/// an argument, a callback, an aggregate element, or a result.
fn reject_builtin_applicable_only_by_name(
    name: &str,
    scheme: &Scheme,
    node: &DeepNode,
) -> Option<CheckError> {
    if !crate::builtins::builtin_env_names().contains(name) {
        return None;
    }
    // Fail closed: a bound builtin with no declaration has no reviewed
    // contract, so it is not a value.
    if let Some(decl) = crate::builtins::builtin_decl(name)
        && crate::builtins::builtin_value_contract_carried(decl, scheme)
    {
        return None;
    }
    let mut error = CheckError::new(
        CheckErrorKind::TypeMismatch,
        with_node_provenance(
            node,
            format!(
                "builtin `{name}` is applicable only by name and is not a function value: \
                 its type scheme does not state its whole operation contract (a static \
                 argument, an operand restriction, or a result rule that only a direct call \
                 checks), so the contract would not travel with a value \
                 (spec/04-type-system.md [04-INF-9])"
            ),
        ),
        vec![format!(
            "Call `{name}` directly, or pass an explicitly typed lambda that calls it \
             (for example `fn (x: tensor[3, f32]) -> sum(x, 0i32)` in place of `sum`)"
        )],
    );
    if let Some(sid) = node_span_id(node) {
        error.span_offset = parse_span_offset(sid);
        error.span_id = Some(sid.to_string());
    }
    Some(error)
}

pub(super) fn infer_var(
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let called = std::mem::take(&mut product.callee_reference);
    let kids = node.children_slice();
    if let Some(name) = kids.first().and_then(|e| symbol_name(e)) {
        // chelis#317: a nullary constructor at a construction site (a bare
        // `Alpha`, desugared to `(var Alpha)`) or an applied constructor
        // head (`Foo(x)` → `(app (var Foo) ...)`) that is out of scope must
        // be an `unknown constructor` error at `check`, not a silent bind
        // to a foreign module's same-terminal tag via the registry's fuzzy
        // fallback. Check exact scope first; the fuzzy `lookup_terminal_unique`
        // is the mis-resolution path the issue reports.
        if bare_constructor_out_of_scope(name, env) {
            let mut err = CheckError::new(
                CheckErrorKind::UnknownConstructor {
                    identifier: name.to_string(),
                },
                with_node_provenance(node, format!("unknown constructor: {name}")),
                vec![format!(
                    "Constructor '{name}' is not in scope. Declare it locally or add it \
                     to an import (e.g. `import Mod ({name})`)"
                )],
            );
            if let Some(sid) = node_span_id(node) {
                if let Some(off) = parse_span_offset(sid) {
                    err.span_offset = Some(off);
                }
                err.span_id = Some(sid.to_string());
            }
            return report(errors, err);
        }
        // [04-INF-4]: a top-level eager value is visible only from its own
        // declaration onward. The test is on source position, never on
        // whether a binding happens to exist: the body schedule reorders
        // function declarations, so binding presence answers a different
        // question. When the not-yet-declared name shadows an outer import,
        // that outer binding is still the one in scope here.
        let resolved_scheme = match env.top_level_value_visibility(name) {
            TopLevelValueVisibility::Visible => env.lookup(name).cloned(),
            TopLevelValueVisibility::NotYetDeclared { shadowed } => shadowed.cloned(),
        };
        if let Some(scheme) = resolved_scheme {
            // One instantiation, all quantifier renamings. chelis#1801 needs
            // the dimension pairing on EVERY reference, not only an in-group
            // one, so the reference instantiates through the single mechanism
            // and takes what it needs from the result;
            // `env::tests::dvar_mapping_instantiation_matches_plain_instantiation`
            // pins that this is the same instantiation the two projections
            // used to perform.
            //
            // A reference to a sibling, a member of the recursive group being
            // inferred other than the declaration that holds the reference,
            // also copies every variable of the member's provisional type that
            // the scheme does not quantify, so it observes nothing a sibling's
            // body has determined ([04-INF-5]); the group's completion links
            // the copies (`group_link::sibling_instance`).
            if !called
                && !env.is_lexically_bound(name)
                && let Some(rejected) = reject_builtin_applicable_only_by_name(name, &scheme, node)
            {
                return report(errors, rejected);
            }
            let holed = env.is_holed_group_reference(name, &scheme);
            let unsigned = !holed && product.is_unsigned_group_reference(name, &scheme);
            let sibling = unsigned || (holed && product.references_a_sibling(name));
            // Unsigned self-references also wait for component linking, so a
            // concrete recursive argument cannot erase the body's raw input
            // identities before result origins have been captured.
            let (instantiated, copies) = if sibling {
                super::group_link::sibling_instance(&scheme, env, vg, subst)
            } else {
                (env.instantiate_scheme(&scheme, vg, subst), Vec::new())
            };
            product.import_result_constraints(subst);
            if !called {
                forbid_keys_a_builtin_value_does_not_admit(name, &instantiated.ty, subst);
            }
            // chelis#1801: the application rule reads these back to decide
            // which of THIS call's fresh dimension variables denote a
            // runtime extent they met (spec/04-type-system.md section 3.2).
            product.record_instantiation_dvars(
                instantiated
                    .dvars
                    .iter()
                    .filter(|(quantified, _)| scheme.dvars.contains(quantified))
                    .map(|(_, fresh)| *fresh),
            );
            if holed || sibling {
                // chelis#2590: an in-group reference to a member whose header
                // omits a type is decided against the member's body when its
                // component completes (`group_link`), and a sibling
                // reference's copies are linked then. Until then its instance
                // belongs to the group's level, so a `let` inside a member
                // cannot generalize over it and escape that decision.
                if let Some(level) = env.group_level() {
                    subst.lower_type_to_level(&instantiated.ty, level);
                }
                let span_id = node_span_id(node).map(str::to_string);
                let span_offset = span_id.as_deref().and_then(parse_span_offset);
                if holed {
                    product.record_group_reference(
                        name,
                        instantiated.ty.clone(),
                        span_id.clone(),
                        span_offset,
                    );
                }
                if sibling {
                    super::recursion::pin(copies.iter().filter_map(|(_, copy)| match *copy {
                        super::group_link::Variable::Type(var) => Some(var),
                        _ => None,
                    }));
                    product.record_sibling_link(
                        name,
                        instantiated.ty.clone(),
                        scheme.body.clone(),
                        copies,
                        unsigned,
                        span_id,
                        span_offset,
                    );
                }
            }
            if super::recursion::should_record_occurrence(name, &scheme) {
                // spec/04 section 3.1.1: inside a recursive binding group,
                // record the instantiation minted for an in-group reference
                // so the group can be validated for uniform recursive
                // instantiation. A sibling reference's copies are linked to
                // the member's own variables, not instantiated.
                let span_id = node_span_id(node).map(str::to_string);
                let span_offset = span_id.as_deref().and_then(parse_span_offset);
                let quantified = instantiated
                    .tvars
                    .iter()
                    .filter(|(quantified, _)| scheme.tvars.contains(quantified))
                    .cloned()
                    .collect::<Vec<_>>();
                super::recursion::record_occurrence(name, &quantified, span_id, span_offset);
            }
            let resolved = subst.apply(&instantiated.ty);
            // RFC D-CHECK: a bare reference to an out-of-module
            // opaque constructor is hidden, and an out-of-module
            // reference to an unexported binding whose signature
            // mentions an opaque type is the sixth rejection. Both
            // return the true type so no error cascades.
            if !crate::opacity::check_ctor_reference(name, adt_reg, errors) {
                crate::opacity::check_unexported_reference(name, &resolved, adt_reg, errors);
            }
            resolved
        } else {
            let mut err = CheckError::new(
                CheckErrorKind::UnboundVariable {
                    identifier: name.to_string(),
                },
                with_node_provenance(node, format!("unbound variable: {name}")),
                vec![format!("Check spelling of '{}'", name)],
            );
            if let Some(sid) = node_span_id(node) {
                if let Some(off) = parse_span_offset(sid) {
                    err.span_offset = Some(off);
                }
                err.span_id = Some(sid.to_string());
            }
            report(errors, err)
        }
    } else {
        malformed_form(node, "var", "a symbol name as its first child", errors)
    }
}

pub(super) fn infer_lit(
    node: &DeepNode,
    env: &Env,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    type_metadata_resolution: Option<&mut Option<OwnedTypeMetadataResolution>>,
) -> Type {
    let meta = node.meta();
    let kids = node.children_slice();

    // D1 (WS-A0 RT-1 fixup): per spec/04-type-system.md §5.3 last
    // paragraph, the lexer parses unsuffixed integer literals at i64
    // so that out-of-range literals can be diagnosed before the
    // i32 narrowing. The desugarer attaches `type: i32` ahead of
    // type-check (because §5.3 declares i32 as the default), so
    // here we check whether the underlying i64 value actually fits in
    // i32. If it doesn't, emit the §5.3 diagnostic before defaulting
    // — silently wrapping to a negative i32 is the bug §5.3 was
    // written to prevent.
    //
    // RT-2 fixup B2/B3: extend the same range check to i8 and
    // i16 contextual positions (spec §5.6 / §P10b). When the
    // contextual tensor-literal rule (chelis-surf desugar) emits a
    // `(lit {type: (t-prim {} i8)} N)` for an `xs: tensor[N, i8]
    // = [..., 200]` source, the underlying i64 value (200) overflows
    // i8 (range [-128, 127]) and silently wraps to -56 if not
    // diagnosed here. Mirror the i32 check for the i8 and i16 rows.
    let value_atom = kids.first();
    // chelis#1125 PP7 / [04-TOT-5]: an `Expr::List`-only destructure of the
    // `type:` metadata VALUE selected no range-check row at all on the stamped
    // ingress, and `(lit {type: (t-prim {} i8)} 200)` was accepted by
    // `check_typed_program` while `check_ir_program` rejected it.
    let meta_prim_name = meta
        .ty()
        .and_then(|ty| match stamped_parts(ty.expression()) {
            Some((DeepTag::TPrim, _, prim_kids)) => prim_kids.first().and_then(symbol_name),
            _ => None,
        });
    let integer_source_marker = meta.literal_source().is_some();
    if let Some(prim_name) = meta_prim_name
        && let Some(deep::Expr::Atom(deep::Atom::Int(n), _)) = value_atom
    {
        // Per-prim range check. Only apply to integer prims; the float
        // contextual cases admit the i64 directly (Surf's float lexer
        // produces a Float atom; an Int atom in a float context is
        // either an error caught elsewhere or a Cons-mismatch).
        let range_check = match prim_name {
            "i8" => Some(("i8", i8::MIN as i64, i8::MAX as i64)),
            "i16" => Some(("i16", i16::MIN as i64, i16::MAX as i64)),
            "i32" => Some(("i32", i32::MIN as i64, i32::MAX as i64)),
            // i64 cannot overflow an i64 atom; bool/string don't
            // accept Int atoms.
            _ => None,
        };
        if let Some((dtype, lo, hi)) = range_check
            && (*n < lo || *n > hi)
        {
            // The i32 default path keeps the WS-A0 D1 message
            // shape (i64 suffix + cast(_, i64) hint) so existing
            // diagnostics-pinning tests stay green; the i8/i16
            // contextual paths cite §5.6 + §5.3 because the
            // narrowing came from contextual inference, not the
            // default. The cast hint spells the prec type name
            // `i64` (§1.1) — `i64` is only the literal-suffix
            // spelling (§5.5) and is not a valid `cast` target, so
            // recommending `cast({n}, i64)` would send the user to a
            // form that re-fires this same diagnostic (issue #308
            // review fix).
            if dtype == "i32" {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "literal {n} out of range for default i32; use the `i64` \
                         suffix (`{n}i64`) or an explicit cast({n}, i64) \
                         (spec/04-type-system.md §5.3, §5.5)"
                    ),
                    vec![
                        "spec/04-type-system.md §5.3: integer literals default to i32; \
                         the lexer parses at i64 so out-of-range tokens can be diagnosed \
                         before the narrowing rather than wrapping silently"
                            .to_string(),
                    ],
                ));
            } else {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "literal {n} out of range for context-inferred {dtype} \
                         [{lo}, {hi}]; use a wider integer type or an explicit \
                         cast (spec/04-type-system.md §5.6, §5.3)"
                    ),
                    vec![format!(
                        "spec/04-type-system.md §5.6 + §5.3: contextual tensor-literal \
                         element-type inference (§P10b) narrows unsuffixed integer \
                         literals to the declared element type ({dtype}). The lexer \
                         parses at i64 so values outside the {dtype} range \
                         [{lo}, {hi}] are diagnosed before the narrowing rather than \
                         wrapping silently"
                    )],
                ));
            }
            return vg.fresh_type();
        }
    }

    // Check metadata for type annotation
    if let Some(ty) = meta.ty() {
        let val = ty.expression();
        let resolved = match resolve_deep_type_with_diagnostic_owner(
            val,
            vg,
            adt_reg,
            TypeUseSite::Annotation,
            annotation_binder_mode(env),
            env.type_resolution_diagnostic_owner(),
            errors,
        ) {
            Ok(ty) => ty,
            Err(witness) => {
                if let Some(owner) = type_metadata_resolution {
                    *owner = Some(OwnedTypeMetadataResolution::Failed(witness));
                }
                return propagate(&witness);
            }
        };
        if let Some(owner) = type_metadata_resolution {
            *owner = Some(OwnedTypeMetadataResolution::Resolved(resolved.clone()));
        }
        // chelis#1131 / spec/03 §6.4: `lit` has a closed canonical
        // atom-to-primitive matrix. The sole cross-family form is an
        // exact Int payload carrying `literal_source: integer` and a
        // float target; it preserves one target-width rounding under
        // [04-NUM-1]/[04-NUM-14]. Enforce the matrix after transparent
        // aliases resolve so every Deep ingress gets one decision.
        if let Type::Prim(prim) = &resolved {
            let atom_family = value_atom.and_then(|value| match value {
                deep::Expr::Atom(deep::Atom::Int(_), _) => Some("integer"),
                deep::Expr::Atom(deep::Atom::Float(_), _) => Some("floating-point"),
                deep::Expr::Atom(deep::Atom::Bool(_), _) => Some("boolean"),
                deep::Expr::Atom(deep::Atom::Str(_), _) => Some("string"),
                _ => None,
            });
            let integer_spelled_float = integer_source_marker
                && prim.is_float()
                && matches!(value_atom, Some(deep::Expr::Atom(deep::Atom::Int(_), _)));
            if integer_source_marker && !integer_spelled_float {
                return report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                "literal_source: integer requires an Int atom and a float primitive type"
                                    .to_string(),
                                vec![
                                    "Remove the marker or emit the canonical integer-spelled float literal form from spec/03-deep-syntax.md §6.4"
                                        .to_string(),
                                ],
                            ),
                        );
            }
            let canonical_pair = integer_spelled_float
                || match value_atom {
                    Some(deep::Expr::Atom(deep::Atom::Int(_), _)) => prim.is_integer(),
                    Some(deep::Expr::Atom(deep::Atom::Float(_), _)) => prim.is_float(),
                    Some(deep::Expr::Atom(deep::Atom::Bool(_), _)) => *prim == Prim::Bool,
                    Some(deep::Expr::Atom(deep::Atom::Str(_), _)) => *prim == Prim::String,
                    _ => false,
                };
            if !canonical_pair {
                let family = atom_family.unwrap_or("non-scalar");
                return report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!(
                            "{family} atom cannot carry `{resolved}` literal metadata: \
                                     Deep literals require the canonical atom/primitive pairing \
                                     (integer→integer, float→float, boolean→bool, \
                                     string→string), or the explicitly marked integer-spelled \
                                     float form (spec/03-deep-syntax.md §6.4)"
                        ),
                        vec![
                            "Emit the atom kind that denotes the declared primitive \
                                     family; use `cast` for a value conversion rather than \
                                     contradictory literal metadata"
                                .to_string(),
                        ],
                    ),
                );
            }
            // [04-LIT-2]: the literal binds at this float primitive by one
            // finalization, and an infinity there is not a literal. The
            // resolved metadata is the same for a suffix, the §5.3 default,
            // every §5.6 position, and hand-written Deep, so this one check
            // covers each ingress.
            if let Some(deep::Expr::Atom(atom, _)) = value_atom
                && super::literal_width::literal_is_non_finite_at(*prim, atom)
            {
                return report(
                    errors,
                    super::literal_width::non_finite_literal_error(atom, *prim),
                );
            }
        } else if integer_source_marker {
            return report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    "literal_source: integer requires a primitive float type".to_string(),
                    vec!["Remove the marker from non-primitive literal metadata".to_string()],
                ),
            );
        }
        // RFC D-CHECK lit-forge gate: `{type: (t-adt ...)}`
        // metadata on a Deep literal outside the defining module
        // forges an opaque value. Surf's separate checking boundary
        // retains the same diagnostic via check_opaque_literal_ascription.
        // `resolve_deep_type` expands transparent
        // aliases, so `0.5 : P2` cannot launder the gate.
        if let Type::Adt(adt_name, _) | Type::KindedAdt(adt_name, _) = &resolved {
            crate::opacity::check_opaque_use(
                crate::opacity::OpaqueAction::LitForge,
                adt_name,
                adt_reg,
                errors,
            );
        }
        return resolved;
    }

    if integer_source_marker {
        return report(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                "literal_source: integer requires explicit float type metadata".to_string(),
                vec![
                    "Add the producer-selected float primitive type or remove the marker"
                        .to_string(),
                ],
            ),
        );
    }

    // Fall back to value-based defaults
    if let Some(val) = kids.first() {
        match val {
            deep::Expr::Atom(deep::Atom::Int(n), _) => {
                // D1 (WS-A0 RT-1 fixup): same check as the metadata
                // path above but for Deep producers that omit the
                // explicit `type: i32` ascription on a `(lit {} N)`
                // form. Without this guard the bare-form path would
                // silently default to i32 and wrap.
                if i32::try_from(*n).is_err() {
                    report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!(
                                "literal {n} out of range for default i32; use the \
                             `{n}i64` literal suffix or an explicit cast({n}, i64) \
                             (spec/04-type-system.md §5.3, §5.5)"
                            ),
                            vec![
                                "spec/04-type-system.md §5.3: integer literals default \
                             to i32; the lexer parses at i64 so out-of-range \
                             tokens can be diagnosed before the narrowing rather \
                             than wrapping silently"
                                    .to_string(),
                            ],
                        ),
                    )
                } else {
                    Type::Prim(Prim::Int32)
                }
            }
            deep::Expr::Atom(atom @ deep::Atom::Float(_), _) => {
                // [04-LIT-2] at the §5.3 default, for Deep producers that
                // omit the `type: f32` ascription.
                if super::literal_width::literal_is_non_finite_at(Prim::F32, atom) {
                    report(
                        errors,
                        super::literal_width::non_finite_literal_error(atom, Prim::F32),
                    )
                } else {
                    Type::Prim(Prim::F32)
                }
            }
            deep::Expr::Atom(deep::Atom::Bool(_), _) => Type::Prim(Prim::Bool),
            deep::Expr::Atom(deep::Atom::Str(_), _) => Type::Prim(Prim::String),
            _ => malformed_form(node, "lit", "a scalar atom value", errors),
        }
    } else {
        malformed_form(node, "lit", "a value atom or a `type:` annotation", errors)
    }
}
