//! Expression dispatch and common expression helpers.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

/// Exact ownership record for a root expression consumer that resolves its
/// own `type:` metadata. `infer_let` uses this instead of inferring ownership
/// from a generic `Type::Error`: an unrelated RHS failure must not suppress an
/// independently malformed let ascription.
pub(super) enum OwnedTypeMetadataResolution {
    Resolved(Type),
    Failed(ErrorWitness),
}

/// The whole of `copy`'s inference, in one place.
///
/// Both `DeepTag::Copy` arms were byte-identical apart from how they spelled
/// the list, so they were two chances to fix a bug once (chelis#1489).
fn infer_copy(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
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
                        },
                    );
                    result
                }
                _ => report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        format!("copy requires tensor input, got {resolved}"),
                        vec!["Wrap only tensor values in copy".to_string()],
                    ),
                ),
            },
        }
    } else {
        malformed_form(list, "copy", "one wrapped expression", errors)
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
        .map(|names| BinderMode::Lexical(names, env.type_resolution_variables()))
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
        expr, env, vg, subst, adt_reg, errors, product, None, None,
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
    )
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

    let result = match expr {
        deep::Expr::Atom(atom, _) => infer_atom(atom, errors),
        deep::Expr::List(list, _) => {
            // chelis#731 Phase 3 (checker_totality.md §C4.2): dispatch on
            // the typed closed vocabulary. The serialized form and the
            // in-memory AST stay frozen; the enum is derived from the tag
            // string here, at the chokepoint. The match is exhaustive with
            // no `_` arm, so a 63rd `DeepTag` variant fails to compile
            // until this dispatch chooses its disposition.
            match get_tag(list) {
                Some(DeepTag::Var) => infer_var(list, env, vg, subst, adt_reg, errors),
                Some(DeepTag::Lit) => {
                    infer_lit(list, env, vg, adt_reg, errors, type_metadata_resolution)
                }
                Some(DeepTag::App) => infer_app(
                    list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    product,
                    expected_result,
                ),
                Some(DeepTag::Fn) => infer_fn(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::Let) => infer_let(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::If) => infer_if(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::Match) => infer_match(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::Pipe) => infer_pipe(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::Tuple) => infer_tuple(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::TupleGet) => {
                    infer_tuple_get(list, env, vg, subst, adt_reg, errors, product)
                }
                Some(DeepTag::Record) => {
                    infer_record(list, env, vg, subst, adt_reg, errors, product)
                }
                Some(DeepTag::Access) => {
                    infer_access(list, env, vg, subst, adt_reg, errors, product)
                }
                Some(DeepTag::RecordUpdate) => {
                    infer_record_update(list, env, vg, subst, adt_reg, errors, product)
                }
                Some(DeepTag::Cast) => infer_cast(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::Grad) => infer_grad(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::Vmap) => infer_vmap(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::Def) => infer_def(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::Defsig) => {
                    // Already handled in first pass
                    Type::Unit
                }
                Some(DeepTag::Deftype | DeepTag::Typealias) => {
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
                Some(
                    DeepTag::Module
                    | DeepTag::Import
                    | DeepTag::ImportAll
                    | DeepTag::Export
                    | DeepTag::Defdim,
                ) => Type::Unit,
                Some(DeepTag::Block) => {
                    // chelis#859: sequenced expressions; the value (and
                    // type) is the last child's (spec/03 §2.3). Every child
                    // is checked in order so non-last children keep their
                    // own diagnostics. A childless block has no value and
                    // is malformed ([04-TOT-3]); lowering raises on the
                    // same shape.
                    let kids = children(list);
                    if kids.is_empty() {
                        malformed_form(list, "block", "at least one child expression", errors)
                    } else {
                        let mut last_ty = Type::Unit;
                        for kid in kids {
                            last_ty = infer_expr(kid, env, vg, subst, adt_reg, errors, product);
                        }
                        last_ty
                    }
                }
                Some(DeepTag::Par) => {
                    // par: evaluate all children, return type of last (v1: sequential)
                    let kids = children(list);
                    let mut last_ty = Type::Unit;
                    for kid in kids {
                        last_ty = infer_expr(kid, env, vg, subst, adt_reg, errors, product);
                    }
                    last_ty
                }
                Some(DeepTag::Jit) => {
                    // jit: compilation trigger; semantically a no-op at eval
                    // (spec/03-deep-syntax.md §2.7). Type is the type of the
                    // wrapped expression.
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        infer_expr(inner, env, vg, subst, adt_reg, errors, product)
                    } else {
                        malformed_form(list, "jit", "one wrapped expression", errors)
                    }
                }
                Some(DeepTag::Realize) => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        infer_expr(inner, env, vg, subst, adt_reg, errors, product)
                    } else {
                        malformed_form(list, "realize", "one wrapped expression", errors)
                    }
                }
                Some(DeepTag::Copy) => infer_copy(list, env, vg, subst, adt_reg, errors, product),
                Some(DeepTag::Borrow) => {
                    let kids = children(list);
                    if let Some(inner) = kids.first() {
                        let inner_ty = infer_expr(inner, env, vg, subst, adt_reg, errors, product);
                        let resolved = subst.apply(&inner_ty);
                        match resolved {
                            Type::Ref(_) => resolved,
                            Type::Tensor(_, _)
                            | Type::Adt(_, _)
                            | Type::KindedAdt(_, _)
                            | Type::Tuple(_)
                            | Type::Error(_) => Type::Ref(Box::new(resolved)),
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
                            _ => report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    format!(
                                        "borrow requires tensor or tensor-carrying input, got {resolved}"
                                    ),
                                    vec!["Use `&x` only with tensor values".to_string()],
                                ),
                            ),
                        }
                    } else {
                        malformed_form(list, "borrow", "one wrapped expression", errors)
                    }
                }
                Some(DeepTag::HandleEffect) => {
                    infer_handle_effect(list, env, vg, subst, adt_reg, errors, product)
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
                Some(
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
                    | DeepTag::Resource),
                ) => {
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
                None => {
                    // chelis#731 [04-TOT-1] / §C1.2: the raw-string entry
                    // boundary. The parser already screens the 62-tag closed
                    // vocabulary, so a string that does not decode here came
                    // from input that never crossed the parser (programmatic
                    // Deep construction) or from version skew - never
                    // ordinary parsed input. Reject it loudly rather than
                    // returning a silent `Type::Error` that would exempt the
                    // whole subtree from checking (the chelis#709 class
                    // defect: the default must be to fail).
                    let named = list.unknown_tag_symbol().unwrap_or("<none>");
                    report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::UnknownForm,
                            format!(
                                "unknown Deep tag `{named}` has no checker disposition \
                             (not in the 62-tag closed vocabulary of \
                             spec/03-deep-syntax.md; chelis#731 [04-TOT-1])"
                            ),
                            vec![],
                        ),
                    )
                }
            }
        }
        deep::Expr::Map(_, _) => Type::Unit,
        deep::Expr::MetaExpr(meta, _) => {
            infer_expr(&meta.expr, env, vg, subst, adt_reg, errors, product)
        }
        // Bridge: reconstruct List so tag-dispatch functions work unchanged (#908)
        // `Node` is a stamped vocabulary node — dispatch like `List` using its tag.
        deep::Expr::Node(node, span) => {
            // Transitional bridge (chelis#998 → consumer migration):
            // reconstruct the List representation so the existing tag-dispatch
            // functions (`infer_var`, `infer_app`, etc.) work unchanged. Once
            // those functions are migrated to accept Node directly, this
            // `to_list` call becomes dead code.
            let list = node.to_list(*span);
            product.register_bridge_children(node.children_slice(), children(&list));
            match node.tag() {
                DeepTag::Var => infer_var(&list, env, vg, subst, adt_reg, errors),
                DeepTag::Lit => {
                    infer_lit(&list, env, vg, adt_reg, errors, type_metadata_resolution)
                }
                DeepTag::App => infer_app(
                    &list,
                    env,
                    vg,
                    subst,
                    adt_reg,
                    errors,
                    product,
                    expected_result,
                ),
                DeepTag::Fn => infer_fn(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Let => infer_let(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::If => infer_if(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Match => infer_match(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Pipe => infer_pipe(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Tuple => infer_tuple(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::TupleGet => {
                    infer_tuple_get(&list, env, vg, subst, adt_reg, errors, product)
                }
                DeepTag::Record => infer_record(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Access => infer_access(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::RecordUpdate => {
                    infer_record_update(&list, env, vg, subst, adt_reg, errors, product)
                }
                DeepTag::Cast => infer_cast(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Grad => infer_grad(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Vmap => infer_vmap(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Def => infer_def(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Defsig => Type::Unit,
                DeepTag::Deftype | DeepTag::Typealias => Type::Unit,
                DeepTag::Module
                | DeepTag::Import
                | DeepTag::ImportAll
                | DeepTag::Export
                | DeepTag::Defdim => Type::Unit,
                DeepTag::Block => {
                    let kids = children(&list);
                    if kids.is_empty() {
                        malformed_form(&list, "block", "at least one child expression", errors)
                    } else {
                        let mut last_ty = Type::Unit;
                        for kid in kids {
                            last_ty = infer_expr(kid, env, vg, subst, adt_reg, errors, product);
                        }
                        last_ty
                    }
                }
                DeepTag::Par => {
                    let kids = children(&list);
                    let mut last_ty = Type::Unit;
                    for kid in kids {
                        last_ty = infer_expr(kid, env, vg, subst, adt_reg, errors, product);
                    }
                    last_ty
                }
                DeepTag::Jit => {
                    let kids = children(&list);
                    if let Some(inner) = kids.first() {
                        infer_expr(inner, env, vg, subst, adt_reg, errors, product)
                    } else {
                        malformed_form(&list, "jit", "one wrapped expression", errors)
                    }
                }
                DeepTag::Realize => {
                    let kids = children(&list);
                    if let Some(inner) = kids.first() {
                        infer_expr(inner, env, vg, subst, adt_reg, errors, product)
                    } else {
                        malformed_form(&list, "realize", "one wrapped expression", errors)
                    }
                }
                DeepTag::Copy => infer_copy(&list, env, vg, subst, adt_reg, errors, product),
                DeepTag::Borrow => {
                    let kids = children(&list);
                    if let Some(inner) = kids.first() {
                        let inner_ty = infer_expr(inner, env, vg, subst, adt_reg, errors, product);
                        let resolved = subst.apply(&inner_ty);
                        match resolved {
                            Type::Ref(_) => resolved,
                            Type::Tensor(_, _)
                            | Type::Adt(_, _)
                            | Type::KindedAdt(_, _)
                            | Type::Tuple(_)
                            | Type::Error(_) => Type::Ref(Box::new(resolved)),
                            Type::Var(tv) => {
                                subst.record_deferred_borrow_var(tv);
                                Type::Ref(Box::new(Type::Var(tv)))
                            }
                            _ => report(
                                errors,
                                CheckError::new(
                                    CheckErrorKind::TypeMismatch,
                                    format!(
                                        "borrow requires tensor or tensor-carrying input, got {resolved}"
                                    ),
                                    vec!["Use `&x` only with tensor values".to_string()],
                                ),
                            ),
                        }
                    } else {
                        malformed_form(&list, "borrow", "one wrapped expression", errors)
                    }
                }
                DeepTag::HandleEffect => {
                    infer_handle_effect(&list, env, vg, subst, adt_reg, errors, product)
                }
                DeepTag::Variant
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
                | DeepTag::Resource => {
                    let named = node.tag().as_str();
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
        // loudly because it normalizes `BareList` into a tagless
        // `Expr::List` first and lands on the `None` arm below. Same
        // program, two verdicts, and the permissive one scored a form with
        // no honest type as checked (the chelis#873-shape fail-open).
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

    if !matches!(result, Type::Error(_)) {
        product.typed_nodes += 1;
    }
    product.record_canonical(expr, result.clone());

    result
}

/// chelis#709 / spec/design/checker_totality.md §C1.5: the `handle-effect`
/// checker case. Before this, `handle-effect` fell through `infer_expr`'s
/// wildcard to a silent `Type::Error`, so `with seed` / `with device` bodies
/// were not type-checked at all and the enclosing `def`'s declared return
/// type went unenforced.
///
/// The typing SHAPE mirrors LaCaDiLE's T-Handle rule (Jeff's 2026-07 note on
/// chelis#709): type the body in the enclosing context and return the BODY's
/// type, so the enclosing signature is enforced against it. The handler is
/// checked per effect kind. chelis#730's `EffectKind` enum does not exist yet,
/// so the two known kinds are string-matched with a loud `MalformedForm` else
/// (the pinned §I1 interlock: migrate to the enum when it lands).
///
/// Handler-form rules (open question 1, decided 2026-07-17; explicit over
/// implicit, since this code is agent-written):
/// * `random` (`with seed`): the seed is semantically int64. A seed written as
///   an integer LITERAL must carry the `i64` suffix; an unsuffixed literal is a
///   `TypeMismatch` naming the suffix (this is the reject-diagnostic half left
///   to chelis#731 Phase 1 by chelis#771, unblocking the parked cross-lane RNG
///   atom on chelis#735). Non-literal seed expressions stay the shared
///   front-end effects gate's responsibility (`chelis-effects`
///   `validate_handler_expr`: "requires an int literal seed", spec/02 §P5), so
///   they are not re-reported here.
/// * `resource` (`with device`): the device is a string literal, checked by the
///   same effects gate; the device-name vocabulary is not validated here
///   (target knowledge, chelis#735). Checking the expression bounds the handler.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_handle_effect(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    let kids = children(list);
    // Structural arity: spec/03-deep-syntax.md gives `handle-effect` EXACTLY
    // two children -- `(handle-effect {effect: K} <handler> <body>)`. Any other
    // count is malformed Deep; reject loudly ([04-TOT-3]). A THIRD child in
    // particular carries an ill-typed subtree the checker would never visit
    // (chelis#731 red team) -- the chelis#710 silent-exemption class -- so
    // tolerating `>= 2` would be a fitness-honesty hole.
    if kids.len() != 2 {
        return malformed_form(
            list,
            "handle-effect",
            "exactly two children (a handler expression and a body)",
            errors,
        );
    }
    let handler = &kids[0];
    let body = &kids[1];

    // The handler's LITERAL-ness (an int literal seed, a string literal device)
    // is enforced by the shared front-end effects gate (`chelis-effects`
    // `validate_handler_expr`, spec/02 §P5), which runs in check, build, and
    // eval. The checker does not type-infer the handler expression here: doing
    // so would surface handler-internal diagnostics (e.g. an out-of-range inner
    // literal in `with seed(-2147483649)`) that preempt the effects gate's
    // "requires an int literal seed" message, and the effects gate already
    // rejects every non-literal form loudly and identically across lanes. The
    // checker's only handler-side addition is the int64-suffix rule below.
    //
    // chelis#730 Phase 2 (section C4.4; the pinned §I1 interlock): the kind
    // is parsed once into the closed [`EffectKind`] set - the same enum the
    // lowering lanes now use - and dispatched with an exhaustive `match`
    // over `EffectKind` (no wildcard arm). Adding a kind is a compile error
    // here until this checker case handles it. Decode failures preserve the
    // missing/malformed/unknown distinction from the shared Deep adapter.
    match decode_effect_kind(list) {
        Ok(EffectKind::Random) => {
            // Open question 1 (decided 2026-07-17): the seed is semantically
            // int64, and a seed written as an integer LITERAL must carry the
            // `i64` suffix (the reject-diagnostic half chelis#771 left to Phase
            // 1). A negative int64 literal is additionally rejected. That is a
            // deliberate narrowing of the accepted FRONT-END surface, held until
            // chelis#735 authors the `with seed` contract, and not a lane
            // limitation: both the DAG lowering (`extract_u64_value`,
            // chelis#794) and the evaluator reinterpret a signed int64 seed as
            // its uint64 two's-complement bits per [05-RNG-1] (chelis#731 red
            // team F2).
            match seed_literal_form(handler) {
                SeedLiteralForm::Unsuffixed => {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        "`with seed(...)` requires an int64-suffixed integer literal seed; \
                         an unsuffixed literal defaults to int32 (spec/02-surf-syntax.md \
                         §P10a; spec/design/checker_totality.md §C1.5)"
                            .to_string(),
                        vec![
                            "Add the `i64` suffix to the seed literal, e.g. \
                             `with seed(42i64) { ... }`"
                                .to_string(),
                        ],
                    ));
                }
                SeedLiteralForm::NegativeInt64 => {
                    errors.push(CheckError::new(
                        CheckErrorKind::TypeMismatch,
                        "`with seed(...)` requires a non-negative seed literal. This \
                         is a deliberate narrowing of the accepted front-end surface, \
                         held until chelis#735 authors the `with seed` contract, not a \
                         lane limitation: the DAG and eval lowerings both reinterpret a \
                         signed int64 seed as its uint64 two's-complement bits per \
                         [05-RNG-1] (spec/design/checker_totality.md §C1.5)"
                            .to_string(),
                        vec![
                            "Use a non-negative int64-suffixed seed, e.g. \
                             `with seed(42i64) { ... }`"
                                .to_string(),
                        ],
                    ));
                }
                SeedLiteralForm::NotIntLiteral | SeedLiteralForm::ValidInt64 => {}
            }
        }
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
                     recognizes only `random` (with seed) and `resource` \
                     (with device) (spec/03-deep-syntax.md; chelis#730, chelis#731)"
                ),
                vec![],
            ));
        }
    }

    // T-Handle: return the BODY's type in the enclosing context. This is the
    // chelis#709 fix -- the enclosing `def` signature is now enforced against
    // the body, so an int64 body in an `-> f32` def, or a tensor body from a
    // scalar-typed fn, is a type error caught before any backend sees it.
    infer_expr(body, env, vg, subst, adt_reg, errors, product)
}

/// Classification of a `random`-effect seed handler for the checker's
/// seed-form rules (chelis#731 §C1.5; the negative case is the chelis#731 red
/// team's F2 finding).
pub(super) enum SeedLiteralForm {
    /// Not an integer literal (a computed expression, a string, ...). This is
    /// the shared effects gate's territory (`chelis-effects`
    /// `validate_handler_expr`); the checker stays silent to avoid a double
    /// diagnostic.
    NotIntLiteral,
    /// An unsuffixed integer literal (a bare `Atom::Int`, or `(lit {type:
    /// int32} N)`). The seed is semantically int64, so this is a type error.
    Unsuffixed,
    /// An int64-suffixed but NEGATIVE literal (`(lit {type: int64} -N)`).
    /// [05-RNG-1] already fixes its meaning: reinterpret the signed int64 seed
    /// as its uint64 two's-complement bits. The DAG lane does that in
    /// `extract_u64_value` (chelis#794, which replaced the old
    /// `extract_usize_value` fold to seed 0) and the evaluator already did.
    /// The rejection here is therefore a deliberate narrowing of the
    /// accepted front-end surface, held until chelis#735 authors the
    /// `with seed` contract, not a lane limitation.
    NegativeInt64,
    /// A valid non-negative int64-suffixed literal (`Ni64` desugars to
    /// `(lit {type: (t-prim {} int64)} N)`, N >= 0). Accepted.
    ValidInt64,
}

/// Classify a `random`-effect seed handler. Only integer literals are the
/// checker's business (the effects gate covers literal-ness); a bare atom is
/// unsuffixed by construction, a `(lit ...)` carries its width in the `type`
/// metadata.
pub(super) fn seed_literal_form(expr: &deep::Expr) -> SeedLiteralForm {
    // chelis#1125 PP7 / [04-TOT-5]: both reads below are carrier-preserving.
    // This function had the `infer_lit` defect twice over -- an
    // `Expr::List`-only match on the seed `lit` itself, and an
    // `Expr::List`-only read of the `t-prim` under its `type:` metadata -- so
    // on the stamped ingress the handler `Expr::Node` fell straight to the
    // default arm, the seed classified as `NotIntLiteral`, and the §P10a
    // int64-suffix rejection never fired at all.
    let int_lit = match expr {
        // A bare integer atom has no suffix metadata: unsuffixed by construction.
        deep::Expr::Atom(deep::Atom::Int(value), _) => Some((false, *value)),
        _ => match stamped_parts(expr) {
            Some((DeepTag::Lit, meta, lit_kids)) => match lit_kids.first() {
                Some(deep::Expr::Atom(deep::Atom::Int(value), _)) => {
                    let is_int64 = meta.ty().is_some_and(|ty| matches!(stamped_parts(ty.expression()), Some((DeepTag::TPrim, _, prim_kids)) if prim_kids.first().and_then(symbol_name) == Some("int64")));
                    Some((is_int64, *value))
                }
                // A `(lit ...)` wrapping a non-int value is not an int seed.
                _ => None,
            },
            _ => None,
        },
    };
    match int_lit {
        None => SeedLiteralForm::NotIntLiteral,
        Some((false, _)) => SeedLiteralForm::Unsuffixed,
        Some((true, value)) if value < 0 => SeedLiteralForm::NegativeInt64,
        Some((true, _)) => SeedLiteralForm::ValidInt64,
    }
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
/// designates as the training signal. The rule already existed one lane down
/// in `chelis_ir::lower::lower_atom`, whose diagnostic states the law this
/// site was breaking: unhandled or malformed forms cannot become Unit or
/// another value ([05-UNS-1]; chelis#730). Reporting here says the same thing
/// at the lane that is the oracle, and re-types the site `Type::Error` so it
/// sits back under the Phase 2 `ErrorWitness` guard (chelis#731 §C3).
pub(super) fn infer_atom(atom: &deep::Atom, errors: &mut DiagnosticSink<'_>) -> Type {
    match atom {
        // D1 (WS-A0 RT-1 fixup): per spec/04-type-system.md §5.3 the
        // lexer parses unsuffixed integer tokens at i64 so that
        // out-of-range literals can be diagnosed before the int32
        // narrowing. The bare-atom path is the value-only fallback for
        // Deep code that bypasses the desugarer's `(lit {type: int32}
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
        deep::Atom::Tag(tag) => report(
            errors,
            CheckError::new(
                CheckErrorKind::MalformedForm,
                format!(
                    "a decoded tag atom `{}` outside a list's tag position is structural \
                     syntax, not an expression, and cannot be typed or lowered to the \
                     executable IR (spec/design/loud_unsupported.md section C1.4; \
                     chelis#710 form 4)",
                    tag.as_str()
                ),
                vec![format!(
                    "`{}` names a form; write `({} {{}} ...)` to use it as one",
                    tag.as_str(),
                    tag.as_str()
                )],
            ),
        ),
    }
}

pub(super) fn infer_var(
    list: &deep::List,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    let kids = children(list);
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
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("unknown constructor: {name}"),
                ),
                vec![format!(
                    "Constructor '{name}' is not in scope. Declare it locally or add it \
                     to an import (e.g. `import Mod ({name})`)"
                )],
            );
            if let Some(sid) = list_span_id(list) {
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
            // spec/04 §3.1.1: inside a recursive binding group, record the
            // instantiation minted for an in-group reference so the group
            // can be validated for uniform recursive instantiation.
            let ty = if super::recursion::should_record_occurrence(name, &scheme) {
                let (ty, mapping) = env.instantiate_with_tvar_mapping(&scheme, vg, subst);
                let span_id = list_span_id(list).map(str::to_string);
                let span_offset = span_id.as_deref().and_then(parse_span_offset);
                super::recursion::record_occurrence(name, &mapping, span_id, span_offset);
                ty
            } else {
                env.instantiate(&scheme, vg, subst)
            };
            let resolved = subst.apply(&ty);
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
                with_macro_provenance(
                    &deep::Expr::List(list.clone(), zero_span()),
                    format!("unbound variable: {name}"),
                ),
                vec![format!("Check spelling of '{}'", name)],
            );
            if let Some(sid) = list_span_id(list) {
                if let Some(off) = parse_span_offset(sid) {
                    err.span_offset = Some(off);
                }
                err.span_id = Some(sid.to_string());
            }
            report(errors, err)
        }
    } else {
        malformed_form(list, "var", "a symbol name as its first child", errors)
    }
}

pub(super) fn infer_lit(
    list: &deep::List,
    env: &Env,
    vg: &mut VarGen,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    type_metadata_resolution: Option<&mut Option<OwnedTypeMetadataResolution>>,
) -> Type {
    let meta = get_meta(list);
    let kids = children(list);

    // D1 (WS-A0 RT-1 fixup): per spec/04-type-system.md §5.3 last
    // paragraph, the lexer parses unsuffixed integer literals at i64
    // so that out-of-range literals can be diagnosed before the
    // int32 narrowing. The desugarer attaches `type: int32` ahead of
    // type-check (because §5.3 declares int32 as the default), so
    // here we check whether the underlying i64 value actually fits in
    // i32. If it doesn't, emit the §5.3 diagnostic before defaulting
    // — silently wrapping to a negative i32 is the bug §5.3 was
    // written to prevent.
    //
    // RT-2 fixup B2/B3: extend the same range check to int8 and
    // int16 contextual positions (spec §5.6 / §P10b). When the
    // contextual tensor-literal rule (chelis-surf desugar) emits a
    // `(lit {type: (t-prim {} int8)} N)` for an `xs: tensor[N, int8]
    // = [..., 200]` source, the underlying i64 value (200) overflows
    // int8 (range [-128, 127]) and silently wraps to -56 if not
    // diagnosed here. Mirror the i32 check for the i8 and i16 rows.
    let value_atom = kids.first();
    // chelis#1125 PP7 / [04-TOT-5]: read the `type:` metadata VALUE through
    // the carrier-preserving `stamped_parts`. `Node::to_list` clones the
    // metadata map verbatim, so on the stamped ingress this value is still an
    // `Expr::Node` even though the enclosing `lit` arrived here as a rebuilt
    // `List`. The old `Expr::List`-only destructure therefore selected no
    // range-check row at all, and `(lit {type: (t-prim {} int8)} 200)` was
    // accepted by `check_typed_program` while `check_ir_program` rejected it.
    let meta_prim_name =
        meta.and_then(|m| m.ty())
            .and_then(|ty| match stamped_parts(ty.expression()) {
                Some((DeepTag::TPrim, _, prim_kids)) => prim_kids.first().and_then(symbol_name),
                _ => None,
            });
    let integer_source_marker = meta.and_then(|m| m.literal_source()).is_some();
    if let Some(prim_name) = meta_prim_name
        && let Some(deep::Expr::Atom(deep::Atom::Int(n), _)) = value_atom
    {
        // Per-prim range check. Only apply to integer prims; the float
        // contextual cases admit the i64 directly (Surf's float lexer
        // produces a Float atom; an Int atom in a float context is
        // either an error caught elsewhere or a Cons-mismatch).
        let range_check = match prim_name {
            "int8" => Some(("int8", i8::MIN as i64, i8::MAX as i64)),
            "int16" => Some(("int16", i16::MIN as i64, i16::MAX as i64)),
            "int32" => Some(("int32", i32::MIN as i64, i32::MAX as i64)),
            // int64 cannot overflow an i64 atom; bool/string don't
            // accept Int atoms.
            _ => None,
        };
        if let Some((dtype, lo, hi)) = range_check
            && (*n < lo || *n > hi)
        {
            // The int32 default path keeps the WS-A0 D1 message
            // shape (i64 suffix + cast(_, int64) hint) so existing
            // diagnostics-pinning tests stay green; the int8/int16
            // contextual paths cite §5.6 + §5.3 because the
            // narrowing came from contextual inference, not the
            // default. The cast hint spells the prec type name
            // `int64` (§1.1) — `i64` is only the literal-suffix
            // spelling (§5.5) and is not a valid `cast` target, so
            // recommending `cast({n}, i64)` would send the user to a
            // form that re-fires this same diagnostic (issue #308
            // review fix).
            if dtype == "int32" {
                errors.push(CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "literal {n} out of range for default int32; use the `i64` \
                         suffix (`{n}i64`) or an explicit cast({n}, int64) \
                         (spec/04-type-system.md §5.3, §5.5)"
                    ),
                    vec![
                        "spec/04-type-system.md §5.3: integer literals default to int32; \
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
    if let Some(ty) = meta.and_then(|m| m.ty()) {
        let val = ty.expression();
        let resolved = match resolve_deep_type(
            val,
            vg,
            adt_reg,
            TypeUseSite::Annotation,
            annotation_binder_mode(env),
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
        // metadata on a literal outside the defining module
        // forges an opaque value. Reachable from BOTH
        // surfaces: Surf expression ascription
        // (`0.5 : Probability`) and block-binding ascription
        // desugar to exactly this metadata (RT-0), so the
        // gate is not scoped to `.dp` ingestion.
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
                // explicit `type: int32` ascription on a `(lit {} N)`
                // form. Without this guard the bare-form path would
                // silently default to int32 and wrap.
                if i32::try_from(*n).is_err() {
                    report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!(
                                "literal {n} out of range for default int32; use the \
                             `{n}i64` literal suffix or an explicit cast({n}, int64) \
                             (spec/04-type-system.md §5.3, §5.5)"
                            ),
                            vec![
                                "spec/04-type-system.md §5.3: integer literals default \
                             to int32; the lexer parses at i64 so out-of-range \
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
            deep::Expr::Atom(deep::Atom::Float(_), _) => Type::Prim(Prim::F32),
            deep::Expr::Atom(deep::Atom::Bool(_), _) => Type::Prim(Prim::Bool),
            deep::Expr::Atom(deep::Atom::Str(_), _) => Type::Prim(Prim::String),
            _ => malformed_form(list, "lit", "a scalar atom value", errors),
        }
    } else {
        malformed_form(list, "lit", "a value atom or a `type:` annotation", errors)
    }
}
