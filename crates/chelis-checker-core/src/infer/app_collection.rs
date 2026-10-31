//! Collection operation and constructor rules.
//!
//! These helpers own registered aggregate application checks and the
//! post-unification list callback and dictionary rules, and preserve list
//! callback diagnostics, deferred result constraints and concat shape rules.

use super::*;
use chelis_type_core::decisions::collection::{
    AggregateRule, ConcatListInfo, TensorConcatCallEvidence, decide_collection_constraint,
    queried_container,
};

/// Finish registered non-concat aggregates before the remaining operation
/// families run. Concat retains its separate tensor/collection selector.
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_registered_aggregate(
    rule: AggregateRule,
    source_site: CheckSite<'_>,
    node: &DeepNode,
    kids: &[deep::Expr],
    func_name: &Option<String>,
    arg_tys: &[Type<ErrorWitness>],
    ret_tv: &Type<ErrorWitness>,
    env: &Env,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    operation: &crate::resources::Operation<'_>,
) -> Type<ErrorWitness> {
    let name = func_name
        .as_deref()
        .expect("registered aggregate has a name");
    if arg_tys
        .iter()
        .any(|ty| matches!(subst.apply(ty), Type::Error(_)))
    {
        return subst.apply(ret_tv);
    }
    let operands = match rule {
        AggregateRule::Fold | AggregateRule::Scan => {
            if arg_tys.len() != 3 {
                return report_builtin_arity(errors, node, source_site, name, 3, arg_tys.len());
            }
            let element = vg.fresh_type();
            if let Err(error) = unify(
                &arg_tys[2],
                &Type::adt("List".to_string(), vec![element.clone()]),
                subst,
            ) {
                return report(errors, error.into());
            }
            // Invoke the callback with the initial accumulator. Its
            // result is an independent slot: equality with the next
            // accumulator belongs to the aggregate rule below.
            let callback_result = match unify_checked_call_contract(
                &kids[1],
                None,
                &arg_tys[0],
                &[arg_tys[1].clone(), element],
                None,
                vg,
                subst,
                errors,
                product,
                operation,
            ) {
                Ok(result) => result,
                Err(rejected) => return rejected,
            };
            if rule == AggregateRule::Scan {
                vec![
                    Type::adt("List".to_string(), vec![arg_tys[1].clone()]),
                    Type::adt("List".to_string(), vec![callback_result]),
                ]
            } else {
                vec![arg_tys[1].clone(), callback_result]
            }
        }
        AggregateRule::Append
        | AggregateRule::DictInsert
        | AggregateRule::DictMerge
        | AggregateRule::Concat => arg_tys.to_vec(),
    };
    match rule.decide(&operands, ret_tv, subst) {
        Ok(Some(equation)) => {
            subst.record_result_constraint(equation);
            subst.apply(ret_tv)
        }
        Ok(None) => {
            let name = func_name
                .as_deref()
                .expect("registered aggregate has a name");
            UnresolvedOperandSite::new(node, kids, name, env).defer(
                arg_tys,
                ret_tv,
                product,
                ret_tv.clone(),
            )
        }
        Err(mut error) => {
            error.message = with_node_provenance(node, error.message);
            report(errors, (*error).into())
        }
    }
}

pub(super) fn collection_helper_type_error(
    node: &DeepNode,
    helper: &str,
    contract: &str,
    te: TypeError,
) -> CheckError {
    let kind = check_error_kind_from_type_error_kind(&te.kind);
    let suggestions = match kind {
        CheckErrorKind::PrecisionMismatch => vec!["Insert explicit cast".to_string()],
        _ => vec![],
    };
    CheckError::new(
        kind,
        with_node_provenance(node, format!("{helper} {contract}; {}", te.message)),
        suggestions,
    )
}

/// spec/04 section 2.6 and spec/05 section 1.3.1: `len` and `index` take
/// their container only by auto-borrow, so an explicit borrow expression is
/// a type error as the container operand of a call whose callee is written
/// `len` or `index`, a pipe stage included. That is the whole rule: a call
/// through any other callee, a function value bound to the query included,
/// is decided by that callee's type, and every typing arm decides a borrowed
/// container on its referent ([`queried_container`]).
///
/// The rule reads the call's source, so its one site is the direct
/// application, at the top of `finish_unified_app`. A borrow expression and
/// a borrowed variable have the same type, so no typing arm can apply it;
/// the borrow node's own typing cannot either, because `&xs` is an ordinary
/// argument everywhere else, and Deep input never passes through the Surf
/// desugarer.
///
/// Returns the refusal once the referent is the query's container. A
/// referent that is still unresolved is not decided yet, and one that is not
/// a container is the query's own type error, which names the real problem
/// instead of advising a spelling that fails next.
pub(super) fn explicit_container_borrow_refusal(
    query: &str,
    operand: &Type<ErrorWitness>,
) -> Option<String> {
    let (container, form, refused) = match query {
        "len" => ("List/Dict", "len(xs)", "len(&xs)"),
        "index" => ("List", "index(xs, i)", "index(&xs, i)"),
        _ => return None,
    };
    match queried_container(operand) {
        Type::Adt(name, _) if name == "List" || (query == "len" && name == "Dict") => {
            Some(format!(
                "{query} auto-borrows its {container} argument, so an explicit `&` is not a \
                 supported surface form: write `{form}`, not `{refused}` (got {operand})"
            ))
        }
        _ => None,
    }
}

/// Whether `expr` is the borrow expression `&e`, through metadata wrappers.
pub(super) fn is_borrow_expression(mut expr: &deep::Expr) -> bool {
    loop {
        match expr.carrier() {
            deep::ExprCarrier::DecodedNode(tag, _, _) => return tag == DeepTag::Borrow,
            deep::ExprCarrier::MetadataExpression(meta) => expr = &meta.expr,
            deep::ExprCarrier::StructuralList(_)
            | deep::ExprCarrier::UndecodableHead(_, _, _)
            | deep::ExprCarrier::Atom(_)
            | deep::ExprCarrier::MetadataMap(_) => return false,
        }
    }
}

/// Direct calls and transported contracts publish the same decisions. Stored
/// value equalities flow through the origin ledger; a computed scalar/tensor
/// result retains its facts after checking the callable's declaration.
pub(super) fn publish_collection_equation(
    constraint: &CollectionConstraint<ErrorWitness>,
    node: &DeepNode,
    subst: &mut Subst<ErrorWitness>,
    errors: &mut DiagnosticSink<'_>,
) -> Option<Type<ErrorWitness>> {
    let produced_result;
    match decide_collection_constraint(constraint, None, subst) {
        Ok(Some(decision)) => {
            produced_result = decision.produced_result();
            if let Err(error) = decision.publish(subst) {
                errors.push(error.into());
            }
        }
        Ok(None) => return None,
        Err(message) => {
            return Some(report(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    with_node_provenance(node, message),
                    vec![],
                ),
            ));
        }
    }
    Some(produced_result.unwrap_or_else(|| constraint.result().clone()))
}

/// The static count of one already-inferred `split_keys`-shaped application,
/// read exactly as `check_split_keys_signature` reads a direct call's.
pub(super) fn split_keys_call_count(
    kids: &[deep::Expr],
    env: &Env,
    operation: &crate::resources::Operation<'_>,
) -> Option<i64> {
    kids.get(2).and_then(|count| {
        fold_static_int_expr_in_operation(count, |name| env.static_size_value(name), operation)
    })
}

/// Read the tensor-concat evidence from one already-inferred application.
///
/// Direct and transported calls share this extraction so neither route can
/// retain the axis while dropping literal element extents, or vice versa.
pub(super) fn tensor_concat_call_evidence(
    kids: &[deep::Expr],
    env: &Env,
    subst: &Subst<ErrorWitness>,
    errors: &mut DiagnosticSink<'_>,
    product: &InferenceProduct,
    operation: &crate::resources::Operation<'_>,
) -> TensorConcatCallEvidence {
    let raw_axis = kids
        .get(2)
        .and_then(|__checker_arg0| extract_int_for_dim(__checker_arg0, operation));
    let list_info = match kids.get(1).and_then(collect_cons_chain_for_shape) {
        Some(elements) => ConcatListInfo::Direct(
            elements
                .iter()
                .map(
                    |elem| match product.current_owner_type(elem, subst, errors) {
                        Some(Type::Tensor(dims, _)) => dims,
                        _ => Vec::new(),
                    },
                )
                .collect(),
        ),
        None => ConcatListInfo::BindingLen(static_list_len(kids.get(1), env)),
    };
    TensorConcatCallEvidence {
        raw_axis,
        list_info,
    }
}

/// Statically-known element count of a list expression (chelis#631): a
/// literal `Cons`/`Nil` chain counts directly; a variable carries a
/// length only when it was bound to a list literal in an enclosing scope
/// ([`Env::list_literal_len`]). `None` for anything else — a function
/// result, a parameter, a `split` output.
pub(super) fn static_list_len(expr: Option<&deep::Expr>, env: &Env) -> Option<usize> {
    let expr = expr?;
    if let Some(elements) = collect_cons_chain_for_shape(expr) {
        return Some(elements.len());
    }
    // chelis#1107: carrier-preserving read; a `List`-only destructure lost the
    // recorded literal length of a stamped `(var {} xs)` on the typed ingress.
    let (tag, _, kids) = stamped_parts(expr)?;
    if tag == DeepTag::Var {
        let name = kids.first().and_then(|e| symbol_name(e))?;
        return env.list_literal_len(name);
    }
    None
}

/// Resolve an ADT constructor application and enforce its call shape.
pub(super) fn prepare_constructor_application(
    func_name: &Option<String>,
    env: &Env,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    operation: &crate::resources::Operation<'_>,
) -> Result<Option<String>, Type<ErrorWitness>> {
    macro_rules! reject {
        ($($arg:tt)*) => {
            return Err(report($($arg)*))
        };
    }

    let constructor = func_name
        .as_ref()
        .and_then(|fname| constructor_for_shape(fname, CallShape::Positional, env, adt_reg));
    let ctor_lookup_name = constructor.map(|_| func_name.as_ref().unwrap().clone());

    // The call site uses positional `(app)` syntax here (named-field
    // record construction lowers through a different builder, not
    // through `infer_app`). When two ADTs in the dep graph define
    // same-named constructors with different shapes (chelis#148: e.g.
    // Coral.Frame.Column.IntCol is positional, School.Data.Dataset.IntCol
    // is record), prefer the positional variant for this call site so
    // the call dispatches to the matching ADT instead of erroring on
    // the colliding record variant. Only emit the "must use named
    // fields" error when EVERY same-named variant in scope is record-
    // shaped, which is the original single-package case the error was
    // written for.
    // RFC D-CHECK: positional application of an out-of-module opaque
    // constructor is rejected (one violation per call site; the
    // callee `var`'s constructor-reference check is suppressed below
    // so the application does not double-report). Inference continues
    // so the call still yields its true type.
    if let Some((adt_name, _, _)) = constructor {
        let adt_name = adt_name.to_string();
        crate::opacity::check_opaque_use(
            crate::opacity::OpaqueAction::CtorApplication,
            &adt_name,
            adt_reg,
            errors,
            operation,
        );
    }

    // chelis#317: do not emit the record-shape diagnostic for an applied
    // constructor whose name is out of scope (a type-only import that calls
    // `Alpha(...)`). The shape check resolves through the same fuzzy
    // terminal fallback that mis-binds out-of-scope names, so firing it here
    // would mask the real defect with a confusing "must use named fields"
    // message. Let the head's `infer_var` report `unknown constructor`
    // instead.
    let ctor_call_out_of_scope = func_name
        .as_deref()
        .is_some_and(|fname| constructor_out_of_scope(fname, env));
    if !ctor_call_out_of_scope
        && let Some(ref fname) = ctor_lookup_name
        && let Some((_adt_name, _, variant)) = constructor
        && !variant.fields.is_empty()
        && variant
            .fields
            .iter()
            .all(|(field_name, _)| field_name.is_some())
    {
        reject!(
            errors,
            CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!(
                    "{fname} is a record constructor and must use named fields: {fname} {{ ... }}"
                ),
                vec![],
            ),
        );
    }

    Ok(ctor_lookup_name)
}

/// [05-OP-56]'s key domain: string, bool, or an active signed-integer scalar.
/// Float, aggregate, tensor and random-key types are not dictionary keys.
fn is_dict_key_type(ty: &Type<ErrorWitness>) -> bool {
    match ty {
        Type::Prim(prim) => matches!(prim, Prim::String | Prim::Bool) || prim.is_integer(),
        _ => false,
    }
}

/// List callback and dictionary rules after generic application
/// unification: `map`, `filter`, `partition`, `flat_map`, `flatten`, `zip`,
/// `enumerate` and the `dict_*` operations. [`finish_unified_app`] calls this
/// for any callee its own dispatch does not name; `None` means `fname` is
/// not one of these operations, or its rule had nothing to decide, and the
/// dispatcher continues exactly as when the arms were inline.
#[allow(clippy::too_many_arguments)]
pub(super) fn finish_collection_app(
    fname: &str,
    source_site: CheckSite<'_>,
    node: &DeepNode,
    arg_tys: &[Type<ErrorWitness>],
    result_ty: &Type<ErrorWitness>,
    site: &UnresolvedOperandSite<'_>,
    vg: &mut VarGen,
    subst: &mut Subst<ErrorWitness>,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Option<Type<ErrorWitness>> {
    match fname {
        "map" => {
            if arg_tys.len() != 2 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    2,
                    arg_tys.len(),
                ));
            }
            let elem_ty = vg.fresh_type();
            let out_ty = vg.fresh_type();
            if let Err(te) = unify(
                &subst.apply(&arg_tys[0]),
                &Type::function(vec![elem_ty.clone()], out_ty.clone()),
                subst,
            ) {
                return Some(report(errors, te.into()));
            }
            if let Err(te) = unify(
                &subst.apply(&arg_tys[1]),
                &Type::adt("List".to_string(), vec![elem_ty]),
                subst,
            ) {
                return Some(report(errors, te.into()));
            }
            return Some(Type::adt("List".to_string(), vec![subst.apply(&out_ty)]));
        }
        "filter" => {
            if arg_tys.len() != 2 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    2,
                    arg_tys.len(),
                ));
            }
            let elem_ty = vg.fresh_type();
            if let Err(te) = unify(
                &subst.apply(&arg_tys[0]),
                &Type::function(vec![elem_ty.clone()], Type::Prim(Prim::Bool)),
                subst,
            ) {
                return Some(report(
                    errors,
                    collection_helper_type_error(
                        node,
                        "filter",
                        "expects a callback that returns bool",
                        te,
                    ),
                ));
            }
            if let Err(te) = unify(
                &subst.apply(&arg_tys[1]),
                &Type::adt("List".to_string(), vec![elem_ty.clone()]),
                subst,
            ) {
                return Some(report(errors, te.into()));
            }
            return Some(Type::adt("List".to_string(), vec![subst.apply(&elem_ty)]));
        }
        "partition" => {
            if arg_tys.len() != 2 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    2,
                    arg_tys.len(),
                ));
            }
            let elem_ty = vg.fresh_type();
            if let Err(te) = unify(
                &subst.apply(&arg_tys[0]),
                &Type::function(vec![elem_ty.clone()], Type::Prim(Prim::Bool)),
                subst,
            ) {
                return Some(report(
                    errors,
                    collection_helper_type_error(
                        node,
                        "partition",
                        "expects a callback that returns bool",
                        te,
                    ),
                ));
            }
            if let Err(te) = unify(
                &subst.apply(&arg_tys[1]),
                &Type::adt("List".to_string(), vec![elem_ty.clone()]),
                subst,
            ) {
                return Some(report(errors, te.into()));
            }
            let out_list = Type::adt("List".to_string(), vec![subst.apply(&elem_ty)]);
            return Some(Type::tuple(vec![out_list.clone(), out_list]));
        }
        "flat_map" => {
            if arg_tys.len() != 2 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    2,
                    arg_tys.len(),
                ));
            }
            let elem_ty = vg.fresh_type();
            let out_elem_ty = vg.fresh_type();
            if let Err(te) = unify(
                &subst.apply(&arg_tys[0]),
                &Type::function(
                    vec![elem_ty.clone()],
                    Type::adt("List".to_string(), vec![out_elem_ty.clone()]),
                ),
                subst,
            ) {
                return Some(report(errors, te.into()));
            }
            if let Err(te) = unify(
                &subst.apply(&arg_tys[1]),
                &Type::adt("List".to_string(), vec![elem_ty]),
                subst,
            ) {
                return Some(report(errors, te.into()));
            }
            return Some(Type::adt(
                "List".to_string(),
                vec![subst.apply(&out_elem_ty)],
            ));
        }
        "flatten" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Adt(outer_name, outer_args)
                        if outer_name == "List" && outer_args.len() == 1 =>
                    {
                        match &outer_args[0] {
                            Type::Adt(inner_name, inner_args)
                                if inner_name == "List" && inner_args.len() == 1 =>
                            {
                                return Some(Type::adt(
                                    "List".to_string(),
                                    vec![inner_args[0].clone()],
                                ));
                            }
                            Type::Error(_) => return Some(result_ty.clone()),
                            Type::Var(_) => {
                                return Some(site.defer(
                                    arg_tys,
                                    result_ty,
                                    product,
                                    result_ty.clone(),
                                ));
                            }
                            other => {
                                return Some(report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(
                                            node,
                                            format!(
                                                "flatten expects List[List[T]] input, got List[{other}]"
                                            ),
                                        ),
                                        vec![],
                                    ),
                                ));
                            }
                        }
                    }
                    Type::Error(_) => return Some(result_ty.clone()),
                    Type::Var(_) => {
                        return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("flatten expects List[List[T]] input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        "zip" => {
            if arg_tys.len() != 2 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    2,
                    arg_tys.len(),
                ));
            }
            let lhs = subst.apply(&arg_tys[0]);
            let rhs = subst.apply(&arg_tys[1]);
            match (lhs, rhs) {
                (Type::Adt(lhs_name, lhs_args), Type::Adt(rhs_name, rhs_args))
                    if lhs_name == "List"
                        && rhs_name == "List"
                        && lhs_args.len() == 1
                        && rhs_args.len() == 1 =>
                {
                    return Some(Type::adt(
                        "List".to_string(),
                        vec![Type::tuple(vec![lhs_args[0].clone(), rhs_args[0].clone()])],
                    ));
                }
                // chelis#1512: an upstream failure keeps the early
                // return, so the cascade still suppresses. Order matters:
                // an (Error, Var) pair must suppress, not suspend.
                (Type::Error(_), _) | (_, Type::Error(_)) => {
                    return Some(result_ty.clone());
                }
                (Type::Var(_), _) | (_, Type::Var(_)) => {
                    return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                }
                (lhs, rhs) => {
                    return Some(report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!("zip expects List inputs, got {lhs} and {rhs}"),
                            ),
                            vec![],
                        ),
                    ));
                }
            }
        }
        "enumerate" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                        return Some(Type::adt(
                            "List".to_string(),
                            vec![Type::tuple(vec![Type::Prim(Prim::Int64), args[0].clone()])],
                        ));
                    }
                    Type::Error(_) => return Some(result_ty.clone()),
                    Type::Var(_) => {
                        return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("enumerate expects List input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        "dict_of" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                        // chelis#2523: `dict_of`'s input is a list of
                        // `(key, value)` pairs, so an element whose type is
                        // still a variable is constrained to a pair rather
                        // than admitted. That ties the result's key and
                        // value types to the list's element, so a declared
                        // result determines them (`dict_of([])`), and the
                        // key rule below decides once the key is known.
                        if let Type::Var(_) = subst.apply(&args[0]) {
                            let pair = Type::tuple(vec![vg.fresh_type(), vg.fresh_type()]);
                            if let Err(te) = unify(&args[0], &pair, subst) {
                                return Some(report(errors, te.into()));
                            }
                        }
                        match subst.apply(&args[0]) {
                            Type::Tuple(items) if items.len() == 2 => {
                                let dict = Type::adt(
                                    "Dict".to_string(),
                                    vec![items[0].clone(), items[1].clone()],
                                );
                                match &items[0] {
                                    key if is_dict_key_type(key) => {}
                                    Type::Error(_) => return Some(result_ty.clone()),
                                    Type::Var(_) => {
                                        return Some(site.defer(arg_tys, result_ty, product, dict));
                                    }
                                    other => {
                                        return Some(report(
                                            errors,
                                            CheckError::new(
                                                CheckErrorKind::TypeMismatch,
                                                with_node_provenance(
                                                    node,
                                                    format!(
                                                        "dict_of keys must be string, bool, or a signed integer scalar ([05-OP-56]), got {other}"
                                                    ),
                                                ),
                                                vec![],
                                            ),
                                        ));
                                    }
                                }
                                return Some(dict);
                            }
                            Type::Error(_) => return Some(result_ty.clone()),
                            other => {
                                return Some(report(
                                    errors,
                                    CheckError::new(
                                        CheckErrorKind::TypeMismatch,
                                        with_node_provenance(
                                            node,
                                            format!(
                                                "dict_of expects List[(K, V)] input, got List[{other}]"
                                            ),
                                        ),
                                        vec![],
                                    ),
                                ));
                            }
                        }
                    }
                    Type::Error(_) => return Some(result_ty.clone()),
                    Type::Var(_) => {
                        return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("dict_of expects List input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        "dict_get" => {
            if arg_tys.len() != 2 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    2,
                    arg_tys.len(),
                ));
            }
            match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                (Type::Adt(name, args), key_ty) if name == "Dict" && args.len() == 2 => {
                    if let Err(te) = unify(&args[0], &key_ty, subst) {
                        return Some(report(errors, te.into()));
                    }
                    return Some(Type::adt("Option".to_string(), vec![subst.apply(&args[1])]));
                }
                (Type::Error(_), _) | (_, Type::Error(_)) => return Some(result_ty.clone()),
                // chelis#2523: not a `Dict` YET. Suspending re-enters this
                // route once the operand binds, and decides it at its
                // instantiations if it never does.
                (Type::Var(_), _) | (_, Type::Var(_)) => {
                    return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                }
                (dict_ty, key_ty) => {
                    return Some(report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!(
                                    "dict_get expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                ),
                            ),
                            vec![],
                        ),
                    ));
                }
            }
        }
        "dict_contains" => {
            if arg_tys.len() != 2 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    2,
                    arg_tys.len(),
                ));
            }
            match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                (Type::Adt(name, args), key_ty) if name == "Dict" && args.len() == 2 => {
                    if let Err(te) = unify(&args[0], &key_ty, subst) {
                        return Some(report(errors, te.into()));
                    }
                    return Some(Type::Prim(Prim::Bool));
                }
                (Type::Error(_), _) | (_, Type::Error(_)) => return Some(result_ty.clone()),
                // chelis#2523: not a `Dict` YET. Suspending re-enters this
                // route once the operand binds, and decides it at its
                // instantiations if it never does.
                (Type::Var(_), _) | (_, Type::Var(_)) => {
                    return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                }
                (dict_ty, key_ty) => {
                    return Some(report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!(
                                    "dict_contains expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                ),
                            ),
                            vec![],
                        ),
                    ));
                }
            }
        }
        "dict_remove" => {
            if arg_tys.len() != 2 {
                return Some(report_builtin_arity(
                    errors,
                    node,
                    source_site,
                    fname,
                    2,
                    arg_tys.len(),
                ));
            }
            match (subst.apply(&arg_tys[0]), subst.apply(&arg_tys[1])) {
                (Type::Adt(name, args), key_ty) if name == "Dict" && args.len() == 2 => {
                    if let Err(te) = unify(&args[0], &key_ty, subst) {
                        return Some(report(errors, te.into()));
                    }
                    return Some(Type::adt(
                        "Dict".to_string(),
                        vec![subst.apply(&args[0]), subst.apply(&args[1])],
                    ));
                }
                (Type::Error(_), _) | (_, Type::Error(_)) => return Some(result_ty.clone()),
                // chelis#2523: not a `Dict` YET. Suspending re-enters this
                // route once the operand binds, and decides it at its
                // instantiations if it never does.
                (Type::Var(_), _) | (_, Type::Var(_)) => {
                    return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                }
                (dict_ty, key_ty) => {
                    return Some(report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            with_node_provenance(
                                node,
                                format!(
                                    "dict_remove expects Dict[K, V] and K, got {dict_ty} and {key_ty}"
                                ),
                            ),
                            vec![],
                        ),
                    ));
                }
            }
        }
        "dict_keys" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                        return Some(Type::adt("List".to_string(), vec![args[0].clone()]));
                    }
                    Type::Error(_) => return Some(result_ty.clone()),
                    Type::Var(_) => {
                        return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("dict_keys expects Dict input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        "dict_values" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                        return Some(Type::adt("List".to_string(), vec![args[1].clone()]));
                    }
                    Type::Error(_) => return Some(result_ty.clone()),
                    Type::Var(_) => {
                        return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("dict_values expects Dict input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        "dict_entries" => {
            if let Some(first_arg) = arg_tys.first() {
                match subst.apply(first_arg) {
                    Type::Adt(name, args) if name == "Dict" && args.len() == 2 => {
                        return Some(Type::adt(
                            "List".to_string(),
                            vec![Type::tuple(vec![args[0].clone(), args[1].clone()])],
                        ));
                    }
                    Type::Error(_) => return Some(result_ty.clone()),
                    Type::Var(_) => {
                        return Some(site.defer(arg_tys, result_ty, product, result_ty.clone()));
                    }
                    other => {
                        return Some(report(
                            errors,
                            CheckError::new(
                                CheckErrorKind::TypeMismatch,
                                with_node_provenance(
                                    node,
                                    format!("dict_entries expects Dict input, got {other}"),
                                ),
                                vec![],
                            ),
                        ));
                    }
                }
            }
        }
        _ => {}
    }
    None
}

#[cfg(test)]
mod aggregate_origin_tests {
    use super::*;

    #[test]
    fn aggregate_admission_diagnostics_name_expected_operands() {
        use chelis_type_core::decisions::collection::AggregateRule;
        for (rule, operands, expected) in [
            (
                AggregateRule::Append,
                vec![Type::Prim(Prim::F32), Type::Prim(Prim::Bool)],
                "append expects List input, got f32",
            ),
            (
                AggregateRule::Concat,
                vec![Type::Prim(Prim::F32), Type::Prim(Prim::Bool)],
                "concat expects matching List inputs, got f32 and bool",
            ),
            (
                AggregateRule::DictInsert,
                vec![
                    Type::Prim(Prim::F32),
                    Type::Prim(Prim::String),
                    Type::Prim(Prim::Bool),
                ],
                "dict_insert expects Dict[K, V], K, and V, got f32, string, and bool",
            ),
            (
                AggregateRule::DictMerge,
                vec![Type::Prim(Prim::F32), Type::Prim(Prim::Bool)],
                "dict_merge expects matching Dict inputs, got f32 and bool",
            ),
        ] {
            let error = rule
                .decide(
                    &operands,
                    &Type::Prim(Prim::Bool),
                    &Subst::<ErrorWitness>::new(),
                )
                .expect_err("invalid collection constructors must reject");
            assert!(
                matches!(
                    error.kind,
                    chelis_type_core::decisions::DecisionFailureKind::TypeMismatch
                ),
                "{rule:?}"
            );
            assert_eq!(error.message, expected, "{rule:?}");
        }
    }

    #[test]
    fn every_registered_aggregate_returns_equality_without_operand_bindings() {
        use builtins::ValueEquality;
        let mut rules = Vec::new();
        for declaration in builtins::BUILTINS {
            for case in declaration.capability.sibling_cases {
                if let ValueEquality::Aggregate(rule) = builtins::case_value_equality(case.case)
                    && !rules.contains(&rule)
                {
                    rules.push(rule);
                }
            }
        }
        for rule in rules {
            let mut vg = VarGen::default();
            let input = vg.fresh_type();
            let result = vg.fresh_type();
            let unknown = Type::function(vec![input.clone()], vg.fresh_type());
            let known = Type::function(vec![Type::Prim(Prim::F32)], Type::Prim(Prim::F32));
            let list = |value| Type::adt("List".to_string(), vec![value]);
            let dict = |value| Type::adt("Dict".to_string(), vec![Type::Prim(Prim::String), value]);
            let operands = match rule {
                AggregateRule::Append => vec![list(known), unknown],
                AggregateRule::Concat => vec![list(unknown), list(known)],
                AggregateRule::DictInsert => vec![dict(known), Type::Prim(Prim::String), unknown],
                AggregateRule::DictMerge => vec![dict(unknown), dict(known)],
                AggregateRule::Fold => vec![unknown, known],
                AggregateRule::Scan => vec![list(unknown), list(known)],
            };
            let subst = Subst::<ErrorWitness>::new();
            let equation = rule.decide(&operands, &result, &subst).unwrap().unwrap();
            assert_eq!(subst.apply(&input), input, "{rule:?} bound an input");
            assert_eq!(subst.apply(&result), result, "{rule:?} bound its result");
            let ResultConstraint::Join { inputs, result } = equation else {
                panic!("{rule:?} did not preserve its result equality");
            };
            // The equation remains required: a conflicting later application
            // must fail, even though the operation could not bind the input.
            let mut applied = subst.clone();
            unify(&input, &Type::Prim(Prim::Bool), &mut applied).unwrap();
            assert!(
                inputs
                    .iter()
                    .any(|ty| unify(&result, ty, &mut applied).is_err()),
                "{rule:?} erased a required equality"
            );
        }
    }
}
