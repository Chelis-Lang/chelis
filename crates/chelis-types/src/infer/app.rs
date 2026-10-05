//! Generic application inference and operation dispatch.
//!
//! This module contains code moved from the former inference monolith.
//! The extraction preserves control flow and diagnostic order.

use super::*;

/// The §4.5.2 join categories are semantic, not `Dim` representation tags.
/// A private labelled Var can represent a concrete name with an optional
/// known extent; only an active authored binder must retain rigid evidence.
enum ListAxis {
    Variable,
    Wildcard,
    Concrete {
        extent: Option<i64>,
        label: Option<String>,
    },
}

impl ListAxis {
    fn classify(dim: &Dim, subst: &Subst) -> Self {
        let observation = subst.observe_dim(dim);
        if observation.rank().is_some()
            || (observation.variable().is_some()
                && (observation.name().is_none() || observation.is_protected()))
        {
            Self::Variable
        } else if observation.is_wildcard() {
            Self::Wildcard
        } else {
            Self::Concrete {
                extent: observation.known_extent(),
                label: observation.name().map(str::to_owned),
            }
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn infer_app(
    expr: &deep::Expr,
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    expected_result: Option<&Type>,
) -> Type {
    // Preserve lexical ownership before inference mutates the environment.
    // This is discovery metadata, not a backend acceptance decision.
    let kids = node.children_slice();
    let builtin = kids.first().and_then(|callee| {
        let (tag, _, parts) = stamped_parts(callee)?;
        if tag != DeepTag::Var {
            return None;
        }
        let name = parts.first().and_then(symbol_name)?;
        if env.is_lexically_bound(name) {
            return None;
        }
        builtins::builtin_decl(name)
    });
    let checkpoint = errors.checkpoint();
    let result = infer_app_inner(
        expr,
        node,
        env,
        vg,
        subst,
        adt_reg,
        errors,
        product,
        expected_result,
    );
    if errors.iter_since(checkpoint).next().is_none()
        && let Some(builtin) = builtin
    {
        // Only overload selectors need operand stamps. Axis syntax and other
        // non-value children must not be re-inferred to discover an identity.
        let positions: &[usize] = match builtin.name {
            "len" | "to_string" | "eq" | "neq" => &[0],
            "concat" => &[0, 1],
            _ => &[],
        };
        let mut arguments = vec![Type::Unit; kids.len().saturating_sub(1)];
        for &position in positions {
            if let Some(child) = kids.get(position + 1)
                && let Some(ty) = product.current_owner_type(child, subst, errors)
            {
                arguments[position] = ty;
            }
        }
        if errors.iter_since(checkpoint).next().is_none() {
            match builtin.semantic_selection(&arguments, subst) {
                Ok(selection) => product.record_builtin_selection(selection),
                Err(reason) => {
                    return report(
                        errors,
                        CheckError::new(
                            CheckErrorKind::TypeMismatch,
                            format!(
                                "{} operand has no declared semantic case: {reason}",
                                builtin.name
                            ),
                            vec![],
                        ),
                    );
                }
            }
        }
    }
    result
}

#[allow(clippy::too_many_arguments)]
fn infer_app_inner(
    expr: &deep::Expr,
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
    expected_result: Option<&Type>,
) -> Type {
    let kids = node.children_slice();
    if kids.is_empty() {
        return malformed_form(node, "app", "a callee expression", errors);
    }

    // Check if func is a comparison op (for special return type handling)
    let source_func_name = stamped_parts(&kids[0]).and_then(|(tag, _, callee_kids)| {
        (tag == DeepTag::Var)
            .then(|| {
                callee_kids
                    .first()
                    .and_then(symbol_name)
                    .map(str::to_string)
            })
            .flatten()
    });
    // Builtin-specific application rules are selected only after ordinary
    // lexical lookup. A parameter, block binding, or pattern binding with the
    // same builtin spelling owns the call; its inferred function type, rather
    // than the builtin's name-keyed checker route, decides whether the
    // application is valid (spec/04-type-system.md §8.6; chelis#1076).
    //
    // Keep the override exact to the closed builtin vocabulary. In particular,
    // applied uppercase heads retain constructor classification under
    // spec/01-nomenclature.md §3.2 even when a single-letter value binder with
    // the same spelling is in scope.
    let func_name = source_func_name.clone().filter(|name| {
        !builtins::BUILTIN_NAMES.contains(&name.as_str()) || !env.is_lexically_bound(name)
    });

    if matches!(func_name.as_deref(), Some("permute")) {
        return infer_permute_app(expr, node, env, vg, subst, adt_reg, errors, product);
    }

    if matches!(func_name.as_deref(), Some("reshape")) {
        let inferred = infer_reshape_app(expr, node, env, vg, subst, adt_reg, errors, product);
        if let Some(expected) = env.exact_stdlib_expected_result()
            && matches!(expected, Type::Tensor(_, _))
        {
            // #1298 owns the general runtime-axis shape relation. The exact
            // [05-OP-35] stdlib graph is structurally locked and its declared
            // result is authoritative at this package-reserved boundary; the
            // ordinary reshape checker still traverses and stamps every child.
            return expected.clone();
        }
        return inferred;
    }

    if matches!(func_name.as_deref(), Some("shrink")) {
        return infer_shrink_app(expr, node, env, vg, subst, adt_reg, errors, product);
    }

    if matches!(func_name.as_deref(), Some("pad")) {
        return infer_pad_app(expr, node, env, vg, subst, adt_reg, errors, product);
    }

    if matches!(func_name.as_deref(), Some("stride")) {
        return infer_stride_app(expr, node, env, vg, subst, adt_reg, errors, product);
    }

    // chelis#339: the four-argument anchored form belongs to `insert`, but
    // its registered HM scheme is arity-3. Route it around generic arity
    // checking. Both operations' malformed arities take this procedural path
    // for their own exact diagnostics; ordinary 3-argument calls use HM.
    if let Some(callee @ ("expand" | "insert")) = func_name.as_deref()
        && kids.len() != 4
    {
        // `&'static str`, not the borrow, so the callee outlives `func_name`.
        let callee = if callee == "insert" {
            "insert"
        } else {
            "expand"
        };
        return infer_expand_app(callee, expr, node, env, vg, subst, adt_reg, errors, product);
    }

    // chelis#339 Part 2: variadic named-axis reduction `sum(x, seq, head)`.
    // The reduction HM schemes are arity-2 (`(input, axis)`), so a 3+-arg
    // call would hit the generic arity check before the reduction arm;
    // dispatch it here. `check_reduction_signature`'s named loop handles N
    // axes (rejecting positional integers, unknown names, and duplicates).
    // The index-returning reductions are routed too, so they get a targeted
    // no-variadic-form rejection instead of a generic arity error.
    if matches!(
        func_name.as_deref(),
        Some(
            "sum"
                | "count"
                | "mean"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
        )
    ) && kids.len() >= 4
    {
        return infer_reduction_app(
            expr,
            node,
            func_name.as_deref().unwrap(),
            env,
            vg,
            subst,
            adt_reg,
            errors,
            product,
        );
    }

    if matches!(
        func_name.as_deref(),
        Some(
            "reduce_window_max" | "reduce_window_min" | "reduce_window_sum" | "reduce_window_mean"
        )
    ) {
        return infer_reduce_window_app(
            expr,
            node,
            func_name.as_deref().unwrap(),
            env,
            vg,
            subst,
            adt_reg,
            errors,
            product,
        );
    }

    match prepare_constructor_application(&func_name, env, adt_reg, errors) {
        Ok(_) => {}
        Err(rejected) => return rejected,
    }

    // Applied uppercase heads are constructor syntax, even when a value
    // binder with the same spelling is present (spec/01 §3.2). Do not send a
    // resolved constructor back through the string-keyed value environment:
    // a parameter named `N` would replace the constructor scheme there.
    // Conversely, do not rediscover an owner by scanning the ADT registry:
    // two positional constructors may share a name (`Option::Some` and a
    // local `Wrapper::Some`), and registry order is not scope. The separate
    // constructor authority preserves the active declaration/import owner
    // and scheme across ordinary lexical shadowing.
    let applied_constructor_head = source_func_name.as_deref().is_some_and(is_constructor_name);
    // chelis#1801: bracket the callee's inference so the absorption below
    // sees exactly the dimension variables THIS application's instantiation
    // minted. The mark is taken before the callee and read immediately after
    // it, so no argument's instantiation is in scope.
    let instantiation_mark = product.instantiation_dvar_mark();
    // chelis#1654: the same bracket owns checked operation-contract
    // instantiations. Unlike type variables, a fully monomorphic contract has
    // no structural identity to rediscover later; the opaque IDs minted while
    // inferring this callee are the exact capabilities this application may
    // consume.
    let collection_contract_mark = subst.collection_contract_mark();
    let func_ty = if applied_constructor_head {
        let source_name = source_func_name.as_deref().unwrap();
        let resolved_constructor = if constructor_out_of_scope(source_name, env) {
            None
        } else {
            constructor_for_shape(source_name, CallShape::Positional, env, adt_reg).map(
                |(_, scheme, _)| {
                    let instantiated = env.instantiate_scheme(scheme, vg, subst);
                    // chelis#1801: a constructor head is instantiated here
                    // rather than through the Var rule, so it records its
                    // own fresh dimension variables or the bracket above
                    // would see none for a `Ctor(...)` application.
                    product.record_instantiation_dvars(
                        instantiated.dvars.iter().map(|(_, fresh)| *fresh),
                    );
                    instantiated.ty
                },
            )
        };
        match resolved_constructor {
            Some(constructor_type) => constructor_type,
            None => {
                let name = source_func_name.as_deref().unwrap();
                report(
                    errors,
                    CheckError::new(
                        CheckErrorKind::UnknownConstructor {
                            identifier: name.to_string(),
                        },
                        with_macro_provenance(&kids[0], format!("unknown constructor: {name}")),
                        vec![format!(
                            "Constructor '{name}' is not in scope. Declare it locally or add it \
                             to an import (e.g. `import Mod ({name})`)"
                        )],
                    ),
                )
            }
        }
    } else {
        product.callee_reference = matches!(
            kids[0].carrier(),
            chelis_deep::ExprCarrier::DecodedNode(DeepTag::Var, _, _)
        );
        infer_expr(&kids[0], env, vg, subst, adt_reg, errors, product)
    };
    // The constructor callee no longer passes through `infer_expr`, but it is
    // still a runtime expression owner and must contribute the same fitness
    // and annotation receipt as every other callee.
    if applied_constructor_head {
        product.total_nodes += 1;
        if !matches!(func_ty, Type::Error(_)) {
            product.typed_nodes += 1;
        }
        product.record_canonical(&kids[0], func_ty.clone());
    }
    let instantiation_dvars = product.instantiation_dvars_since(instantiation_mark);
    let callee_collection_contracts = subst.collection_contract_ids_since(collection_contract_mark);
    macro_rules! return_with_collection_cleanup {
        ($value:expr) => {{
            subst.cancel_collection_contract_application(collection_contract_mark);
            return $value;
        }};
    }
    // A reduction's axis argument may name a *dimension* of the operand
    // (`sum(x, seq)`, Tier-3 named-axis reduction, spec §4.5.3), not a bound
    // *value*. Like `expand`'s symbolic size arg below, such a name is typed as
    // an axis (`i32`) rather than inferred as a value — otherwise the
    // name-resolution pass would report a spurious `unbound variable`. The
    // actual name is read back from the arg expr in `check_reduction_signature`.
    let is_named_reduction = matches!(
        func_name.as_deref(),
        Some(
            "sum"
                | "count"
                | "mean"
                | "max_reduce"
                | "min_reduce"
                | "prod_reduce"
                | "argmax_reduce"
                | "argmin_reduce"
        )
    );
    let arg_tys: Vec<Type> = kids[1..]
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            // `expand` axis slots that may carry a dim NAME instead of a bound
            // value: the size (index 2, §4.7.2 form 2) and — chelis#339
            // named-axis expand — the inserted-axis name (index 1). The
            // inserted-axis slot is scope-discriminated: a name bound in the
            // value environment is a *runtime value* (the issue #259 class,
            // `expand(&x, ax, 4)` with `ax: i32`), not a dim name, and must
            // keep flowing through ordinary inference into the
            // compile-time-constant rejection. The size slot is discriminated
            // the same way: a name bound in the value environment is a runtime
            // size whose own type must be `i64`, so `insert(x, 0, j)` with
            // `j: i32` keeps its `i32` and reaches the size-dtype rejection
            // below rather than being retyped as an extent (chelis#469). The
            // 4-arg anchored form routes through `infer_expand_app` instead
            // and never reaches this loop.
            let is_expand = matches!(func_name.as_deref(), Some("expand") | Some("insert"));
            let names_a_dim =
                symbolic_dim_ref_name(arg).is_some_and(|name| env.lookup(name).is_none());
            let is_expand_size = is_expand && index == 2 && names_a_dim;
            let is_expand_inserted_name = is_expand && index == 1 && names_a_dim;
            let is_reduction_axis = is_named_reduction && index >= 1;
            if (is_expand_size || is_expand_inserted_name || is_reduction_axis)
                && symbolic_dim_ref_name(arg).is_some()
            {
                // [05-DIM-1]: a dim name in the size slot is an extent
                // (i64); the inserted-axis name and reduction axes are
                // axis-domain (i32).
                if is_expand_size {
                    Type::Prim(Prim::Int64)
                } else {
                    Type::Prim(Prim::Int32)
                }
            } else {
                infer_expr(arg, env, vg, subst, adt_reg, errors, product)
            }
        })
        .collect();

    // [05-OP-67]: `drop` is the one-argument linearity consume and returns
    // unit for every operand type. Its arity is exact here rather than in
    // `app_post`, because this route returns before unification runs; the
    // list slice that once shared the name is `skip` ([05-OP-54]).
    if matches!(func_name.as_deref(), Some("drop")) {
        if arg_tys.len() != 1 {
            return_with_collection_cleanup!(report_builtin_arity(
                errors,
                node,
                CheckSite::Expr(expr),
                "drop",
                1,
                arg_tys.len()
            ));
        }
        // The operand is owned, so a borrowed one is a type error rather
        // than an implicit consume of its owner.
        let operand = subst.apply(&arg_tys[0]);
        if matches!(operand, Type::Ref(_)) {
            return_with_collection_cleanup!(report_at_check_site(
                errors,
                CheckError::with_types(
                    CheckErrorKind::TypeMismatch,
                    with_node_provenance(
                        node,
                        format!(
                            "drop argument 1: expected an owned value, got borrowed `{operand}`; \
                             `drop` ends its operand's lifetime and cannot take a borrow \
                             ([05-OP-67])"
                        ),
                    ),
                    "an owned value".to_string(),
                    operand.to_string(),
                    vec!["Drop the owner itself: write `drop(x)`, not `drop(&x)`".to_string()],
                ),
                CheckSite::Expr(expr),
            ));
        }
        return_with_collection_cleanup!(Type::Unit);
    }

    // #2413: the counter-stream draws took no key. A call at the retired
    // arity names the retired spelling and points at explicit keys, rather
    // than reporting a bare arity count.
    let retired_draw = match func_name.as_deref() {
        Some("dropout") if arg_tys.len() == 2 => Some(("dropout(x, rate)", "dropout(k, x, rate)")),
        Some("uniform_like") if arg_tys.len() == 3 => Some((
            "uniform_like(t, low, high)",
            "uniform_like(k, t, low, high)",
        )),
        _ => None,
    };
    if let Some((retired, keyed)) = retired_draw {
        let builtin = if func_name.as_deref() == Some("dropout") {
            "dropout"
        } else {
            "uniform_like"
        };
        let expected = format!("{} arguments", arg_tys.len() + 1);
        let got = format!("{} arguments", arg_tys.len());
        return_with_collection_cleanup!(report_at_check_site(
            errors,
            CheckError::with_types(
                CheckErrorKind::ArityMismatch,
                with_node_provenance(
                    node,
                    format!(
                        "call `{builtin}`: expected {expected}, got {got}; \
                         `{retired}` is the retired counter-stream spelling: a random draw \
                         takes an explicit key first, `{keyed}` (spec/05-risc-primitives.md \
                         section 2.7)"
                    ),
                ),
                expected,
                got,
                vec![format!(
                    "Pass a key first: make one with `key_from_seed(seed)` and derive more \
                     with `split_key`, `split_keys` or `fold_in`, as in `{keyed}`"
                )],
            ),
            CheckSite::Expr(expr),
        ));
    }

    // [04-DTYPE-2] restricts a bounded type variable to primitive dtypes.
    // It therefore has a scalar surface even before specialization. Reject
    // mixed surfaces before unification can emit an unrelated occurs-check
    // error for p beside tensor[D, p]. Unrestricted variables remain unknown.
    if let Some(ref fname) = func_name
        && (builtins::COMPARISON_OPS.contains(&fname.as_str())
            || matches!(
                fname.as_str(),
                "add" | "sub" | "mul" | "div" | "floor_div" | "trunc_div" | "max_elem" | "min_elem"
            ))
        && arg_tys.len() == 2
    {
        let lhs = type_for_readonly_check(&arg_tys[0], subst);
        let rhs = type_for_readonly_check(&arg_tys[1], subst);
        let scalar = |ty: &Type| {
            matches!(ty, Type::Prim(_))
                || matches!(ty, Type::Var(p) if subst.tvar_restriction(*p).is_some_and(|bound| !bound.is_value_constraint()))
        };
        if (matches!(lhs, Type::Tensor(..)) && scalar(&rhs))
            || (scalar(&lhs) && matches!(rhs, Type::Tensor(..)))
        {
            let authority = if builtins::COMPARISON_OPS.contains(&fname.as_str()) {
                "spec/05-risc-primitives.md [05-OP-36] makes a mixed surface a type error, and section 1.2 admits no broadcasting exception"
            } else {
                "spec/05-risc-primitives.md section 1.2 and spec/04-type-system.md section 4.3 require explicit shape construction"
            };
            let mut error = CheckError::new(
                CheckErrorKind::TypeMismatch,
                format!("`{fname}` does not admit a scalar beside a tensor, got {lhs} and {rhs}: {authority}"),
                vec![
                    "Give the scalar the tensor's shape explicitly. For a rank-one tensor xs and a scalar c of the same dtype, use `insert(scalar_to_tensor(c), 0i32, shape(xs, 0i32))`; insert each axis for higher ranks.".to_string(),
                    "For concrete f32 values, another explicit spelling is `gt(xs, expand(to_tensor([1.5f32]), 0i32, shape(xs, 0i32)))`.".to_string(),
                ],
            );
            if let Some(id) = node_span_id(node) {
                error.span_offset = parse_span_offset(id);
                error.span_id = Some(id.to_string());
            } else {
                let off = expr.span().offset;
                if off > 0 {
                    error.span_offset = Some(off);
                }
            }
            return_with_collection_cleanup!(report(errors, error));
        }
    }

    product.record_call_operand_contracts(&func_ty, func_name.as_deref(), &arg_tys, env, subst);

    // Preserve the direct operation's diagnostic before its scheme enforces
    // the same family through unification. Indirect calls need no name lookup:
    // their checked function value carries the restriction itself.
    // Matmul's concrete integer refusal belongs to its signature checker.
    // Run that existing refusal before the scheme's family error can hide it.
    if let Some(fname) = func_name.as_deref()
        && (INT_BINOPS.contains(&fname) || INT_SHIFT_OPS.contains(&fname))
        && arg_tys.iter().any(|ty| {
            // [05-OP-64]: `mod` also admits floats (chelis#626).
            matches!(type_for_readonly_check(ty, subst),
                Type::Prim(prim) | Type::Tensor(_, TensorPrec::Concrete(prim))
                    if !prim.is_integer() && !(fname == "mod" && prim.is_float()))
        })
        && let Some(rejected) = integer_binop_result_type(
            node,
            Some(fname),
            &arg_tys,
            vg,
            subst,
            errors,
            None,
            &Type::Unit,
            product,
        )
    {
        // Preserve the existing direct operation's diagnostic. Symbolic
        // family requirements still travel through ordinary unification.
        return_with_collection_cleanup!(rejected);
    }
    if func_name.as_deref() == Some("matmul") && arg_tys.len() == 2 {
        let lhs = type_for_readonly_check(&arg_tys[0], subst);
        let rhs = type_for_readonly_check(&arg_tys[1], subst);
        if let (
            Type::Tensor(_, TensorPrec::Concrete(left)),
            Type::Tensor(_, TensorPrec::Concrete(right)),
        ) = (&lhs, &rhs)
            && left == right
            && left.is_integer()
        {
            return_with_collection_cleanup!(check_matmul_signature(
                CheckSite::Expr(expr),
                &arg_tys,
                &Type::Unit,
                subst,
                errors
            ));
        }
    }
    let mixed_division_precisions =
        matches!(func_name.as_deref(), Some("div" | "trunc_div")) && arg_tys.len() == 2 && {
            let precision = |ty: &Type| match type_for_readonly_check(ty, subst) {
                Type::Prim(prim) | Type::Tensor(_, TensorPrec::Concrete(prim)) => Some(prim),
                _ => None,
            };
            matches!((precision(&arg_tys[0]), precision(&arg_tys[1])),
                (Some(left), Some(right)) if left != right)
        };
    if let Some(fname) = func_name.as_deref()
        && operand_family_policy(fname).is_some()
        && !mixed_division_precisions
    {
        let operands = if matches!(fname, "mean" | "softmax") {
            &arg_tys[..arg_tys.len().min(1)]
        } else {
            &arg_tys[..]
        };
        for operand in operands {
            let resolved = type_for_readonly_check(operand, subst);
            if let Some((kind, message, hints)) = operand_dtype_rejection(fname, &resolved) {
                return_with_collection_cleanup!(report(
                    errors,
                    CheckError::new(kind, with_node_provenance(node, message,), hints,),
                ));
            }
            if let Type::Tensor(_, TensorPrec::Var(var)) = resolved
                && let Some(rejected) =
                    decide_precision_variable_operand(node, fname, var, env, subst, errors)
            {
                return_with_collection_cleanup!(rejected);
            }
        }
    }

    // The builtin signature structurally shares one precision variable across
    // both tensors and the tolerance. Preserve the operation-specific direct
    // mismatch diagnostics by inspecting concrete operands before generic
    // unification reports its lower-level precision pair. Unresolved generic
    // wrappers pass this precheck and are constrained by the shared variable.
    if matches!(func_name.as_deref(), Some("test_assert_close_tensor"))
        && let Some(rejected) = reject_test_assert_close_tensor_operand_dtypes(
            node,
            &arg_tys,
            subst,
            errors,
            // No suspension: this pass runs before signature unification has
            // constrained anything, and `finish_unified_app` calls the same
            // function again on the types unification produced. That later call
            // is the one that decides, so it is the one that suspends.
            None,
            &Type::Unit,
            product,
        )
    {
        return_with_collection_cleanup!(rejected);
    }

    // [05-DIM-3]: the semantic registry owns axis dtype slots. `concat`
    // is overloaded with ordinary list concatenation and therefore runs
    // the same shared gate only after its tensor-list arm is identified in
    // `postprocess_application`; every unambiguous builtin is screened here.
    if let Some(fname) = func_name.as_deref()
        && fname != "concat"
        && let Err(rejected) = enforce_registered_axis_dtypes(
            fname,
            &arg_tys,
            node,
            CheckSite::Expr(expr),
            subst,
            errors,
        )
    {
        return_with_collection_cleanup!(rejected);
    }

    // [05-DIM-1] fix-naming diagnostic for expand's extent slot: a wrong
    // size dtype would otherwise surface as the scheme unification's bare
    // `precision mismatch` pair. Pre-check the resolved size type here so
    // the rejection names the fix, mirroring shrink/pad/stride/reshape.
    if let Some(callee @ ("expand" | "insert")) = func_name.as_deref()
        && arg_tys.len() >= 3
        && let Type::Prim(p) = subst.apply(&arg_tys[2])
        && p != Prim::Int64
    {
        return_with_collection_cleanup!(report_at_check_site(
            errors,
            CheckError::with_types(
                CheckErrorKind::TypeMismatch,
                with_node_provenance(
                    node,
                    format!(
                        "{callee} argument 3 (size): expected i64, got {} (write Ni64 or cast(N, i64))",
                        p.name()
                    ),
                ),
                "i64".to_string(),
                p.name().to_string(),
                vec![],
            ),
            CheckSite::Expr(expr),
        ));
    }

    // If the *callee* is Error, propagate. With no resolved callee scheme
    // there is no return type to produce: `subst.apply(&ret_tv)` (see below)
    // would leak a bare `Var`, because the `unify` against `expected_fn` is
    // Error-permissive and never binds `ret_tv`. A callee-`Error` already
    // carries its own diagnostic from the failed `var` lookup, so this
    // early-out suppresses no checking.
    //
    // chelis#773: we do NOT short-circuit on an Error-typed *argument*. The
    // previous arm collapsed the whole call to `Type::Error` the moment any
    // (non-`expand`-size) argument inferred to `Error`, which disabled type
    // checking of every sibling argument in the same call — one reported
    // error in one slot masked genuine mismatches in the others. Error-typed
    // arguments now flow into the per-slot `unify` below, which treats
    // `Type::Error` as permissive (`unify.rs`: `(Error, _) | (_, Error) =>
    // Ok(())`): siblings still check against their scheme slots, and the
    // call returns the callee's resolved return type instead of `Error`.
    //
    // The chelis#530 `expand` SIZE-slot concern is subsumed, not lost: both
    // expand size gates read the size from the raw AST (not from
    // `arg_tys[2]`), and now that no argument short-circuits here, an
    // Error-typed size slot always reaches the per-form located diagnostic.
    // A genuinely sourced size never infers to `Error`.
    if let Some(err) = propagate_if_error([&func_ty]) {
        return_with_collection_cleanup!(err);
    }

    // Issue Chelis-Lang/chelis#218 R3 HIGH-CONCAT: when `Cons` is
    // called with two concrete tensor-typed args (head: tensor,
    // tail: List<tensor>), skip the generic per-dim equality
    // unification and produce a per-axis join (Lit(n) when both
    // dims are Lit(n) and equal; Wildcard otherwise). The generic
    // `unify` recursion walks dims pairwise and rejects
    // `Lit(2) vs Lit(3)`, which breaks `concat([a, b], 0)` after
    // the to_tensor source fix made nested-list literals emit
    // concrete dims.
    //
    // Precondition guards:
    //   * exactly two args (Cons signature)
    //   * head is a concrete tensor type
    //   * tail is `List<tensor[...]>` with a concrete tensor element
    //   * head and tail-element have matching rank (no rank join;
    //     mismatched ranks remain structural errors)
    //   * head and tail-element have matching precision (no
    //     precision join; mismatched precisions would mask real
    //     type errors)
    //
    // When any precondition fails, fall through to the generic
    // unify path so other Cons shapes (e.g. `Cons(scalar, list)` or
    // `Cons(head, Nil)` where `Nil`'s tvar binds the element type)
    // keep their existing semantics.
    if matches!(func_name.as_deref(), Some("Cons")) && arg_tys.len() == 2 {
        let head_resolved = subst.apply(&arg_tys[0]);
        let tail_resolved = subst.apply(&arg_tys[1]);
        if let (Type::Tensor(head_dims, head_prec), Type::Adt(list_name, list_args)) =
            (&head_resolved, &tail_resolved)
            && list_name == "List"
            && list_args.len() == 1
            && let Type::Tensor(tail_dims, tail_prec) = subst.apply(&list_args[0])
        {
            if head_dims.len() != tail_dims.len() {
                // chelis#255: surface the rank-uniform rule and the
                // reshape/flatten remediation in the diagnostic itself,
                // so users (and agents reading JSON output) are not
                // left guessing why a `List[tensor[k, f32]]` rejected
                // a rank-mixed literal. The dim slot `k` is a
                // dimension variable, not a shape-vector variable;
                // see spec/04-type-system.md §4.5.1.
                return_with_collection_cleanup!(report_at_check_site(
                    errors,
                    CheckError::new(
                        CheckErrorKind::DimensionMismatch,
                        with_node_provenance(
                            node,
                            format!(
                                "call `Cons` arguments 1 and 2: list element rank mismatch: {} dims vs {} dims; \
                                 List[tensor[...]] requires rank-uniform elements \
                                 (the dim slot is a dimension variable, not a \
                                 shape-vector variable). Reshape or flatten \
                                 elements to a common rank before listing \
                                 (spec/04-type-system.md §4.5.1).",
                                head_dims.len(),
                                tail_dims.len(),
                            ),
                        ),
                        vec![],
                    ),
                    CheckSite::Expr(expr),
                ));
            }
            if let Err(te) = unify_tensor_prec(head_prec, &tail_prec, subst) {
                return_with_collection_cleanup!(report(errors, te.into()));
            }
            // Per-axis join (chelis#218 concat ergonomics, tightened
            // by chelis#272). Resolve each dim through the current
            // substitution first so already-bound dim variables compare
            // as their concrete value.
            //
            //   * equal concrete literals or equal names  -> keep them;
            //   * genuinely-mismatched concrete literals
            //     (e.g. `Lit(2)` vs `Lit(3)`) or mismatched concrete
            //     names                                    -> widen to
            //     `Wildcard`. This is the deliberate #218 behavior that
            //     lets bare `concat([a, b], axis)` accept ragged
            //     concrete axes; and
            //   * any pair that involves a dimension *variable*
            //     (a declared rigid dim parameter such as `k`/`m`)
            //     -> `unify_dim` the two dims instead of widening.
            //
            // The last arm is the #272 fix: the old `_ => Wildcard`
            // erased named dim variables, so a list body that violated
            // the §4.4 rigid-distinct-dim guarantee
            // (`def make[k, m](a: tensor[k], b: tensor[m])
            //   -> List[tensor[k]] = [a, b]`) collapsed `k`/`m` to a
            // wildcard before `check_declared_dvars_rigid` ran. Unifying
            // them instead keeps the surviving evidence: distinct rigid
            // dims unify with each other (the guard then reports the
            // collapse) and a `(rigid, concrete)` pair pins the rigid
            // dim to a literal (the guard reports the pin). A
            // `unify_dim` failure here (which the permissive
            // Name/Lit/Wildcard arms make rare) surfaces as a structural
            // dimension mismatch rather than being silently widened.
            let mut joined_dims: Vec<Dim> = Vec::with_capacity(head_dims.len());
            for (h, t) in head_dims.iter().zip(tail_dims.iter()) {
                let hr = subst.apply_dim(h);
                let tr = subst.apply_dim(t);
                let joined = match (
                    ListAxis::classify(&hr, subst),
                    ListAxis::classify(&tr, subst),
                ) {
                    (
                        ListAxis::Concrete {
                            extent: a,
                            label: an,
                        },
                        ListAxis::Concrete {
                            extent: b,
                            label: bn,
                        },
                    ) => {
                        // Names and literal extents are separate observations.
                        // Name/Lit and distinct names widen even at equal sizes;
                        // equal names cannot conceal a known extent conflict.
                        if an == bn && !matches!((a, b), (Some(a), Some(b)) if a != b) {
                            hr.clone()
                        } else {
                            Dim::Wildcard
                        }
                    }
                    // Keep genuine variable constraints for the body-rigidity
                    // guards, and preserve the specified wildcard head bias.
                    _ => {
                        if let Err(te) = unify_dim(&hr, &tr, subst) {
                            return_with_collection_cleanup!(report(errors, te.into()));
                        }
                        subst.apply_dim(&hr)
                    }
                };
                joined_dims.push(joined);
            }
            let joined_prec = subst.apply_tensor_prec(head_prec);
            let elem = Type::Tensor(joined_dims, joined_prec);
            return_with_collection_cleanup!(Type::Adt("List".to_string(), vec![elem]));
        }
    }

    let direct_collection_builtin = matches!(
        func_name.as_deref(),
        Some("len" | "index" | "append" | "concat")
    );
    let aggregate_call_context = match func_name.as_deref() {
        Some(operation @ ("fold" | "scan")) => Some((
            operation,
            "expects a callback whose accumulator/result type matches the initial accumulator",
        )),
        _ => None,
    };
    let mut ret_tv = match unify_checked_call_contract(
        expr,
        source_func_name.as_deref(),
        &func_ty,
        &arg_tys,
        aggregate_call_context,
        vg,
        subst,
        errors,
        product,
    ) {
        Ok(ret_ty) => ret_ty,
        Err(rejected) => return_with_collection_cleanup!(rejected),
    };
    product.replay_ready_shape_checks(vg, subst, adt_reg, errors);
    if direct_collection_builtin {
        subst.discard_collection_contracts(&callee_collection_contracts);
    } else if matches!(
        subst.apply(&func_ty),
        Type::Fn(ref params, _) if params.len() == arg_tys.len()
    ) {
        let alternatives = product.callable_result_alternatives(&func_ty, subst);
        let tensor_concat = subst
            .collection_contracts_include_concat(&callee_collection_contracts, &alternatives)
            .then(|| tensor_concat_call_evidence(kids, env, subst, errors, product));
        subst.prepare_collection_contract_call(
            &callee_collection_contracts,
            &alternatives,
            tensor_concat,
        );
    }
    let related_results = product
        .result_equations_for(&subst.apply(&ret_tv), subst)
        .into_iter()
        .flat_map(|equation| equation.types().into_iter().cloned().collect::<Vec<_>>())
        .collect::<Vec<_>>();
    if let Some(checked_result) = subst.finish_collection_contract_application(
        collection_contract_mark,
        &arg_tys,
        &ret_tv,
        &related_results,
    ) {
        ret_tv = checked_result;
    }
    absorb_runtime_extents_into_call_variables(&instantiation_dvars, subst);
    // chelis#1512: watch whether the eager pass rejects this call. A
    // route can suspend on one operand and then reject on another in
    // the same pass, and the replay re-enters the whole route, so the
    // rejection would be reported a second time. A call that has
    // already failed has nothing left to decide, so its suspension is
    // cancelled here.
    let checkpoint = errors.checkpoint();
    let contract_name = func_name.clone();
    let applied = finish_unified_app(
        CheckSite::Expr(expr),
        node,
        kids,
        func_name,
        arg_tys,
        ret_tv,
        env,
        vg,
        subst,
        adt_reg,
        errors,
        product,
        expected_result,
    );
    if errors.iter_since(checkpoint).next().is_some() {
        subst.cancel_collection_contract_application(collection_contract_mark);
        product.cancel_post_app_check_for(node);
    } else {
        product.record_call_result_contracts(&func_ty, contract_name.as_deref(), env, subst);
    }
    applied
}

/// Bind every alias class this application's instantiation minted a member of
/// that met a runtime extent `*` and that no argument claimed (chelis#1801).
///
/// `spec/04-type-system.md` section 3.2 Application: a dimension variable
/// minted by an application's instantiation that unifies with a runtime
/// extent `*`, and that no argument of that application binds to a literal or
/// named dimension, denotes that runtime extent and is `*` in the
/// application's result. A literal or name another argument of the same
/// application binds to it is a claim on the runtime extent, checked by a
/// section 4.7 guard.
///
/// The three questions below are that sentence and nothing else. Unification
/// identifies dimension variables, so the subject of every clause is the
/// ALIAS CLASS rather than the variable this application happens to hold:
/// "unifies with a runtime extent" is the class's wildcard flag, "binds to a
/// literal" is `constraint_dim` answering a non-variable, and "or named
/// dimension" is the class's pin, which covers both an authored name that
/// `bind_dvar` recorded as a label without binding and a declared binder of
/// the definition under check. `Subst` maintains both flags ON the class root
/// and merges them at every union, so one question per class is a complete
/// answer.
///
/// chelis#1925's rounds 1 and 2 both reported the same defect class against
/// weaker forms of this: asking the variable alone missed a class rooted by a
/// polymorphic argument's own instantiation, and asking two ends missed a
/// three-member class carrying the meeting on the middle one. Neither the
/// two-end query nor the "some instantiation minted this root" membership
/// index survives, because both approximated a sentence the representation
/// now states.
///
/// Why here, and not in `unify_dim`. Binding the variable where it meets the
/// wildcard freezes it: `f(x: tensor[d, f32], y: tensor[d, f32])` applied to
/// a runtime-extent argument and a `tensor[3, f32]` one would read `*` when
/// the wildcard came first and `3` when it came second, so the answer would
/// depend on argument order. That is the loss the Wildcard-against-Var
/// invariant on `unify_dim` exists to avoid, and it is why this runs after
/// the WHOLE call has unified: by then every argument has had its chance to
/// constrain the class, and the orders agree.
///
/// Why here, and not at generalization. A declared result reaches the
/// definition boundary before generalization does, so a signature such as
/// `-> tensor[100, f32]` would have pinned the variable to `100` first and
/// the absorption could never see it. It would also need a let/def
/// distinction that this rule does not.
///
/// `constraint_dim`, not `apply_dim`, decides "still unbound", because that
/// is the question `constraint_dim` answers: it resolves the variable and
/// nothing else. `apply_dim` deliberately answers `Var(v)` for a LABELLED
/// variable that resolved to a concrete dim, to keep the label's identity
/// available to name-sensitive operations, so it cannot tell a free class
/// from one an argument just bound to a literal.
/// `unify::tests::a_recorded_meeting_does_not_by_itself_mean_the_class_is_still_free`
/// builds that state directly and locks which predicate answers correctly in
/// it.
fn absorb_runtime_extents_into_call_variables(instantiation_dvars: &[DimVar], subst: &mut Subst) {
    for &dv in instantiation_dvars {
        let Dim::Var(root) = subst.constraint_dim(&Dim::Var(dv)) else {
            // Bound to a literal, a name or a rank: an argument of this
            // application supplied a claim on the runtime extent, and
            // spec/04-type-system.md section 4.7 guards it at run time.
            continue;
        };
        if !subst.dvar_class_met_wildcard(root) || subst.dvar_class_is_binder_pinned(root) {
            continue;
        }
        subst.insert_dim(root, Dim::Wildcard);
    }
}
