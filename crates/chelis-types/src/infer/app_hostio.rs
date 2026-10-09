//! Concrete signatures for host-lane numeric-text and CSV builtins.
//!
//! JSON is source-defined by `Std.Io.Json`; no compiler-owned JSON value or
//! compatibility builtin is registered here.

use super::*;

fn csv_table_ty() -> Type {
    Type::Adt(
        "List".to_string(),
        vec![Type::Adt(
            "Dict".to_string(),
            vec![Type::Prim(Prim::String), Type::Prim(Prim::String)],
        )],
    )
}

fn reject_host_builtin_slot(
    errors: &mut DiagnosticSink<'_>,
    node: &DeepNode,
    fname: &str,
    slot_description: &str,
    got: &Type,
) -> Type {
    report(
        errors,
        CheckError::new(
            CheckErrorKind::TypeMismatch,
            with_node_provenance(
                node,
                format!("{fname} expects {slot_description}, got {got}"),
            ),
            vec![],
        ),
    )
}

fn unify_host_slot(
    errors: &mut DiagnosticSink<'_>,
    node: &DeepNode,
    fname: &str,
    slot: &Type,
    expected: &Type,
    slot_description: &str,
    subst: &mut Subst,
) -> Option<Type> {
    // chelis#1489: decide this BEFORE unifying. `unify` binds an unbound
    // variable to `expected` and returns Ok, so a variable check placed after
    // the unification can never see one -- it would be dead code, and an
    // earlier revision of this change shipped it that way.
    //
    // A concrete type still fails immediately: this defers the decision on an
    // unknown, it does not soften a known-bad operand.
    if let Type::Var(tv) = subst.apply(slot) {
        subst.record_deferred_tensor_operand(
            tv,
            DeferredOperandGate::HostSlot {
                fname: fname.to_string(),
                description: slot_description.to_string(),
                expected: Box::new(expected.clone()),
            },
        );
        return None;
    }
    unify_host_slot_eager(errors, node, fname, slot, expected, slot_description, subst)
}

/// Unify a host-lane slot and reject an unresolved operand immediately: the
/// behaviour every slot had before chelis#1489.
///
/// `round_to` still uses this. The deferring form carries ONE expected type
/// and discharge unifies the operand against it, but `round_to` accepts more
/// than one: the arm above admits [05-OP-1]'s four float dtypes and returns
/// the operand's own precision. Deferring it against `f64` alone made an
/// operand that later bound to `f32` emit a rejection naming `f32` as
/// acceptable -- false on its face, and a disagreement with the eager arm. The
/// ten csv slots each accept exactly one type, so they defer correctly.
///
/// Converting this slot therefore needs the gate to carry an accepted SET and
/// a deferred result. That is more than the relabelling the rest of the
/// conversion is, so it stays on chelis#1489 rather than being bolted on here.
fn unify_host_slot_eager(
    errors: &mut DiagnosticSink<'_>,
    node: &DeepNode,
    fname: &str,
    slot: &Type,
    expected: &Type,
    slot_description: &str,
    subst: &mut Subst,
) -> Option<Type> {
    if unify(slot, expected, subst).is_err() {
        return Some(reject_host_builtin_slot(
            errors,
            node,
            fname,
            slot_description,
            &subst.apply(slot),
        ));
    }
    None
}

#[allow(clippy::too_many_arguments)]
fn loose_integer_host_slot(
    errors: &mut DiagnosticSink<'_>,
    node: &DeepNode,
    fname: &str,
    slot: &Type,
    slot_description: &str,
    subst: &mut Subst,
) -> Option<Type> {
    let ok = match slot {
        Type::Prim(p) => p.is_integer(),
        Type::Error(_) => true,
        Type::Var(_) => unify(slot, &Type::Prim(Prim::Int64), subst).is_ok(),
        _ => false,
    };
    (!ok).then(|| {
        reject_host_builtin_slot(errors, node, fname, slot_description, &subst.apply(slot))
    })
}

/// [05-OP-1]: `round_to` admits every active float dtype and returns it.
pub(super) fn check_round_to_builtin_signature(
    node: &DeepNode,
    arg_tys: &[Type],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    const FNAME: &str = "round_to";
    if arg_tys.len() != 2 {
        return report_builtin_arity(errors, node, CheckSite::Node(node), FNAME, 2, arg_tys.len());
    }
    let operand = type_for_readonly_check(&arg_tys[0], subst);
    let result = match &operand {
        Type::Prim(p @ (Prim::F64 | Prim::F32 | Prim::F16 | Prim::Bf16)) => Type::Prim(*p),
        Type::Error(_) => Type::Prim(Prim::F64),
        // chelis#1489: deliberately NOT deferred -- see `unify_host_slot_eager`.
        Type::Var(_) => {
            if let Some(error) = unify_host_slot_eager(
                errors,
                node,
                FNAME,
                &operand,
                &Type::Prim(Prim::F64),
                "a float (f64, f32, f16, or bf16) first argument",
                subst,
            ) {
                return error;
            }
            Type::Prim(Prim::F64)
        }
        other => {
            return reject_host_builtin_slot(
                errors,
                node,
                FNAME,
                "a float (f64, f32, f16, or bf16) first argument",
                other,
            );
        }
    };
    let places = type_for_readonly_check(&arg_tys[1], subst);
    if let Some(error) = loose_integer_host_slot(
        errors,
        node,
        FNAME,
        &places,
        "an integer `places` second argument",
        subst,
    ) {
        return error;
    }
    result
}

/// [05-OP-2..5]'s CSV carrier is exactly `List[Dict[string,string]]`.
pub(super) fn check_csv_builtin_signature(
    fname: &str,
    node: &DeepNode,
    arg_tys: &[Type],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    let expected_arity = match fname {
        "parse_csv" | "to_csv" | "csv_nrows" | "csv_cols" => 1,
        "csv_f64" | "csv_int" | "csv_str" => 3,
        _ => 2,
    };
    if arg_tys.len() != expected_arity {
        return report_builtin_arity(
            errors,
            node,
            CheckSite::Node(node),
            fname,
            expected_arity,
            arg_tys.len(),
        );
    }

    macro_rules! require_slot {
        ($idx:expr, $expected:expr, $desc:expr) => {{
            let slot = type_for_readonly_check(&arg_tys[$idx], subst);
            if let Some(error) =
                unify_host_slot(errors, node, fname, &slot, &$expected, $desc, subst)
            {
                return error;
            }
        }};
    }

    if fname != "parse_csv" {
        require_slot!(
            0,
            csv_table_ty(),
            "List[Dict[string,string]] as the first argument"
        );
    }
    match fname {
        "parse_csv" => {
            require_slot!(0, Type::Prim(Prim::String), "a string argument");
            csv_table_ty()
        }
        "to_csv" => Type::Prim(Prim::String),
        "csv_nrows" => Type::Prim(Prim::Int64),
        "csv_cols" => Type::Adt("List".to_string(), vec![Type::Prim(Prim::String)]),
        "csv_f64s" | "csv_ints" | "csv_strs" => {
            require_slot!(
                1,
                Type::Prim(Prim::String),
                "a column-name string second argument"
            );
            match fname {
                "csv_f64s" => Type::Adt("List".to_string(), vec![Type::Prim(Prim::F64)]),
                "csv_ints" => Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]),
                _ => Type::Adt("List".to_string(), vec![Type::Prim(Prim::String)]),
            }
        }
        "csv_f64" | "csv_int" | "csv_str" => {
            let row = type_for_readonly_check(&arg_tys[1], subst);
            if let Some(error) = loose_integer_host_slot(
                errors,
                node,
                fname,
                &row,
                "an integer row index second argument",
                subst,
            ) {
                return error;
            }
            require_slot!(
                2,
                Type::Prim(Prim::String),
                "a column-name string third argument"
            );
            match fname {
                "csv_f64" => Type::Prim(Prim::F64),
                "csv_int" => Type::Prim(Prim::Int64),
                _ => Type::Prim(Prim::String),
            }
        }
        other => unreachable!("check_csv_builtin_signature dispatched on `{other}`"),
    }
}

/// [05-OP-79]: `mmap_tensor(mapped, offset, count, T)` borrows a mapped file,
/// takes an exact `i64` byte offset and element count, and returns a rank-one
/// tensor of the dtype its type-node child `T` states (spec/03 §6.4). A
/// literal count gives a literal extent; any other count a fresh extent.
#[allow(clippy::too_many_arguments)]
pub(super) fn infer_mmap_tensor_app(
    expr: &deep::Expr,
    node: &DeepNode,
    env: &mut Env,
    vg: &mut VarGen,
    subst: &mut Subst,
    adt_reg: &AdtRegistry,
    errors: &mut DiagnosticSink<'_>,
    product: &mut InferenceProduct,
) -> Type {
    const FNAME: &str = "mmap_tensor";
    let kids = node.children_slice();
    let dtype_child = kids.last().filter(|child| {
        matches!(
            stamped_parts(child),
            Some((DeepTag::TPrim | DeepTag::TVar, _, _))
        )
    });
    let (Some(dtype_child), 5) = (dtype_child, kids.len()) else {
        return report_at_check_site(
            errors,
            CheckError::with_types(
                CheckErrorKind::ArityMismatch,
                format!(
                    "mmap_tensor expects (mapped, offset, count, T) with a dtype `T` as its \
                     fourth argument, got {} argument(s) ([05-OP-79])",
                    kids.len().saturating_sub(1)
                ),
                "(mapped, offset, count, T)".to_string(),
                format!("{} argument(s)", kids.len().saturating_sub(1)),
                vec![],
            ),
            CheckSite::Expr(expr),
        );
    };
    let callee_ty = super::app::infer_callee(&kids[0], env, vg, subst, adt_reg, errors, product);
    let arg_tys: Vec<Type> = kids[1..4]
        .iter()
        .map(|arg| infer_expr(arg, env, vg, subst, adt_reg, errors, product))
        .collect();
    if let Some(err) = propagate_if_error(arg_tys.iter()) {
        return err;
    }
    let mapped = Type::Adt("MappedFile".to_string(), Vec::new());
    let slots = [
        (&mapped, "a MappedFile handle"),
        (&Type::Prim(Prim::Int64), "an i64 byte offset"),
        (&Type::Prim(Prim::Int64), "an i64 element count"),
    ];
    for (slot, (expected, description)) in arg_tys.iter().zip(slots) {
        if let Some(rejected) =
            unify_host_slot_eager(errors, node, FNAME, slot, expected, description, subst)
        {
            return rejected;
        }
    }
    let dtype = {
        let mut resolver = DeepTypeResolver::new(
            TypeUseSite::Annotation,
            annotation_binder_mode(env),
            adt_reg.resolution_env(),
            vg,
            errors,
        )
        .with_diagnostic_owner(expr);
        match resolver.resolve(dtype_child) {
            Ok(dtype) => dtype.into_type(),
            Err(witness) => return propagate(&witness),
        }
    };
    // A `t-prim` child must name a data element dtype. A `t-var` child names
    // a dtype binder whose declared bound admits only data element dtypes
    // ([04-DTYPE-2]): every §5.9 family and explicit set does, and an
    // unbounded binder does not, since it could stand for `string`.
    let precision = match (&dtype, stamped_parts(dtype_child)) {
        (Type::Prim(prim), _) if prim.is_numeric() || *prim == Prim::Bool => {
            TensorPrec::Concrete(*prim)
        }
        (Type::Var(tv), Some((DeepTag::TVar, _, _)))
            if matches!(
                subst.tvar_restriction(*tv),
                Some(
                    TypeVarRestriction::ActiveFloat
                        | TypeVarRestriction::ActiveInt
                        | TypeVarRestriction::ActiveNumeric
                        | TypeVarRestriction::ActiveSet(_)
                )
            ) =>
        {
            TensorPrec::Var(*tv)
        }
        _ => {
            // A binder is named as written, not by its inference variable.
            let written = match stamped_parts(dtype_child) {
                Some((DeepTag::TVar, _, [name])) => symbol_name(name)
                    .map(|name| format!("the unbounded binder `{name}`"))
                    .unwrap_or_else(|| dtype.to_string()),
                _ => dtype.to_string(),
            };
            return report_at_check_site(
                errors,
                CheckError::new(
                    CheckErrorKind::TypeMismatch,
                    format!(
                        "mmap_tensor's dtype argument must be an active data element dtype or \
                         a dtype binder bounded by a dtype family or set, not {written} ([05-OP-79])"
                    ),
                    vec![],
                ),
                CheckSite::Expr(expr),
            );
        }
    };
    let extent = match extract_int_literal(&kids[3]) {
        Some(count) if count >= 0 => Dim::Lit(count),
        _ => Dim::Wildcard,
    };
    let result = Type::Tensor(vec![extent], precision);
    let signature = Type::Fn(
        vec![mapped, Type::Prim(Prim::Int64), Type::Prim(Prim::Int64)],
        Box::new(result.clone()),
    );
    if let Err(error) = unify(&callee_ty, &signature, subst) {
        return report_at_check_site(errors, error.into(), CheckSite::Expr(expr));
    }
    result
}
