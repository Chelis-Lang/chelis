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
/// than one: the arm above admits `f64` and `f32` and returns the operand's own
/// precision. Deferring it against `f64` alone made an operand that later bound
/// to `f32` emit a rejection naming `f32` as acceptable -- false on its face,
/// and a disagreement with the eager arm. The ten csv slots each accept exactly
/// one type, so they defer correctly.
///
/// `{f64, f32}` is this CHECKER's set, not the normative one. [05-OP-1] declares
/// four operand dtypes -- f64, f32, f16 and bf16 -- and the narrowing to two is
/// a known divergence tracked as chelis#1295, not authority for the pair. Do not
/// cite the atom for it.
///
/// Converting this slot therefore needs the gate to carry an accepted SET, sized
/// by the atom rather than by today's checker, and a deferred result. That is
/// more than the relabelling the rest of the conversion is, so it stays on
/// chelis#1489 rather than being bolted on here.
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

/// [05-OP-1]'s current f32/f64 `round_to` checker boundary.
pub(super) fn check_round_to_builtin_signature(
    node: &DeepNode,
    arg_tys: &[Type],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    const FNAME: &str = "round_to";
    if arg_tys.len() != 2 {
        return report_builtin_arity(errors, node, FNAME, 2, arg_tys.len());
    }
    let operand = type_for_readonly_check(&arg_tys[0], subst);
    let result = match &operand {
        Type::Prim(p @ (Prim::F64 | Prim::F32)) => Type::Prim(*p),
        Type::Error(_) => Type::Prim(Prim::F64),
        // chelis#1489: deliberately NOT deferred -- see `unify_host_slot_eager`.
        Type::Var(_) => {
            if let Some(error) = unify_host_slot_eager(
                errors,
                node,
                FNAME,
                &operand,
                &Type::Prim(Prim::F64),
                "an f64 or f32 first argument",
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
                "an f64 or f32 first argument",
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
        return report_builtin_arity(errors, node, fname, expected_arity, arg_tys.len());
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
