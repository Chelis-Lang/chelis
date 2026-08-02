//! Concrete signature contracts for the host-lane JSON I/O builtins
//! (chelis#890) and CSV I/O builtins (chelis#903).
//!
//! The env schemes (`builtin_env`) only declare arity; these arms pin the
//! real types, and they pin them by **unification**, not permissive
//! matching: an argument whose type is still a `Type::Var` (an
//! un-annotated parameter, say) is unified with the slot's expected type
//! instead of waved through, so `fn (doc) -> json_f64(doc, "a")` genuinely
//! types `Json -> f64` rather than `forall a. a -> f64`.
//!
//! Numeric slots follow spec/05-risc-primitives.md §3.7:
//! - `round_to` ([05-OP-1]) accepts exactly f64 or f32 operands and
//!   preserves the operand dtype; f16/bf16 are rejected loudly (the atom
//!   authors no semantics for them), and an unresolved `Var` pins to f64.
//!   `places` accepts ANY integer precision (unsuffixed literals default
//!   to int32 per spec/04 §5.3), pinning `Var` to int64.
//! - `jnum` is exactly f64 and `jint` exactly int64 ([05-OP-4]): their
//!   output feeds the byte-exact serialization channel, and silently
//!   widening an f32 literal would serialize `0.1f32` as
//!   `0.10000000149011612`. The diagnostics name the fix (suffix or cast).
//! - The CSV row-index slot accepts any integer precision so a bare
//!   `csv_f64(c, 0, "px")` literal works, pinning `Var` to int64.

use super::*;

/// The prelude `Json` value type used by the host-I/O builtin signatures
/// (the chelis#890 JSON family and the chelis#903 CSV family, whose Csv
/// documents ride the same ADT).
fn host_json_ty() -> Type {
    Type::Adt("Json".to_string(), Vec::new())
}

/// Reject one host-I/O builtin slot with a named diagnostic; returns the
/// propagated error type. Shared by [`check_json_builtin_signature`] and
/// [`check_csv_builtin_signature`].
fn reject_host_builtin_slot(
    errors: &mut DiagnosticSink<'_>,
    list: &deep::List,
    fname: &str,
    slot_description: &str,
    got: &Type,
) -> Type {
    report(
        errors,
        CheckError::new(
            CheckErrorKind::TypeMismatch,
            with_macro_provenance(
                &deep::Expr::List(list.clone(), zero_span()),
                format!("{fname} expects {slot_description}, got {got}"),
            ),
            vec![],
        ),
    )
}

/// Unify one resolved slot with its expected type, recording Var bindings
/// in `subst` (contracts are enforced by unification, never by waving
/// `Type::Var` through -- `unify` treats `Type::Error` as success, so
/// already-diagnosed slots do not cascade). On failure emits the standard
/// slot diagnostic; returns the propagated error type to bubble.
fn unify_host_slot(
    errors: &mut DiagnosticSink<'_>,
    list: &deep::List,
    fname: &str,
    slot: &Type,
    expected: &Type,
    slot_description: &str,
    subst: &mut Subst,
) -> Option<Type> {
    if unify(slot, expected, subst).is_err() {
        return Some(reject_host_builtin_slot(
            errors,
            list,
            fname,
            slot_description,
            &subst.apply(slot),
        ));
    }
    None
}

/// A loose integer slot: any concrete integer precision passes
/// (unsuffixed literals default to int32 per spec/04-type-system.md §5.3,
/// the same contract as `string_slice`'s index slots); an unresolved
/// `Var` is pinned to int64 so the contract is never vacuous through an
/// un-annotated parameter.
#[allow(clippy::too_many_arguments)]
fn loose_integer_host_slot(
    errors: &mut DiagnosticSink<'_>,
    list: &deep::List,
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
    if !ok {
        return Some(reject_host_builtin_slot(
            errors,
            list,
            fname,
            slot_description,
            &subst.apply(slot),
        ));
    }
    None
}

/// Concrete argument/return contracts for the host-lane JSON I/O builtins
/// (chelis#890): `parse_json`, `to_json`, the dot-path accessors
/// (`json_f64`/`json_int`/`json_str`/`json_list`/`json_f64s`/`json_ints`),
/// the output constructors (`jnum`/`jint`/`jstr`/`jlist`/`jdict`/
/// `json_set`), and `round_to`.
pub(super) fn check_json_builtin_signature(
    fname: &str,
    list: &deep::List,
    arg_tys: &[Type],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    let expected_arity: usize = match fname {
        "parse_json" | "to_json" | "jnum" | "jint" | "jstr" | "jlist" | "jdict" => 1,
        "json_set" => 3,
        _ => 2,
    };
    if arg_tys.len() != expected_arity {
        return report_builtin_arity(errors, list, fname, expected_arity, arg_tys.len());
    }

    // Slot checks: each resolves the argument fresh (earlier unifications
    // may have refined it), then unifies with the expected type.
    macro_rules! require_slot {
        ($idx:expr, $expected:expr, $desc:expr) => {{
            let slot = type_for_readonly_check(&arg_tys[$idx], subst);
            if let Some(error) =
                unify_host_slot(errors, list, fname, &slot, &$expected, $desc, subst)
            {
                return error;
            }
        }};
    }

    const JSON_FIRST: &str = "a Json first argument (from `parse_json` or a J* constructor)";
    const PATH_SECOND: &str = "a dot-separated string path second argument (e.g. \"a.b.c\")";

    match fname {
        "parse_json" => {
            require_slot!(0, Type::Prim(Prim::String), "a string argument");
            host_json_ty()
        }
        "to_json" => {
            require_slot!(
                0,
                host_json_ty(),
                "a Json argument (from `parse_json` or a J* constructor)"
            );
            Type::Prim(Prim::String)
        }
        "json_f64" | "json_int" | "json_str" | "json_list" | "json_f64s" | "json_ints" => {
            require_slot!(0, host_json_ty(), JSON_FIRST);
            require_slot!(1, Type::Prim(Prim::String), PATH_SECOND);
            match fname {
                "json_f64" => Type::Prim(Prim::F64),
                "json_int" => Type::Prim(Prim::Int64),
                "json_str" => Type::Prim(Prim::String),
                "json_list" => Type::Adt("List".to_string(), vec![host_json_ty()]),
                "json_f64s" => Type::Adt("List".to_string(), vec![Type::Prim(Prim::F64)]),
                _ => Type::Adt("List".to_string(), vec![Type::Prim(Prim::Int64)]),
            }
        }
        "jnum" => {
            require_slot!(
                0,
                Type::Prim(Prim::F64),
                "an f64 argument (suffix the literal, `0.1f64`, or use cast(n, f64); \
                 an f32 value would quantize through the byte-exact serializer, [05-OP-4])"
            );
            host_json_ty()
        }
        "jint" => {
            require_slot!(
                0,
                Type::Prim(Prim::Int64),
                "an int64 argument (suffix the literal, `1i64`, or use cast(n, int64); \
                 [05-OP-4])"
            );
            host_json_ty()
        }
        "jstr" => {
            require_slot!(0, Type::Prim(Prim::String), "a string argument");
            host_json_ty()
        }
        "jlist" => {
            let slot = type_for_readonly_check(&arg_tys[0], subst);
            match &slot {
                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                    let element = subst.apply(&args[0]);
                    if let Some(error) = unify_host_slot(
                        errors,
                        list,
                        fname,
                        &element,
                        &host_json_ty(),
                        "List[Json] input",
                        subst,
                    ) {
                        return error;
                    }
                }
                Type::Error(_) => {}
                Type::Var(_) => {
                    let expected = Type::Adt("List".to_string(), vec![host_json_ty()]);
                    if let Some(error) = unify_host_slot(
                        errors,
                        list,
                        fname,
                        &slot,
                        &expected,
                        "List[Json] input",
                        subst,
                    ) {
                        return error;
                    }
                }
                other => {
                    return reject_host_builtin_slot(
                        errors,
                        list,
                        fname,
                        "List[Json] input",
                        other,
                    );
                }
            }
            host_json_ty()
        }
        "jdict" => {
            let entry_ty = || Type::Tuple(vec![Type::Prim(Prim::String), host_json_ty()]);
            let slot = type_for_readonly_check(&arg_tys[0], subst);
            match &slot {
                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                    match subst.apply(&args[0]) {
                        Type::Tuple(items) if items.len() == 2 => {
                            let key_ty = subst.apply(&items[0]);
                            if let Some(error) = unify_host_slot(
                                errors,
                                list,
                                fname,
                                &key_ty,
                                &Type::Prim(Prim::String),
                                "(string, Json) entry tuples (string keys)",
                                subst,
                            ) {
                                return error;
                            }
                            let value_ty = subst.apply(&items[1]);
                            if let Some(error) = unify_host_slot(
                                errors,
                                list,
                                fname,
                                &value_ty,
                                &host_json_ty(),
                                "(string, Json) entry tuples (Json values)",
                                subst,
                            ) {
                                return error;
                            }
                        }
                        Type::Error(_) => {}
                        element @ Type::Var(_) => {
                            if let Some(error) = unify_host_slot(
                                errors,
                                list,
                                fname,
                                &element,
                                &entry_ty(),
                                "List[(string, Json)] input",
                                subst,
                            ) {
                                return error;
                            }
                        }
                        other => {
                            return reject_host_builtin_slot(
                                errors,
                                list,
                                fname,
                                "List[(string, Json)] input",
                                &other,
                            );
                        }
                    }
                }
                Type::Error(_) => {}
                Type::Var(_) => {
                    let expected = Type::Adt("List".to_string(), vec![entry_ty()]);
                    if let Some(error) = unify_host_slot(
                        errors,
                        list,
                        fname,
                        &slot,
                        &expected,
                        "List[(string, Json)] input",
                        subst,
                    ) {
                        return error;
                    }
                }
                other => {
                    return reject_host_builtin_slot(
                        errors,
                        list,
                        fname,
                        "List[(string, Json)] input",
                        other,
                    );
                }
            }
            host_json_ty()
        }
        "json_set" => {
            require_slot!(0, host_json_ty(), JSON_FIRST);
            require_slot!(1, Type::Prim(Prim::String), PATH_SECOND);
            require_slot!(
                2,
                host_json_ty(),
                "a Json third argument (wrap raw values with jnum/jint/jstr/jlist/jdict)"
            );
            host_json_ty()
        }
        "round_to" => {
            // [05-OP-1]: per-dtype at declared arithmetic widths, f64 and
            // f32 only. The operand's exact binary value decimal-rounds
            // and finalizes ONCE to the operand's own storage width, so
            // the result preserves the operand dtype. f16/bf16 have no
            // authored semantics in the atom and are rejected loudly
            // ([04-NUM-8] provides no exception vocabulary for computing
            // them at another width). An unresolved `Var` pins to f64,
            // the canonical I/O-pipeline dtype.
            let operand = type_for_readonly_check(&arg_tys[0], subst);
            let result = match &operand {
                Type::Prim(p @ (Prim::F64 | Prim::F32)) => Type::Prim(*p),
                Type::Error(_) => Type::Prim(Prim::F64),
                Type::Var(_) => {
                    if let Some(error) = unify_host_slot(
                        errors,
                        list,
                        fname,
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
                        list,
                        fname,
                        "an f64 or f32 first argument ([05-OP-1] authors decimal rounding \
                         at those widths only; cast an f16/bf16 operand explicitly)",
                        other,
                    );
                }
            };
            let places_ty = type_for_readonly_check(&arg_tys[1], subst);
            if let Some(error) = loose_integer_host_slot(
                errors,
                list,
                fname,
                &places_ty,
                "an integer `places` second argument",
                subst,
            ) {
                return error;
            }
            result
        }
        other => unreachable!("check_json_builtin_signature dispatched on `{other}`"),
    }
}

/// Concrete argument/return contracts for the host-lane CSV I/O builtins
/// (chelis#903): `parse_csv`/`to_csv` plus the column accessors
/// (`csv_f64s`/`csv_ints`/`csv_strs`/`csv_nrows`/`csv_cols`/`csv_f64`/
/// `csv_int`/`csv_str`).
///
/// A Csv document rides the prelude `Json` ADT as the fixed shape
/// `{"columns": .., "rows": ..}` (see `runtime/csv.rs`), so document slots
/// type as `Json` -- there is deliberately no `Csv` prelude type, and the
/// `json_*` accessors compose with these documents.
pub(super) fn check_csv_builtin_signature(
    fname: &str,
    list: &deep::List,
    arg_tys: &[Type],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    let expected_arity: usize = match fname {
        "parse_csv" | "to_csv" | "csv_nrows" | "csv_cols" => 1,
        "csv_f64" | "csv_int" | "csv_str" => 3,
        _ => 2,
    };
    if arg_tys.len() != expected_arity {
        return report_builtin_arity(errors, list, fname, expected_arity, arg_tys.len());
    }

    macro_rules! require_slot {
        ($idx:expr, $expected:expr, $desc:expr) => {{
            let slot = type_for_readonly_check(&arg_tys[$idx], subst);
            if let Some(error) =
                unify_host_slot(errors, list, fname, &slot, &$expected, $desc, subst)
            {
                return error;
            }
        }};
    }

    // Every builtin except `parse_csv` takes the Csv document (a Json
    // value) first; the column name is always the last argument.
    if fname != "parse_csv" {
        require_slot!(
            0,
            host_json_ty(),
            "a Csv document first argument (the Json value `parse_csv` returns)"
        );
    }

    match fname {
        "parse_csv" => {
            require_slot!(0, Type::Prim(Prim::String), "a string argument");
            host_json_ty()
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
            let row_ty = type_for_readonly_check(&arg_tys[1], subst);
            if let Some(error) = loose_integer_host_slot(
                errors,
                list,
                fname,
                &row_ty,
                "an integer row index second argument (0-based data row)",
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
