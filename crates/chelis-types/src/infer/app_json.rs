//! Concrete argument/return contracts for the host-lane JSON I/O builtin
//! family (chelis#890): the parse/serialize pair, the dot-path accessors,
//! the output constructors, and `round_to`. Split from `app_post.rs` when
//! that file crossed the 3000-line architecture limit -- this family is a
//! single self-contained responsibility with one entry point.

use super::*;

/// Concrete argument/return contracts for the host-lane JSON I/O builtins
/// (chelis#890): `parse_json`, `to_json`, the dot-path accessors
/// (`json_f64`/`json_str`/`json_list`/`json_f64s`), the output
/// constructors (`jnum`/`jstr`/`jlist`/`jdict`/`json_set`), and
/// `round_to`.
///
/// The env schemes (`builtin_env`) only declare arity; this arm pins the
/// real types, and — per the chelis#891 review (finding 6) — it pins them
/// by **unification**, not by permissive matching: an argument whose type
/// is still a `Type::Var` (an un-annotated parameter, say) is unified
/// with the slot's expected type instead of waved through, so
/// `fn (doc) -> json_f64(doc, "a")` genuinely types `Json -> f64` rather
/// than `forall a. a -> f64`.
///
/// Numeric slots: `round_to` takes exactly an f64 operand, which is the
/// dtype set [05-OP-1] declares (spec/05-risc-primitives.md §3.7); its
/// `places` slot accepts any integer precision, unifying an unresolved
/// `Var` with the canonical int64. `jnum` is exactly f64 on a different
/// rule (chelis#891 review finding 7): its output feeds the byte-exact
/// `to_json` channel, and silently widening an f32 literal would
/// serialize `0.1f32` as `0.10000000149011612`. Both diagnostics name
/// the fix (suffix the literal or cast).
pub(super) fn check_json_builtin_signature(
    fname: &str,
    list: &deep::List,
    arg_tys: &[Type],
    subst: &mut Subst,
    errors: &mut DiagnosticSink<'_>,
) -> Type {
    fn json_ty() -> Type {
        Type::Adt("Json".to_string(), Vec::new())
    }

    let expected_arity: usize = match fname {
        "parse_json" | "to_json" | "jnum" | "jstr" | "jlist" | "jdict" => 1,
        "json_set" => 3,
        _ => 2,
    };
    if arg_tys.len() != expected_arity {
        return report_builtin_arity(errors, list, fname, expected_arity, arg_tys.len());
    }

    let mut reject = |slot_description: String, got: &Type| -> Type {
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
    };

    // Slot checks: each resolves the argument fresh (earlier unifications
    // may have refined it), then unifies with the expected type. `unify`
    // treats `Type::Error` as success, so already-diagnosed slots do not
    // cascade.
    macro_rules! require_slot {
        ($idx:expr, $expected:expr, $desc:expr) => {{
            let slot = type_for_readonly_check(&arg_tys[$idx], subst);
            if unify(&slot, &$expected, subst).is_err() {
                return reject($desc.to_string(), &subst.apply(&slot));
            }
        }};
    }

    // `round_to`'s two slots accept any concrete float/integer precision;
    // only an unresolved Var is pinned (to the canonical f64/int64) so the
    // contract is never vacuous through an un-annotated parameter.
    macro_rules! require_loose_numeric_slot {
        ($idx:expr, $is_kind:ident, $default:expr, $desc:expr) => {{
            let slot = type_for_readonly_check(&arg_tys[$idx], subst);
            let ok = match &slot {
                Type::Prim(p) if p.$is_kind() => true,
                Type::Error(_) => true,
                Type::Var(_) => unify(&slot, &Type::Prim($default), subst).is_ok(),
                _ => false,
            };
            if !ok {
                return reject($desc.to_string(), &subst.apply(&slot));
            }
        }};
    }

    match fname {
        "parse_json" => {
            require_slot!(0, Type::Prim(Prim::String), "a string argument");
            json_ty()
        }
        "to_json" => {
            require_slot!(
                0,
                json_ty(),
                "a Json argument (from `parse_json` or a J* constructor)"
            );
            Type::Prim(Prim::String)
        }
        "json_f64" | "json_int" | "json_str" | "json_list" | "json_f64s" => {
            require_slot!(
                0,
                json_ty(),
                "a Json first argument (from `parse_json` or a J* constructor)"
            );
            require_slot!(
                1,
                Type::Prim(Prim::String),
                "a dot-separated string path second argument (e.g. \"a.b.c\")"
            );
            match fname {
                "json_f64" => Type::Prim(Prim::F64),
                "json_int" => Type::Prim(Prim::Int64),
                "json_str" => Type::Prim(Prim::String),
                "json_list" => Type::Adt("List".to_string(), vec![json_ty()]),
                _ => Type::Adt("List".to_string(), vec![Type::Prim(Prim::F64)]),
            }
        }
        "jnum" => {
            require_slot!(
                0,
                Type::Prim(Prim::F64),
                "an f64 argument (suffix the literal, `0.1f64`, or use cast(n, f64); \
                 an f32 value would quantize through the byte-exact serializer)"
            );
            json_ty()
        }
        "jint" => {
            require_slot!(
                0,
                Type::Prim(Prim::Int64),
                "an int64 argument (suffix the literal, `1i64`, or use cast(n, int64))"
            );
            json_ty()
        }
        "jstr" => {
            require_slot!(0, Type::Prim(Prim::String), "a string argument");
            json_ty()
        }
        "jlist" => {
            let slot = type_for_readonly_check(&arg_tys[0], subst);
            match &slot {
                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                    let element = subst.apply(&args[0]);
                    if unify(&element, &json_ty(), subst).is_err() {
                        return reject("List[Json] input".to_string(), &subst.apply(&element));
                    }
                }
                Type::Error(_) => {}
                Type::Var(_) => {
                    let expected = Type::Adt("List".to_string(), vec![json_ty()]);
                    if unify(&slot, &expected, subst).is_err() {
                        return reject("List[Json] input".to_string(), &subst.apply(&slot));
                    }
                }
                other => return reject("List[Json] input".to_string(), other),
            }
            json_ty()
        }
        "jdict" => {
            let entry_ty = || Type::Tuple(vec![Type::Prim(Prim::String), json_ty()]);
            let slot = type_for_readonly_check(&arg_tys[0], subst);
            match &slot {
                Type::Adt(name, args) if name == "List" && args.len() == 1 => {
                    match subst.apply(&args[0]) {
                        Type::Tuple(items) if items.len() == 2 => {
                            let key_ty = subst.apply(&items[0]);
                            if unify(&key_ty, &Type::Prim(Prim::String), subst).is_err() {
                                return reject(
                                    "(string, Json) entry tuples (string keys)".to_string(),
                                    &subst.apply(&key_ty),
                                );
                            }
                            let value_ty = subst.apply(&items[1]);
                            if unify(&value_ty, &json_ty(), subst).is_err() {
                                return reject(
                                    "(string, Json) entry tuples (Json values)".to_string(),
                                    &subst.apply(&value_ty),
                                );
                            }
                        }
                        Type::Error(_) => {}
                        element @ Type::Var(_) => {
                            if unify(&element, &entry_ty(), subst).is_err() {
                                return reject(
                                    "List[(string, Json)] input".to_string(),
                                    &subst.apply(&element),
                                );
                            }
                        }
                        other => {
                            return reject("List[(string, Json)] input".to_string(), &other);
                        }
                    }
                }
                Type::Error(_) => {}
                Type::Var(_) => {
                    let expected = Type::Adt("List".to_string(), vec![entry_ty()]);
                    if unify(&slot, &expected, subst).is_err() {
                        return reject(
                            "List[(string, Json)] input".to_string(),
                            &subst.apply(&slot),
                        );
                    }
                }
                other => return reject("List[(string, Json)] input".to_string(), other),
            }
            json_ty()
        }
        "json_set" => {
            require_slot!(
                0,
                json_ty(),
                "a Json first argument (from `parse_json` or a J* constructor)"
            );
            require_slot!(
                1,
                Type::Prim(Prim::String),
                "a dot-separated string path second argument (e.g. \"a.b.c\")"
            );
            require_slot!(
                2,
                json_ty(),
                "a Json third argument (wrap raw values with jnum/jstr/jlist/jdict)"
            );
            json_ty()
        }
        "round_to" => {
            // f64-only, per [05-OP-1] (spec/05-risc-primitives.md
            // section 3.7, authored at chelis#891): the implementation
            // widens the operand to f64, rounds decimally, and re-narrows,
            // which at f32 computes at other than the declared arithmetic
            // width -- non-conforming under [04-NUM-8], whose atom
            // deliberately has no exception vocabulary. The narrower
            // widths return when the atom declares their per-dtype rule at
            // declared widths. Same strictness shape as `jnum` below; the
            // diagnostic names the remediation.
            require_slot!(
                0,
                Type::Prim(Prim::F64),
                "an f64 first argument (spec/05-risc-primitives.md \
                 [05-OP-1] declares `round_to`'s dtype set as f64 only; \
                 suffix the literal `f64` or `cast` the operand)"
            );
            require_loose_numeric_slot!(
                1,
                is_integer,
                Prim::Int64,
                "an integer `places` second argument"
            );
            Type::Prim(Prim::F64)
        }
        // The guarantee: this function is reached only through the JSON
        // dispatch arm, whose match list and the arity table above are the
        // same closed name set. A name here means those two lists drifted
        // apart in one edit, which is a compiler bug, never a program
        // condition (loud_unsupported.md section C1.3: name the guarantee,
        // not the input).
        other => unreachable!(
            "check_json_builtin_signature is reachable only for the closed \
             host-lane JSON builtin set that its caller dispatches on; \
             `{other}` means the caller's name list and this function's \
             arms drifted apart"
        ),
    }
}
