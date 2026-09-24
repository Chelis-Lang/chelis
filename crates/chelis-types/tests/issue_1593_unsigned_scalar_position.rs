//! chelis#1593 at the checker: an unsigned dtype spelling in a SCALAR type
//! position is rejected with the `spec/04-type-system.md` §1.1.1 / §1.1.2
//! diagnostic, the same one the cast target and the `tensor[...]` precision
//! slot already produce.
//!
//! Before this, the scalar `TypeExpr::Named` desugar arm made any lowercase
//! non-primitive name a `(t-var {} <name>)`, so `def f(x: u8) -> u8 = x` typed
//! as `forall u8. u8 -> u8`, checked clean, and scored 1.0: a program written
//! against a reserved and rejected dtype was accepted with an empty error
//! vector. In the positions that did reject (`deftype` field, `typealias`
//! body, value annotation) the message was "undeclared type variable `u8`",
//! which names the wrong defect and offers no remedy.
//!
//! The repair also covers two neighbours of that arm. An explicit `[..]`
//! quantifier list does not rebind a reserved spelling, in a scalar position
//! or in a tensor precision slot, because §5.8.1 states the rule on the
//! category. And the arm consults the tensor precision slot's predicate pair,
//! which rejects the other §1.1.1 reserved-but-deferred names as well as the
//! unsigned ones, so `complex64` in a scalar position reaches its own §1.1.1
//! diagnostic instead of scoring 1.0.
//!
//! Test labels are recorded in each function's doc comment. The suite claims
//! the eight §1.1.2 spellings in the six scalar positions it enumerates below,
//! those eight again under an explicit binder list in a scalar and a tensor
//! precision position, and the deferred family at `complex64` and `int4`
//! only; it locks the two `t-prim` positions that already worked. No other
//! type position and no other reserved spelling is claimed.

use chelis_types::check_ir_program;

/// The eight §1.1.2 spellings, exactly as `is_unsigned_dtype_name` lists them.
const UNSIGNED: [&str; 8] = [
    "u8", "u16", "u32", "u64", "uint8", "uint16", "uint32", "uint64",
];

/// Two of the other §1.1.1 reserved-but-deferred spellings, which ride the
/// same repair because the scalar arm consults the tensor precision slot's
/// predicate pair. Claimed at these two names only.
const DEFERRED: [&str; 2] = ["complex64", "int4"];

/// The deferred family's own §1.1.1 diagnostic, whose wording differs from the
/// unsigned one.
fn assert_deferred_diagnostic(source: &str, name: &str, position: &str) {
    let messages = messages(source);
    assert!(
        messages
            .iter()
            .any(|message| message.contains(&format!("`{name}`"))
                && message.contains("is reserved but deferred")),
        "the diagnostic for `{name}` in {position} must name the spelling and \
         cite §1.1.1; got: {messages:?}"
    );
}

fn parse_core_fixture(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parser::parse_str(source).expect("Deep fixture")
}

fn messages(source: &str) -> Vec<String> {
    let deep = parse_core_fixture(source);
    match check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ) {
        Ok(_) => Vec::new(),
        Err(report) => report
            .errors
            .iter()
            .map(|error| error.message.clone())
            .collect(),
    }
}

/// The three things the diagnostic owes a reader per §1.1.1 / §1.1.2: the
/// offending spelling, the governing sections, and a remedy.
fn assert_unsigned_diagnostic(source: &str, name: &str, position: &str) {
    let messages = messages(source);
    assert!(
        !messages.is_empty(),
        "`{name}` in {position} must be rejected, not accepted with an empty \
         error vector"
    );
    assert!(
        messages
            .iter()
            .any(|message| message.contains(&format!("`{name}`"))
                && message.contains("unsigned integer types are deferred")),
        "the diagnostic for `{name}` in {position} must name the spelling and \
         say why it is rejected; got: {messages:?}"
    );
}

/// REGRESSION test. The named instance of chelis#1593: `def f(x: u8) -> u8 = x`
/// scored 1.0 with no error at all.
#[test]
fn a_scalar_parameter_and_return_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!(
                r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} f)
  (defsig {{}} f (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))
  (def {{}}
    f
    (fn {{}}
      (params {{}} (x {{type: (t-var {{}} _)}}))
      (var {{span: "surf:45..46"}} x))))
"###
            ),
            name,
            "a scalar parameter and return",
        );
    }
}

/// REGRESSION test. A standalone `sig` also scored 1.0.
#[test]
fn a_sig_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!(
                r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} f)
  (defsig {{}} f (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))
  (def {{}} f (fn {{}} (params {{}} x) (var {{span: "surf:51..52"}} x))))
"###
            ),
            name,
            "a sig",
        );
    }
}

/// REGRESSION test. A record field rejected before, with the wrong message
/// ("undeclared type variable"). The claim here is the message, not the
/// rejection.
#[test]
fn a_record_field_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!(
                r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} Box)
  (deftype {{}} Box () (variant {{}} Box (field {{}} v (t-prim {{}} {name})))))
"###
            ),
            name,
            "a record field",
        );
    }
}

/// REGRESSION test. Same shape as the record field, positional variant form.
#[test]
fn a_positional_adt_field_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!(
                r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} Wrap)
  (deftype {{}} Wrap () (variant {{}} W (t-prim {{}} {name}))))
"###
            ),
            name,
            "a positional ADT field",
        );
    }
}

/// REGRESSION test. A `typealias` body.
#[test]
fn a_typealias_body_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!(
                r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} S)
  (typealias {{}} S () (t-prim {{}} {name})))
"###
            ),
            name,
            "a typealias body",
        );
    }
}

/// REGRESSION test. A lambda parameter annotation: a value-position annotation
/// with no enclosing quantifier scope, which §5.8.1's last sentence sends to
/// "the existing rule".
#[test]
fn a_lambda_annotation_is_rejected_with_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!(
                r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} f)
  (defsig {{}} f (t-fn {{}} (t-prim {{}} i32)))
  (def {{}}
    f
    (fn {{}}
      (params {{}})
      (app {{span: "surf:40..66"}}
        (fn {{span: "surf:40..59"}}
          (params {{}} (x {{type: (t-prim {{}} {name})}}))
          (lit {{span: "surf:55..59", surf_literal_style: "explicit", type: (t-prim {{}} i32)}} 1))
        (lit {{span: "surf:61..65", surf_literal_style: "explicit", type: (t-prim {{}} i32)}} 1)))))
"###
            ),
            name,
            "a lambda parameter annotation",
        );
    }
}

/// DISPOSITION LOCK. Green in both states. The cast target is the position that
/// already produced this diagnostic; it must keep producing exactly it.
#[test]
fn a_cast_target_still_carries_the_spec_diagnostic() {
    for name in UNSIGNED {
        assert_unsigned_diagnostic(
            &format!(
                r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} f)
  (defsig {{}} f (t-fn {{}} (t-prim {{}} i32)))
  (def {{}}
    f
    (fn {{}}
      (params {{}})
      (cast {{span: "surf:39..54"}}
        (lit {{span: "surf:44..48", surf_literal_style: "explicit", type: (t-prim {{}} i32)}} 1)
        (t-prim {{}} {name})))))
"###
            ),
            name,
            "a cast target",
        );
    }
}

/// DISPOSITION LOCK. Green in both states. The `tensor[...]` precision slot is
/// the positive precedent this change copies. Its dimension binder is explicit
/// under chelis#1854.
#[test]
fn a_tensor_precision_slot_still_carries_the_spec_diagnostic() {
    for name in UNSIGNED {
        let messages = messages(&format!(
            r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} f)
  (defsig {{}}
    f
    (d)
    (t-fn {{}}
      (t-tensor {{}} (d-var {{}} d) (t-prim {{}} {name}))
      (t-tensor {{}} (d-var {{}} d) (t-prim {{}} {name}))))
  (def {{}} f (fn {{}} (params {{}} x) (var {{span: "surf:76..77"}} x))))
"###
        ));
        assert!(
            messages.iter().any(
                |message| message.contains("unsigned integer types are deferred")
                    && message.contains("tensor element")
            ),
            "the tensor precision slot must keep its own tensor-flavoured \
             §1.1.1 diagnostic for `{name}`; got: {messages:?}"
        );
    }
}

/// DISPOSITION LOCK. An ordinary explicitly listed lowercase binder still
/// checks clean.
#[test]
fn a_genuine_explicit_lowercase_binder_still_checks_clean() {
    for source in [
        include_str!(
            "fixtures/core/tests/issue_1593_unsigned_scalar_position/a_genuine_explicit_lowercase_binder_still_checks_clean_242.deep"
        ),
        include_str!(
            "fixtures/core/tests/issue_1593_unsigned_scalar_position/a_genuine_explicit_lowercase_binder_still_checks_clean_243.deep"
        ),
    ] {
        let messages = messages(source);
        assert!(
            messages.is_empty(),
            "a genuine lowercase type variable must still check clean: \
             {source}\n{messages:?}"
        );
    }
}

/// DISPOSITION LOCK. Green in both states. Negative parity for the predicate:
/// the SIGNED spellings, including chelis#1587's short aliases, must not match
/// the unsigned family. `i8`..`i64` share the suffix digits and `i8`
/// contains no `uint8` only because the check is exact.
#[test]
fn signed_spellings_do_not_match_the_unsigned_family() {
    for name in [
        "i8", "i16", "i32", "i64", "i8", "i16", "i32", "i64", "f32", "f64", "bool",
    ] {
        let messages = messages(&format!(
            r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} f)
  (defsig {{}} f (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))
  (def {{}}
    f
    (fn {{}}
      (params {{}} (x {{type: (t-var {{}} _)}}))
      (var {{span: "surf:45..46"}} x))))
"###
        ));
        assert!(
            messages.is_empty(),
            "`{name}` is an active primitive (or an accepted input spelling \
             for one) and must still be accepted: {messages:?}"
        );
    }
}

/// DISPOSITION LOCK. Green in both states, and the positive control for the
/// row above: an explicit binder with an ordinary lowercase name still binds
/// and the program still checks clean, in both positions.
#[test]
fn an_explicit_binder_with_an_ordinary_name_still_checks_clean() {
    for source in [
        include_str!(
            "fixtures/core/tests/issue_1593_unsigned_scalar_position/an_explicit_binder_with_an_ordinary_name_still_checks_clean_317.deep"
        ),
        include_str!(
            "fixtures/core/tests/issue_1593_unsigned_scalar_position/an_explicit_binder_with_an_ordinary_name_still_checks_clean_318.deep"
        ),
    ] {
        let messages = messages(source);
        assert!(
            messages.is_empty(),
            "an ordinary explicit binder must still bind: {source}\n{messages:?}"
        );
    }
}

/// REGRESSION test. The reserved-but-deferred family in a scalar position.
/// Before, `def f(x: complex64) -> complex64 = x` scored 1.0 with an empty
/// error vector, exactly as the unsigned family did.
#[test]
fn the_deferred_family_is_rejected_in_a_scalar_position() {
    for name in DEFERRED {
        assert_deferred_diagnostic(
            &format!(
                r###"(module {{surf_path: "P.M"}}
  p.m
  (export {{}} f)
  (defsig {{}} f (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))
  (def {{}}
    f
    (fn {{}}
      (params {{}} (x {{type: (t-var {{}} _)}}))
      (var {{span: "surf:45..46"}} x))))
"###
            ),
            name,
            "a scalar parameter and return",
        );
    }
}

/// REGRESSION test. Round 1 P1: the checker's own Deep ingress. A `t-var`
/// naming a rejected dtype spelling is not a type variable, so hand-written
/// Deep that the desugarer would never emit is rejected too. Before, this
/// scored 1.0 with an empty error vector on both carriers of one program.
#[test]
fn a_deep_type_variable_naming_a_rejected_spelling_is_rejected() {
    for name in UNSIGNED.iter().chain(DEFERRED.iter()) {
        for (label, source) in [
            (
                "scalar",
                format!(
                    "(module {{surf_path: \"P.M\"}}\n  p.m\n  (export {{}} f)\n  \
                     (defsig {{}} f ({name}) (t-fn {{}} (t-var {{}} {name}) (t-var {{}} {name})))\n  \
                     (def {{}} f (fn {{}} (params {{}} (x {{type: (t-var {{}} {name})}})) \
                     (var {{}} x))))\n"
                ),
            ),
            (
                "tensor precision",
                format!(
                    "(module {{surf_path: \"P.M\"}}\n  p.m\n  (export {{}} f)\n  \
                     (defsig {{}} f ({name}) (t-fn {{}} (t-tensor {{}} (d-lit {{}} 3) (t-var {{}} {name})) \
                     (t-tensor {{}} (d-lit {{}} 3) (t-var {{}} {name}))))\n  \
                     (def {{}} f (fn {{}} (params {{}} x) (var {{}} x))))\n"
                ),
            ),
        ] {
            let deep = chelis_deep::parser::parse_and_stamp_file(&source).expect("deep parse");
            let messages: Vec<String> = match check_ir_program(
                &deep,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ) {
                Ok(_) => Vec::new(),
                Err(report) => report.errors.iter().map(|e| e.message.clone()).collect(),
            };
            assert!(
                messages
                    .iter()
                    .any(|message| message.contains(&format!("`{name}`"))
                        && message.contains("cannot be a `defsig` binder")),
                "a Deep type variable named `{name}` in a {label} position must \
                 be rejected by the explicit binder-list owner; got: {messages:?}"
            );
        }
    }
}

/// DISPOSITION LOCK. Green in both states. An ordinary Deep type variable is
/// still a type variable; the repair narrowed the rule to rejected spellings.
#[test]
fn an_ordinary_deep_type_variable_still_checks_clean() {
    for name in ["a", "p", "elem"] {
        let source = format!(
            "(module {{surf_path: \"P.M\"}}\n  p.m\n  (export {{}} f)\n  \
             (defsig {{}} f ({name}) (t-fn {{}} (t-var {{}} {name}) (t-var {{}} {name})))\n  \
             (def {{}} f (fn {{}} (params {{}} (x {{type: (t-var {{}} {name})}})) \
             (var {{}} x))))\n"
        );
        let deep = chelis_deep::parser::parse_and_stamp_file(&source).expect("deep parse");
        assert!(
            check_ir_program(
                &deep,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None)
            )
            .is_ok(),
            "`{name}` must still be a legal Deep type variable"
        );
    }
}
