//! chelis#1606 / chelis#1527: `f8e4m3` rejected by the type checker in
//! EVERY named type position, not only the tensor element slot and the
//! `cast` target that `spec/04-type-system.md` §1.1.1 already covered.
//!
//! Before this repair, `Prim::parse_name("f8e4m3")` succeeded (it is a real
//! `Prim` variant, per §1.1.1's design note), so the resolver's `t-prim` arm
//! returned `Type::Prim(F8e4m3)` before ever consulting whether the parsed
//! primitive is admissible. Only the tensor precision slot and the cast
//! target ran their own explicit `Prim::F8e4m3` check downstream of the
//! resolver, so every other position scored a clean 1.0. See
//! `f8e4m3_rejection.rs` for the pre-existing tensor/cast lock this suite
//! must not disturb.

use chelis_types::{check_ir_program, check_typed_program};
use serde_json::Value;

fn core_fixture(source: &str) -> Vec<chelis_deep::Expr> {
    serde_json::from_str(source).expect("lossless core fixture")
}

// Mutate the primitive name in the original AST, preserving its legacy carrier,
// metadata copies, and source spans. Both reserved spellings have equal length.
fn with_precision(source: &str, name: &str) -> Vec<chelis_deep::Expr> {
    fn replace(value: &mut Value, name: &str) {
        match value {
            Value::String(value) if value == "f8e4m3" => *value = name.to_string(),
            Value::Array(values) => values.iter_mut().for_each(|value| replace(value, name)),
            Value::Object(values) => values.values_mut().for_each(|value| replace(value, name)),
            _ => {}
        }
    }
    let mut encoded: Value = serde_json::from_str(source).expect("lossless core fixture");
    replace(&mut encoded, name);
    serde_json::from_value(encoded).expect("primitive mutation preserves the AST")
}

fn hand_authored_deep_property(
    signature_type: &str,
    parameter_type: &str,
) -> Vec<chelis_deep::Expr> {
    hand_authored_deep_property_types(
        &format!("(t-prim {{}} {signature_type})"),
        &format!("(t-prim {{}} {parameter_type})"),
    )
}

fn hand_authored_deep_property_types(
    signature_type: &str,
    parameter_type: &str,
) -> Vec<chelis_deep::Expr> {
    hand_authored_deep_property_slots(
        &[signature_type.to_string()],
        &[parameter_type.to_string()],
        "(lit {} true)",
    )
}

fn hand_authored_deep_property_slots(
    signature_types: &[String],
    parameter_types: &[String],
    body: &str,
) -> Vec<chelis_deep::Expr> {
    let source = hand_authored_deep_property_slots_source(signature_types, parameter_types, body);
    chelis_deep::parse_and_stamp_file(&source)
        .unwrap_or_else(|error| panic!("valid hand-authored Deep property:\n{source}\n{error:?}"))
}

fn hand_authored_deep_property_slots_source(
    signature_types: &[String],
    parameter_types: &[String],
    body: &str,
) -> String {
    let parameter_names = ["x", "y", "z", "w", "v", "u"];
    assert!(
        signature_types.len() <= parameter_names.len()
            && parameter_types.len() <= parameter_names.len()
    );
    let signature = signature_types.join(" ");
    let quantifiers = parameter_types
        .iter()
        .enumerate()
        .map(|(index, ty)| format!("({} {{type: {ty}}})", parameter_names[index]))
        .collect::<Vec<_>>()
        .join(" ");
    format!(
        "(defsig {{}} classify \
           (t-fn {{}} {signature} (t-prim {{}} bool)))\n\
         (def {{chelis_role: \"property\", property_source_kind: \"user\", \
                property_quantifiers: \
                  (params {{}} {quantifiers}), \
                property_preconditions: (tuple {{}})}} \
           classify \
           (fn {{}} (params {{}} {quantifiers}) {body}))"
    )
}

fn hand_authored_nominal_property(application: &str) -> Vec<chelis_deep::Expr> {
    let source = format!(
        "(deftype {{}} Pair (a b) \
           (variant {{}} Pair \
             (field {{}} left (t-var {{}} a)) \
             (field {{}} right (t-var {{}} b))))\n\
         (defsig {{}} inspect (t-fn {{}} {application} (t-prim {{}} bool)))\n\
         (def {{}} inspect (fn {{}} (params {{}} x) (lit {{}} true)))"
    );
    chelis_deep::parse_and_stamp_file(&source)
        .unwrap_or_else(|error| panic!("valid nominal recovery fixture:\n{source}\n{error:?}"))
}

fn assert_api_diagnostic_counts(
    program: &[chelis_deep::Expr],
    reserved: usize,
    mismatches: usize,
    label: &str,
) {
    use chelis_types::errors::CheckErrorKind;

    for (entry, result) in [
        (
            "ir",
            check_ir_program(
                program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ),
        (
            "typed",
            check_typed_program(
                program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ),
    ] {
        let report = result.expect_err("the adversarial property must reject");
        assert_eq!(
            report
                .errors
                .iter()
                .filter(|error| matches!(error.kind, CheckErrorKind::UnsupportedTensorPrecision))
                .count(),
            reserved,
            "{entry}/{label}: {:?}",
            report.errors
        );
        assert_eq!(
            report
                .errors
                .iter()
                .filter(|error| matches!(error.kind, CheckErrorKind::TypeMismatch))
                .count(),
            mismatches,
            "{entry}/{label}: {:?}",
            report.errors
        );
    }
}

fn legacy_list_root(expr: &chelis_deep::Expr) -> chelis_deep::Expr {
    let chelis_deep::Expr::Node(node, span) = expr else {
        panic!("fixture must be a stamped node: {expr:?}");
    };
    let mut elements = vec![
        chelis_deep::Expr::Atom(
            chelis_deep::Atom::Tag(node.tag()),
            chelis_deep::Span::new(span.offset + 1, 1),
        ),
        chelis_deep::Expr::Map(
            node.meta().clone(),
            chelis_deep::Span::new(span.offset + 2, 2),
        ),
    ];
    elements.extend(node.children_slice().iter().cloned());
    chelis_deep::Expr::List(chelis_deep::List { elements }, *span)
}

fn defsig_parameter_as_legacy_list(mut program: Vec<chelis_deep::Expr>) -> Vec<chelis_deep::Expr> {
    let chelis_deep::Expr::Node(defsig, defsig_span) = program.remove(0) else {
        panic!("fixture starts with defsig");
    };
    let (defsig_tag, defsig_meta, mut defsig_children) = defsig.into_parts();
    let chelis_deep::Expr::Node(signature, signature_span) = defsig_children.remove(1) else {
        panic!("defsig type is a stamped function type");
    };
    let (signature_tag, signature_meta, mut signature_children) = signature.into_parts();
    signature_children[0] = legacy_list_root(&signature_children[0]);
    defsig_children.push(chelis_deep::Expr::node(
        signature_tag,
        signature_meta,
        signature_children,
        signature_span,
    ));
    program.insert(
        0,
        chelis_deep::Expr::node(defsig_tag, defsig_meta, defsig_children, defsig_span),
    );
    program
}

fn fn_params_as_bare_list(mut program: Vec<chelis_deep::Expr>) -> Vec<chelis_deep::Expr> {
    let def_index = program
        .iter()
        .position(|expr| expr.tag() == Some(chelis_deep::DeepTag::Def))
        .expect("fixture contains a property def");
    let chelis_deep::Expr::Node(def, def_span) = program.remove(def_index) else {
        panic!("property def is a stamped node");
    };
    let (def_tag, def_meta, mut def_children) = def.into_parts();
    let chelis_deep::Expr::Node(function, function_span) = def_children.remove(1) else {
        panic!("property body is a stamped fn");
    };
    let (function_tag, function_meta, mut function_children) = function.into_parts();
    let chelis_deep::Expr::Node(params, params_span) = function_children.remove(0) else {
        panic!("function params are a stamped node");
    };
    function_children.insert(
        0,
        chelis_deep::Expr::BareList(params.children_slice().to_vec(), params_span),
    );
    let mut function_elements = vec![
        chelis_deep::Expr::Atom(chelis_deep::Atom::Tag(function_tag), function_span),
        chelis_deep::Expr::Map(function_meta, function_span),
    ];
    function_elements.extend(function_children);
    def_children.push(chelis_deep::Expr::List(
        chelis_deep::List {
            elements: function_elements,
        },
        function_span,
    ));
    let mut def_elements = vec![
        chelis_deep::Expr::Atom(chelis_deep::Atom::Tag(def_tag), def_span),
        chelis_deep::Expr::Map(def_meta, def_span),
    ];
    def_elements.extend(def_children);
    program.insert(
        def_index,
        chelis_deep::Expr::List(
            chelis_deep::List {
                elements: def_elements,
            },
            def_span,
        ),
    );
    program
}

fn assert_rejected(fixture: &str, position: &str) {
    for name in ["f8e4m3", "f8e5m2"] {
        let legacy = with_precision(fixture, name);
        let text = chelis_deep::printer::print_canonical_flat(&legacy);
        let stamped = chelis_deep::parse_and_stamp_file(&text).expect("stamp canonical Deep");
        for (carrier, exprs) in [("legacy", legacy), ("stamped", stamped)] {
            for (entry, result) in [
                (
                    "ir",
                    check_ir_program(
                        &exprs,
                        chelis_types::ProgramOrigin::Unlinked,
                        &mut chelis_types::NativeServices::new(None),
                    ),
                ),
                (
                    "typed",
                    check_typed_program(
                        &exprs,
                        chelis_types::ProgramOrigin::Unlinked,
                        &mut chelis_types::NativeServices::new(None),
                    ),
                ),
            ] {
                let report = result.expect_err("reserved primitive must reject");
                assert!(
                    report.errors.iter().any(|error| {
                        matches!(
                            error.kind,
                            chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
                        ) && error.message.contains(name)
                    }),
                    "{carrier}/{entry}/{position}: {:?}",
                    report.errors
                );
            }
        }
    }
}

/// The expected sequence names one declaration-owned diagnostic per spelling.
fn assert_one_report_per_site(program: Vec<chelis_deep::Expr>, expected: &[(&str, usize)]) {
    for (entry, result) in [
        (
            "ir",
            check_ir_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ),
        (
            "typed",
            check_typed_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ),
    ] {
        let report = result.expect_err("reserved dtype sites must reject");
        assert_eq!(
            report.errors.len(),
            expected.len(),
            "{entry}: {:?}",
            report.errors
        );
        for (error, (name, offset)) in report.errors.iter().zip(expected) {
            assert!(
                matches!(
                    error.kind,
                    chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
                ),
                "{entry}: {error:?}"
            );
            assert!(error.message.contains(name), "{entry}: {error:?}");
            assert_eq!(error.span_offset, Some(*offset), "{entry}: {error:?}");
            let span = format!("source:{offset}..{}", offset + name.len());
            assert_eq!(
                error.span_id.as_deref(),
                Some(span.as_str()),
                "{entry}: {error:?}"
            );
        }
    }
}

fn legacy_tensor_precision(name: &str) -> Vec<chelis_deep::Expr> {
    let mut program = with_precision(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/legacy_tensor.json"),
        name,
    );
    let chelis_deep::Expr::List(alias, _) = &mut program[0] else {
        panic!("legacy type alias fixture");
    };
    let chelis_deep::Expr::List(tensor, _) = &mut alias.elements[4] else {
        panic!("legacy tensor fixture");
    };
    let chelis_deep::Expr::List(precision, span) = &mut tensor.elements[3] else {
        panic!("legacy precision fixture");
    };
    // A legacy precision string had no source location. Recreate that checker
    // boundary directly, without erasing the enclosing tensor's known span.
    *span = chelis_deep::Span::new(0, 0);
    for element in &mut precision.elements {
        match element {
            chelis_deep::Expr::Atom(_, span) | chelis_deep::Expr::Map(_, span) => {
                *span = chelis_deep::Span::new(0, 0);
            }
            _ => panic!("primitive fixture only contains atoms and metadata"),
        }
    }
    program
}

#[test]
fn a_reserved_parameter_site_reports_once_at_both_entries() {
    for name in ["f8e4m3", "f8e5m2"] {
        assert_one_report_per_site(
            with_precision(
                include_str!(
                    "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/parameter.json"
                ),
                name,
            ),
            &[(name, 16)],
        );
    }
}

#[test]
fn no_clause_inline_tensor_precision_is_rejected_as_undeclared() {
    let source = include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/undeclared_precision.json"
    );
    let program = core_fixture(source);
    for result in [
        check_ir_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
        check_typed_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
    ] {
        let report = result.expect_err("an unlisted precision name is not a binder");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.message.contains("primitive") && error.message.contains("`p`")),
            "{:?}",
            report.errors
        );
    }

    let explicit = core_fixture(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/explicit_precision.json"
    ));
    for result in [
        check_ir_program(
            &explicit,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
        check_typed_program(
            &explicit,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
    ] {
        result.expect("an explicit matching clause remains valid");
    }

    let excluded = core_fixture(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/excluded_precision.json"
    ));
    for result in [
        check_ir_program(
            &excluded,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
        check_typed_program(
            &excluded,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
    ] {
        let report = result.expect_err("an explicit clause is authoritative over `p`");
        assert!(
            report
                .errors
                .iter()
                .any(|error| error.message.contains("`p`")),
            "{:?}",
            report.errors
        );
    }
}

#[test]
fn property_quantifier_reserved_types_have_one_semantic_diagnostic_owner() {
    for name in ["f8e4m3", "f8e5m2"] {
        for (fixture, offset) in [
            (
                include_str!(
                    "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/property_scalar.json"
                ),
                29,
            ),
            (
                include_str!(
                    "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/property_tensor.json"
                ),
                39,
            ),
        ] {
            assert_one_report_per_site(with_precision(fixture, name), &[(name, offset)]);
        }
    }
}

#[test]
fn hand_authored_property_parameter_contract_remains_independent_from_defsig() {
    let program = hand_authored_deep_property("f64", "f32");
    for (entry, result) in [
        (
            "ir",
            check_ir_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ),
        (
            "typed",
            check_typed_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ),
    ] {
        let report = result.expect_err("the authored f32 parameter must conflict with defsig f64");
        assert_eq!(report.errors.len(), 1, "{entry}: {:?}", report.errors);
        assert!(
            matches!(
                report.errors[0].kind,
                chelis_types::errors::CheckErrorKind::TypeMismatch
            ),
            "{entry}: {:?}",
            report.errors
        );
        assert!(
            report.errors[0].message.contains("f32") && report.errors[0].message.contains("f64"),
            "{entry}: {:?}",
            report.errors
        );
    }
}

#[test]
fn matching_hand_authored_property_parameter_contract_remains_valid() {
    let program = hand_authored_deep_property("f32", "f32");
    for (entry, result) in [
        (
            "ir",
            check_ir_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ),
        (
            "typed",
            check_typed_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ),
    ] {
        result.unwrap_or_else(|report| {
            panic!("{entry}: matching authored property contracts must pass: {report:?}")
        });
    }
}

#[test]
fn property_copy_ownership_uses_spanless_semantic_type_syntax() {
    use chelis_types::errors::CheckErrorKind;

    for name in ["f8e4m3", "f8e5m2"] {
        for ty in [
            format!("(t-prim {{doc: \"same\"}} {name})"),
            format!(
                "(t-tensor {{doc: \"same\"}} \
                   (d-lit {{doc: \"dimension\"}} 3) \
                   (t-prim {{doc: \"precision\"}} {name}))"
            ),
            format!(
                "(t-prim {{doc: \"same\", span: \"parameter\", \
                   loc: (loc \"parameter.dp\" 9 8), source: (parameter copy)}} {name})"
            ),
            format!(
                "(t-prim {{doc: \"same\", \
                   tool_data: {{nested: (payload \"same\")}}}} {name})"
            ),
            format!(
                "(t-prim {{doc: \"same\", span_start: 7, span_end: 13, \
                   span_file: \"parameter.dp\"}} {name})"
            ),
        ] {
            let signature = ty
                .replace("span: \"parameter\"", "span: \"signature\"")
                .replace("parameter.dp", "signature.dp")
                .replace("(parameter copy)", "(signature copy)")
                .replace("span_start: 7", "span_start: 70")
                .replace("span_end: 13", "span_end: 130");
            let program = hand_authored_deep_property_types(&signature, &ty);
            for (entry, result) in [
                (
                    "ir",
                    check_ir_program(
                        &program,
                        chelis_types::ProgramOrigin::Unlinked,
                        &mut chelis_types::NativeServices::new(None),
                    ),
                ),
                (
                    "typed",
                    check_typed_program(
                        &program,
                        chelis_types::ProgramOrigin::Unlinked,
                        &mut chelis_types::NativeServices::new(None),
                    ),
                ),
            ] {
                let report = result.expect_err("the reserved dtype must reject");
                assert_eq!(report.errors.len(), 1, "{entry}/{name}/{ty}: {report:?}");
                assert!(
                    matches!(
                        report.errors[0].kind,
                        CheckErrorKind::UnsupportedTensorPrecision
                    ),
                    "{entry}/{name}/{ty}: {report:?}"
                );
                assert!(
                    report.errors[0].message.contains(name),
                    "{entry}/{name}/{ty}: {report:?}"
                );
            }
        }
    }
}

#[test]
fn property_copy_ownership_ignores_nonsemantic_metadata_differences() {
    use chelis_types::errors::CheckErrorKind;

    for name in ["f8e4m3", "f8e5m2"] {
        for (signature, parameter) in [
            (
                format!("(t-prim {{doc: \"signature\"}} {name})"),
                format!("(t-prim {{doc: \"parameter\"}} {name})"),
            ),
            (
                format!(
                    "(t-tensor {{doc: \"signature\"}} \
                       (d-lit {{}} 3) (t-prim {{}} {name}))"
                ),
                format!(
                    "(t-tensor {{doc: \"parameter\"}} \
                       (d-lit {{}} 3) (t-prim {{}} {name}))"
                ),
            ),
        ] {
            let program = hand_authored_deep_property_types(&signature, &parameter);
            for (entry, result) in [
                (
                    "ir",
                    check_ir_program(
                        &program,
                        chelis_types::ProgramOrigin::Unlinked,
                        &mut chelis_types::NativeServices::new(None),
                    ),
                ),
                (
                    "typed",
                    check_typed_program(
                        &program,
                        chelis_types::ProgramOrigin::Unlinked,
                        &mut chelis_types::NativeServices::new(None),
                    ),
                ),
            ] {
                let report = result.expect_err("the declaration-owned reserved spelling rejects");
                assert_eq!(
                    report
                        .errors
                        .iter()
                        .filter(|error| {
                            matches!(error.kind, CheckErrorKind::UnsupportedTensorPrecision)
                        })
                        .count(),
                    1,
                    "{entry}/{name}: {report:?}"
                );
            }
        }
    }
}

#[test]
fn property_copy_ownership_is_classified_per_parameter_slot() {
    for name in ["f8e4m3", "f8e5m2"] {
        for reserved in [
            format!("(t-prim {{}} {name})"),
            format!("(t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} {name}))"),
        ] {
            for (signature, parameter) in [
                (
                    vec![reserved.clone(), "(t-prim {} f64)".to_string()],
                    vec![reserved.clone(), "(t-prim {} f32)".to_string()],
                ),
                (
                    vec!["(t-prim {} f64)".to_string(), reserved.clone()],
                    vec!["(t-prim {} f32)".to_string(), reserved.clone()],
                ),
            ] {
                let program =
                    hand_authored_deep_property_slots(&signature, &parameter, "(lit {} true)");
                assert_api_diagnostic_counts(
                    &program,
                    1,
                    1,
                    &format!("{name}/{signature:?}/{parameter:?}"),
                );
            }
        }
    }
}

#[test]
fn property_copy_ownership_keeps_mixed_slot_dispositions_independent() {
    for name in ["f8e4m3", "f8e5m2"] {
        let scalar = format!("(t-prim {{}} {name})");
        let tensor = format!("(t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} {name}))");
        let signature = vec![
            scalar.clone(),
            "(t-prim {} f64)".to_string(),
            tensor.clone(),
            "(t-prim {} i64)".to_string(),
        ];
        let parameter = vec![
            scalar,
            "(t-prim {} f32)".to_string(),
            tensor,
            "(t-prim {} bool)".to_string(),
        ];
        let program = hand_authored_deep_property_slots(&signature, &parameter, "(lit {} true)");
        assert_api_diagnostic_counts(&program, 1, 1, name);
    }
}

#[test]
fn property_copy_ownership_is_per_slot_when_parameter_arity_disagrees() {
    for name in ["f8e4m3", "f8e5m2"] {
        let reserved = format!("(t-prim {{}} {name})");
        for (signature, parameter, label) in [
            (
                vec![reserved.clone(), "(t-prim {} f32)".to_string()],
                vec![reserved.clone()],
                "missing parameter",
            ),
            (
                vec![reserved.clone()],
                vec![reserved.clone(), "(t-prim {} f32)".to_string()],
                "extra parameter",
            ),
        ] {
            let program =
                hand_authored_deep_property_slots(&signature, &parameter, "(lit {} true)");
            assert_api_diagnostic_counts(&program, 1, 1, &format!("{name}/{label}"));
        }
    }
}

#[test]
fn property_copy_ownership_fails_closed_per_invalid_or_noncanonical_slot() {
    use chelis_types::errors::CheckErrorKind;

    for name in ["f8e4m3", "f8e5m2"] {
        let reserved = format!("(t-prim {{}} {name})");
        let invalid = hand_authored_deep_property_slots(
            &[reserved.clone(), "(t-prim {} f32)".to_string()],
            &[reserved.clone(), "(t-prim {} madeup)".to_string()],
            "(lit {} true)",
        );
        for (entry, result) in [
            (
                "ir",
                check_ir_program(
                    &invalid,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ),
            (
                "typed",
                check_typed_program(
                    &invalid,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ),
        ] {
            let report = result.expect_err("the invalid neighboring slot must reject");
            assert_eq!(
                report
                    .errors
                    .iter()
                    .filter(|error| matches!(
                        error.kind,
                        CheckErrorKind::UnsupportedTensorPrecision
                    ))
                    .count(),
                1,
                "{entry}/{name}: {:?}",
                report.errors
            );
            assert!(
                report
                    .errors
                    .iter()
                    .any(|error| error.message.contains("madeup")),
                "{entry}/{name}: {:?}",
                report.errors
            );
        }

        let noncanonical = fn_params_as_bare_list(hand_authored_deep_property_slots(
            std::slice::from_ref(&reserved),
            std::slice::from_ref(&reserved),
            "(lit {} true)",
        ));
        for (entry, result) in [
            (
                "ir",
                check_ir_program(
                    &noncanonical,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ),
            (
                "typed",
                check_typed_program(
                    &noncanonical,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ),
        ] {
            let report = result.expect_err("a noncanonical params carrier must fail closed");
            assert_eq!(
                report.errors.len(),
                1,
                "{entry}/{name}: {:?}",
                report.errors
            );
            assert!(
                matches!(report.errors[0].kind, CheckErrorKind::MalformedForm),
                "{entry}/{name}: {:?}",
                report.errors
            );
            assert!(
                report.errors[0].message.contains("property_quantifiers"),
                "{entry}/{name}: {:?}",
                report.errors
            );
        }

        let malformed_source = hand_authored_deep_property_slots_source(
            &[reserved.clone(), "(t-prim {} f32)".to_string()],
            &[reserved.clone(), "(unknown-type {})".to_string()],
            "(lit {} true)",
        );
        assert!(
            chelis_deep::parse_and_stamp_file(&malformed_source).is_err(),
            "a malformed type carrier must fail closed at Deep ingress"
        );
    }
}

#[test]
fn nominal_arity_recovery_visits_every_header_owned_type_argument() {
    for name in ["f8e4m3", "f8e5m2"] {
        let other = if name == "f8e4m3" { "f8e5m2" } else { "f8e4m3" };
        let cases = [
            (
                format!("(t-adt {{}} Pair (t-tuple {{}} (t-prim {{}} {name})))"),
                1,
                "too few with one nested rejection",
            ),
            (
                format!(
                    "(t-adt {{}} Pair \
                       (t-prim {{}} {name}) \
                       (t-tensor {{}} (d-lit {{}} 3) (t-prim {{}} {other})) \
                       (t-prim {{}} f32))"
                ),
                2,
                "too many with two nested rejections",
            ),
            (
                format!(
                    "(t-adt {{}} Pair \
                       (t-tuple {{}} (t-prim {{}} {name}) (t-prim {{}} {other})) \
                       (t-ref {{}} (t-prim {{}} {name})) \
                       (t-prim {{}} f32))"
                ),
                2,
                "nested composites preserve one rejection per spelling",
            ),
            (
                "(t-adt {} Pair (t-prim {} f32))".to_string(),
                0,
                "valid supplied child still leaves the arity witness",
            ),
            (
                "(t-adt {} Pair (t-prim {} f32) (t-prim {} f64) (t-prim {} bool))".to_string(),
                0,
                "valid extra-arity control",
            ),
        ];
        for (application, reserved, label) in cases {
            let program = hand_authored_nominal_property(&application);
            assert_api_diagnostic_counts(&program, reserved, 1, &format!("{name}/{label}"));
        }
    }
}

#[test]
fn property_copy_ownership_accepts_the_transitional_list_type_carrier() {
    use chelis_types::errors::CheckErrorKind;

    for name in ["f8e4m3", "f8e5m2"] {
        for ty in [
            format!("(t-prim {{doc: \"same\"}} {name})"),
            format!(
                "(t-tensor {{doc: \"same\"}} \
                   (d-lit {{}} 3) (t-prim {{}} {name}))"
            ),
        ] {
            let program =
                defsig_parameter_as_legacy_list(hand_authored_deep_property_types(&ty, &ty));
            for (entry, result) in [
                (
                    "ir",
                    check_ir_program(
                        &program,
                        chelis_types::ProgramOrigin::Unlinked,
                        &mut chelis_types::NativeServices::new(None),
                    ),
                ),
                (
                    "typed",
                    check_typed_program(
                        &program,
                        chelis_types::ProgramOrigin::Unlinked,
                        &mut chelis_types::NativeServices::new(None),
                    ),
                ),
            ] {
                let report = result.expect_err("the reserved dtype must reject");
                assert_eq!(report.errors.len(), 1, "{entry}/{name}/{ty}: {report:?}");
                assert!(
                    matches!(
                        report.errors[0].kind,
                        CheckErrorKind::UnsupportedTensorPrecision
                    ),
                    "{entry}/{name}/{ty}: {report:?}"
                );
            }
        }
    }
}

#[test]
fn property_quantifier_recovery_keeps_an_independent_body_error() {
    use chelis_types::errors::CheckErrorKind;

    let program = core_fixture(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/property_body_error.json"
    ));
    for result in [
        check_ir_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
        check_typed_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
    ] {
        let report = result.expect_err("the binder and non-bool body must both reject");
        assert_eq!(report.errors.len(), 2, "{:?}", report.errors);
        assert_eq!(
            report
                .errors
                .iter()
                .filter(|error| matches!(error.kind, CheckErrorKind::UnsupportedTensorPrecision))
                .count(),
            1,
            "{:?}",
            report.errors
        );
        assert!(
            report
                .errors
                .iter()
                .any(|error| matches!(error.kind, CheckErrorKind::TypeMismatch)),
            "{:?}",
            report.errors
        );
    }
}

#[test]
fn legacy_unknown_tensor_precision_spans_do_not_inherit_the_tensor_span() {
    for name in ["f8e4m3", "f8e5m2"] {
        let program = legacy_tensor_precision(name);
        for (entry, result) in [
            (
                "ir",
                check_ir_program(
                    &program,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ),
            (
                "typed",
                check_typed_program(
                    &program,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ),
        ] {
            let report = result.expect_err("legacy reserved precision must reject");
            assert_eq!(
                report.errors.len(),
                1,
                "{entry}/{name}: {:?}",
                report.errors
            );
            let error = &report.errors[0];
            assert!(
                matches!(
                    error.kind,
                    chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
                ),
                "{entry}/{name}: {error:?}"
            );
            assert_eq!(error.span_offset, None, "{entry}/{name}: {error:?}");
            assert_eq!(error.span_id, None, "{entry}/{name}: {error:?}");
        }
    }
}

#[test]
fn separate_reserved_sites_keep_separate_diagnostics() {
    assert_one_report_per_site(
        core_fixture(include_str!(
            "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/separate_sites.json"
        )),
        &[("f8e4m3", 12), ("f8e5m2", 47)],
    );
}

#[test]
fn repeated_reserved_spelling_in_distinct_declarations_is_not_deduplicated() {
    assert_one_report_per_site(
        core_fixture(include_str!(
            "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/repeated_sites.json"
        )),
        &[("f8e4m3", 12), ("f8e4m3", 47)],
    );
}

#[test]
fn calls_propagate_the_failed_signature_without_a_second_report() {
    assert_one_report_per_site(
        core_fixture(include_str!(
            "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/failed_call.json"
        )),
        &[("f8e4m3", 16)],
    );
}

#[test]
fn one_signature_keys_reserved_diagnostics_by_spelling() {
    assert_one_report_per_site(
        core_fixture(include_str!(
            "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/distinct_parameters.json"
        )),
        &[("f8e4m3", 16), ("f8e5m2", 27)],
    );
    assert_one_report_per_site(
        core_fixture(include_str!(
            "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/repeated_parameters.json"
        )),
        &[("f8e4m3", 16)],
    );
}

#[test]
fn parameter_and_return_sites_each_report_once() {
    assert_one_report_per_site(
        core_fixture(include_str!(
            "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/parameter_return.json"
        )),
        &[("f8e4m3", 16), ("f8e5m2", 27)],
    );
}

#[test]
fn nested_type_components_each_report_once() {
    for (fixture, first, second) in [
        (
            include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/nested_tuple.json"),
            17,
            25,
        ),
        (
            include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/nested_arrow.json"),
            17,
            28,
        ),
        (
            include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/nested_dict.json"),
            21,
            29,
        ),
    ] {
        assert_one_report_per_site(
            core_fixture(fixture),
            &[("f8e4m3", first), ("f8e5m2", second)],
        );
    }
}

#[test]
fn a_failed_signature_does_not_hide_an_independent_body_site() {
    let source =
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/body_cast.json");
    let program = core_fixture(source);
    for result in [
        check_ir_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
        check_typed_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
    ] {
        let report = result.expect_err("both reserved sites must reject");
        assert_eq!(report.errors.len(), 2, "{:?}", report.errors);
        for (error, name) in report.errors.iter().zip(["f8e4m3", "f8e5m2"]) {
            assert!(
                matches!(
                    error.kind,
                    chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
                ),
                "{error:?}"
            );
            assert!(error.message.contains(name), "{error:?}");
        }
        assert_eq!(report.errors[0].span_id.as_deref(), Some("source:16..22"));
        let start = 33;
        let expected = "surf:33..51";
        assert_eq!(report.errors[1].span_offset, Some(start));
        assert_eq!(report.errors[1].span_id.as_deref(), Some(expected));
    }
}

#[test]
fn declaration_ownership_does_not_absorb_a_same_spelling_cast_failure() {
    let source = include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/same_spelling_cast.json"
    );
    let program = core_fixture(source);
    for result in [
        check_ir_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
        check_typed_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
    ] {
        let report = result.expect_err("the declaration and cast sites must both reject");
        assert_eq!(report.errors.len(), 2, "{:?}", report.errors);
        assert!(
            report
                .errors
                .iter()
                .all(|error| error.message.contains("f8e4m3")),
            "{:?}",
            report.errors
        );
        assert_eq!(report.errors[0].span_id.as_deref(), Some("source:16..22"));
        let start = 33;
        let expected = "surf:33..51";
        assert_eq!(report.errors[1].span_offset, Some(start));
        assert_eq!(report.errors[1].span_id.as_deref(), Some(expected));
    }
}

#[test]
fn a_rejected_tensor_precision_preserves_dimensions_for_the_body() {
    use chelis_types::errors::CheckErrorKind;

    for name in ["f8e4m3", "f8e5m2"] {
        let program = with_precision(
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/dimension_error.json"
            ),
            name,
        );
        let offset = 25;
        let span = format!("source:{offset}..{}", offset + name.len());

        for (entry, result) in [
            (
                "ir",
                check_ir_program(
                    &program,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ),
            (
                "typed",
                check_typed_program(
                    &program,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ),
        ] {
            let report = result.expect_err("the reserved dtype and invalid axis must reject");
            assert_eq!(
                report.errors.len(),
                2,
                "{entry}/{name}: {:?}",
                report.errors
            );

            let reserved: Vec<_> = report
                .errors
                .iter()
                .filter(|error| matches!(error.kind, CheckErrorKind::UnsupportedTensorPrecision))
                .collect();
            assert_eq!(reserved.len(), 1, "{entry}/{name}: {:?}", report.errors);
            assert!(reserved[0].message.contains(name), "{entry}: {reserved:?}");
            assert_eq!(
                reserved[0].span_offset,
                Some(offset),
                "{entry}: {reserved:?}"
            );
            assert_eq!(
                reserved[0].span_id.as_deref(),
                Some(span.as_str()),
                "{entry}: {reserved:?}"
            );

            let dimensions: Vec<_> = report
                .errors
                .iter()
                .filter(|error| matches!(error.kind, CheckErrorKind::DimensionMismatch))
                .collect();
            assert_eq!(dimensions.len(), 1, "{entry}/{name}: {:?}", report.errors);
        }
    }
}

#[test]
fn a_tensor_element_reserved_site_has_one_located_owner() {
    for name in ["f8e4m3", "f8e5m2"] {
        for (fixture, offset) in [
            (
                include_str!(
                    "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/located_tensor.json"
                ),
                26,
            ),
            (
                include_str!(
                    "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/located_tensor_whitespace.json"
                ),
                28,
            ),
        ] {
            assert_one_report_per_site(with_precision(fixture, name), &[(name, offset)]);
        }
    }
}

#[test]
fn legacy_tensor_precision_preserves_its_diagnostic_span() {
    for name in ["f8e4m3", "f8e5m2"] {
        assert_one_report_per_site(
            with_precision(
                include_str!(
                    "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/located_tensor_unexpanded.json"
                ),
                name,
            ),
            &[(name, 26)],
        );
    }
}

#[test]
fn valid_parts_of_a_failed_signature_still_constrain_the_body() {
    use chelis_types::errors::CheckErrorKind;
    for (source, expected_kind) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/body_result_mismatch.json"
            ),
            CheckErrorKind::TypeMismatch,
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/body_parameter_mismatch.json"
            ),
            CheckErrorKind::PrecisionMismatch,
        ),
    ] {
        let program = core_fixture(source);
        for result in [
            check_ir_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            let report = result.expect_err("reserved type and independent mismatch must reject");
            assert_eq!(report.errors.len(), 2, "{source}: {:?}", report.errors);
            assert!(matches!(
                report.errors[0].kind,
                chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
            ));
            assert!(
                matches!(
                    (&report.errors[1].kind, &expected_kind),
                    (CheckErrorKind::TypeMismatch, CheckErrorKind::TypeMismatch)
                        | (
                            CheckErrorKind::PrecisionMismatch,
                            CheckErrorKind::PrecisionMismatch
                        )
                ),
                "{:?}",
                report.errors
            );
        }
    }
}

#[test]
fn a_failed_signature_preserves_its_other_binder_bounds() {
    assert_one_report_per_site(
        core_fixture(include_str!(
            "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/other_binder_bounds.json"
        )),
        &[("f8e4m3", 26)],
    );
}

#[test]
fn matching_deep_signature_and_annotation_share_a_declaration_owner() {
    for display_span in ["same", "source:16..22"] {
        let source = format!(
            "(defsig {{}} classify (t-fn {{}} (t-prim {{span: \"{display_span}\"}} f8e4m3) (t-prim {{}} i32)))\n\
             (def {{}} classify (fn {{}} (params {{}} (x {{type: (t-prim {{span: \"{display_span}\"}} f8e4m3)}})) (lit {{type: (t-prim {{}} i32)}} 0)))"
        );
        let program = chelis_deep::parse_and_stamp_file(&source).expect("parse two authored sites");
        for result in [
            check_ir_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            let report = result.expect_err("the declaration-owned annotation must reject");
            assert_eq!(report.errors.len(), 1, "{:?}", report.errors);
            assert!(
                report
                    .errors
                    .iter()
                    .all(|error| error.message.contains("f8e4m3"))
            );
        }
    }
}

#[test]
fn failed_recursive_signatures_keep_each_members_diagnostics() {
    assert_one_report_per_site(
        core_fixture(include_str!(
            "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/recursive_signatures.json"
        )),
        &[("f8e4m3", 12), ("f8e5m2", 91)],
    );
}

#[test]
fn a_bad_cast_target_is_checked_even_when_its_operand_already_failed() {
    let program = core_fixture(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/failed_operand_cast.json"
    ));
    for result in [
        check_ir_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
        check_typed_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
    ] {
        let report = result.expect_err("both reserved sites must reject");
        assert_eq!(report.errors.len(), 2, "{:?}", report.errors);
        assert!(report.errors[0].message.contains("f8e4m3"));
        assert!(report.errors[1].message.contains("f8e5m2"));
    }
}

#[test]
fn an_arity_mismatch_does_not_recheck_parameter_annotations() {
    for extra in ["", " y"] {
        let program = chelis_deep::parse_and_stamp_file(&format!(
            "(defsig {{}} classify (t-fn {{}} (t-prim {{}} f8e4m3) (t-prim {{}} i32)))\n\
             (def {{}} classify (fn {{}} (params {{}} (x {{type: (t-prim {{}} f8e5m2)}}){extra}) (lit {{type: (t-prim {{}} i32)}} 0)))"
        )).expect("parse independent annotations");
        for result in [
            check_ir_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            let report = result.expect_err("both reserved sites reject");
            let reserved: Vec<_> = report
                .errors
                .iter()
                .filter(|error| {
                    matches!(
                        error.kind,
                        chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
                    )
                })
                .collect();
            assert_eq!(reserved.len(), 2, "{extra:?}: {:?}", report.errors);
            assert!(reserved[0].message.contains("f8e4m3"));
            assert!(reserved[1].message.contains("f8e5m2"));
        }
    }
}

#[test]
fn checked_parameters_receive_their_actual_declared_types() {
    use chelis_types::types::{Dim, Prim, TensorPrec, Type};
    for (fixture, function, expected) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/parameter_scalar.json"
            ),
            "identity",
            Type::Prim(Prim::F32),
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/parameter_reference.json"
            ),
            "add_self",
            Type::Ref(Box::new(Type::Tensor(
                vec![Dim::Lit(3)],
                TensorPrec::Concrete(Prim::F32),
            ))),
        ),
    ] {
        let program = core_fixture(fixture);
        for result in [
            check_ir_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            let checked = result.expect("valid declared parameter");
            let signature = checked
                .signature_inference()
                .functions
                .get(function)
                .expect("function report");
            assert_eq!(signature.params[0].checked_type, expected);
        }
    }
}

#[test]
fn a_canonical_parameter_hole_does_not_erase_declared_shape_evidence() {
    let source =
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/parameter_shape.json");
    let desugared = core_fixture(source);
    let text = chelis_deep::printer::print_canonical_flat(&desugared);
    let stamped = chelis_deep::parse_and_stamp_file(&text).expect("stamp canonical Deep");
    for (carrier, expressions) in [("desugared", desugared), ("stamped", stamped)] {
        let report = check_ir_program(
            &expressions,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect_err("the zero-stride validator must retain declared tensor shapes");
        assert_eq!(report.errors.len(), 1, "{carrier}: {:?}", report.errors);
        assert!(
            matches!(
                report.errors[0].kind,
                chelis_types::errors::CheckErrorKind::DimensionMismatch
            ),
            "{carrier}: {:?}",
            report.errors
        );
    }
}

#[test]
fn source_annotation_presence_and_read_only_inference_survive_transport() {
    let program = core_fixture(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/read_only.json"
    ));
    let text = chelis_deep::printer::print_canonical_flat(&program);
    let printed = chelis_deep::parse_and_stamp_file(&text).expect("parse printed input");
    let encoded = serde_json::to_vec(&program).expect("encode input");
    let decoded: Vec<chelis_deep::Expr> = serde_json::from_slice(&encoded).expect("decode input");
    for expressions in [&program, &printed, &decoded] {
        for result in [
            check_ir_program(
                expressions,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                expressions,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            let checked = result.expect("read-only inference stays valid");
            let function = checked
                .signature_inference()
                .functions
                .get("readonly")
                .expect("function report");
            assert_eq!(function.params.len(), 2);
            assert!(!function.params[0].written);
            assert!(function.params[0].inferred_read_only);
            assert!(function.params[1].written);
            assert!(!function.params[1].inferred_read_only);
        }
    }
}

#[test]
fn a_parameter_hole_does_not_erase_an_independent_body_mismatch() {
    for (body, accepted) in [
        ("(lit {type: (t-prim {} i32)} 0)", true),
        ("(lit {} true)", false),
    ] {
        let program = chelis_deep::parse_and_stamp_file(&format!(
            "(defsig {{}} classify (t-fn {{}} (t-prim {{}} i32) (t-prim {{}} i32)))\n\
             (def {{}} classify (fn {{}} (params {{}} (x {{type: (t-var {{}} _)}})) {body}))"
        ))
        .expect("parameter-hole Deep fixture");
        for result in [
            check_ir_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            if accepted {
                result.expect("the body agrees with the declared result");
            } else {
                let report = result.expect_err("the body must satisfy the declared result");
                assert!(
                    report.errors.iter().any(|error| matches!(
                        error.kind,
                        chelis_types::errors::CheckErrorKind::TypeMismatch
                    )),
                    "{:?}",
                    report.errors
                );
            }
        }
    }
}

#[test]
fn multiple_reserved_sites_survive_printing_and_serialization() {
    for (source, expected) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/distinct_parameters.json"
            ),
            &["f8e4m3", "f8e5m2"][..],
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/repeated_parameters.json"
            ),
            &["f8e4m3"][..],
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/parameter_return.json"
            ),
            &["f8e4m3", "f8e5m2"][..],
        ),
        (
            include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/body_cast.json"),
            &["f8e4m3", "f8e5m2"][..],
        ),
    ] {
        let program = core_fixture(source);
        let text = chelis_deep::printer::print_canonical_flat(&program);
        let printed = chelis_deep::parse_and_stamp_file(&text).expect("parse printed input");
        let encoded = serde_json::to_vec(&program).expect("encode input");
        let decoded: Vec<chelis_deep::Expr> =
            serde_json::from_slice(&encoded).expect("decode input");
        for expressions in [&program, &printed, &decoded] {
            for result in [
                check_ir_program(
                    expressions,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
                check_typed_program(
                    expressions,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ] {
                let report = result.expect_err("all authored reserved sites reject");
                assert_eq!(
                    report.errors.len(),
                    expected.len(),
                    "{source}: {:?}",
                    report.errors
                );
                for (error, name) in report.errors.iter().zip(expected.iter()) {
                    assert!(
                        matches!(
                            error.kind,
                            chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
                        ),
                        "{error:?}"
                    );
                    assert!(error.message.contains(name), "{error:?}");
                    assert!(error.span_offset.is_some(), "{error:?}");
                    assert!(error.span_id.is_some(), "{error:?}");
                }
            }
        }
    }
}

#[test]
fn signature_ownership_requires_explicit_authority() {
    for (fixture, accepted) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/authority_absent.json"
            ),
            false,
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/authority_explicit.json"
            ),
            true,
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/authority_excluded.json"
            ),
            false,
        ),
    ] {
        let program = core_fixture(fixture);
        for result in [
            check_ir_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                &program,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            if accepted {
                result.expect("the applicable precision binder remains valid");
            } else {
                let report =
                    result.expect_err("a precision outside the explicit clause remains invalid");
                assert!(
                    report
                        .errors
                        .iter()
                        .any(|error| error.message.contains("primitive")
                            && error.message.contains("`p`")),
                    "{:?}",
                    report.errors
                );
            }
        }
    }
}

#[test]
fn a_printed_program_preserves_its_diagnostic_count() {
    let program = core_fixture(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/parameter.json"
    ));
    let text = chelis_deep::printer::print_canonical_flat(&program);
    let reparsed = chelis_deep::parse_and_stamp_file(&text).expect("parse printed Deep");
    let saved = serde_json::to_vec(&program).expect("encode Deep");
    let decoded: Vec<chelis_deep::Expr> = serde_json::from_slice(&saved).expect("decode Deep");
    let mut counts = Vec::new();
    for (carrier, expressions) in [
        ("desugared", &program),
        ("printed", &reparsed),
        ("decoded", &decoded),
    ] {
        for result in [
            check_ir_program(
                expressions,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                expressions,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            let report = result.expect_err("the reserved type must reject on every carrier");
            counts.push((carrier, report.errors.len()));
        }
    }
    assert_eq!(
        counts,
        vec![
            ("desugared", 1),
            ("desugared", 1),
            ("printed", 1),
            ("printed", 1),
            ("decoded", 1),
            ("decoded", 1)
        ]
    );
}

#[test]
fn a_divergent_binder_lowering_does_not_duplicate_the_reserved_site() {
    let program = core_fixture(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/divergent_binder.json"
    ));
    for result in [
        check_ir_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
        check_typed_program(
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ),
    ] {
        let report = result.expect_err("the reserved site must reject");
        let reserved = report
            .errors
            .iter()
            .filter(|error| error.message.contains("f8e4m3"))
            .count();
        assert_eq!(reserved, 1, "{:?}", report.errors);
    }
}

#[test]
fn handwritten_deep_cannot_bypass_the_type_resolver() {
    for name in ["f8e4m3", "f8e5m2"] {
        let source = format!(
            "(defsig {{}} f (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))\n\
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        );
        let exprs = chelis_deep::parse_and_stamp_file(&source).expect("stamp Deep fixture");
        for result in [
            check_ir_program(
                &exprs,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                &exprs,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            let report = result.expect_err("reserved primitive must reject");
            assert!(
                report.errors.iter().any(|error| {
                    matches!(
                        error.kind,
                        chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
                    ) && error.message.contains(name)
                }),
                "{:?}",
                report.errors
            );
        }
    }
}

#[test]
fn handwritten_reserved_type_variable_cannot_bypass_the_resolver() {
    for name in ["f8e4m3", "f8e5m2"] {
        let source = format!(
            "(defsig {{}} f ({name}) (t-fn {{}} (t-var {{}} {name}) (t-var {{}} {name})))\n\
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        );
        let exprs = chelis_deep::parse_and_stamp_file(&source).expect("stamp Deep fixture");
        for result in [
            check_ir_program(
                &exprs,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                &exprs,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            let report = result.expect_err("reserved type variable must reject");
            assert!(
                report
                    .errors
                    .iter()
                    .any(|error| { error.message.contains(name) }),
                "{:?}",
                report.errors
            );
        }
    }
}

#[test]
fn aliases_fields_and_value_annotations_reject_fp8() {
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_alias.json"),
        "a type alias",
    );
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_field.json"),
        "an ADT field",
    );
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_value.json"),
        "a value annotation",
    );
}

fn assert_accepted(source: &str) {
    let desugared = core_fixture(source);
    let text = chelis_deep::printer::print_canonical_flat(&desugared);
    let stamped = chelis_deep::parse_and_stamp_file(&text).expect("stamp canonical Deep");
    for exprs in [&desugared, &stamped] {
        for result in [
            check_ir_program(
                exprs,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
            check_typed_program(
                exprs,
                chelis_types::ProgramOrigin::Unlinked,
                &mut chelis_types::NativeServices::new(None),
            ),
        ] {
            assert!(result.is_ok(), "{source}: {result:?}");
        }
    }
}

#[test]
fn scalar_parameter_and_return_rejected() {
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_scalar.json"),
        "a scalar parameter and return",
    );
}

#[test]
fn standalone_sig_rejected() {
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_sig.json"),
        "a standalone sig",
    );
}

#[test]
fn reference_type_rejected() {
    assert_rejected(
        include_str!(
            "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_reference.json"
        ),
        "a reference type",
    );
}

#[test]
fn tuple_element_rejected() {
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_tuple.json"),
        "a tuple element",
    );
}

#[test]
fn arrow_parameter_rejected() {
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_arrow.json"),
        "an arrow parameter",
    );
}

#[test]
fn list_element_rejected() {
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_list.json"),
        "a List element",
    );
}

/// DISPOSITION LOCK. The pre-existing locks (`f8e4m3_rejection.rs`) must
/// keep rejecting after this repair: the tensor slot and the cast target.
#[test]
fn tensor_slot_and_cast_target_still_rejected() {
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_tensor.json"),
        "a tensor element slot",
    );
    assert_rejected(
        include_str!("fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_cast.json"),
        "a cast target",
    );
}

/// DISPOSITION LOCK. Every currently-active float dtype must keep checking
/// clean; the admissibility check added for `f8e4m3` must not widen.
#[test]
fn every_active_float_is_still_accepted() {
    for name in ["f32", "f64", "f16", "bf16"] {
        let legacy = with_precision(
            include_str!(
                "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/rejected_scalar.json"
            ),
            name,
        );
        let text = chelis_deep::printer::print_canonical_flat(&legacy);
        let stamped =
            chelis_deep::parse_and_stamp_file(&text).expect("stamp active primitive fixture");
        for program in [&legacy, &stamped] {
            for result in [
                check_ir_program(
                    program,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
                check_typed_program(
                    program,
                    chelis_types::ProgramOrigin::Unlinked,
                    &mut chelis_types::NativeServices::new(None),
                ),
            ] {
                result.expect("active float remains admissible");
            }
        }
    }
}

/// DISPOSITION LOCK. An ordinary explicitly listed lowercase binder still
/// checks clean; the reserved-name routing must not widen to catch it.
#[test]
fn an_ordinary_type_variable_still_checks_clean() {
    assert_accepted(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/accepted_variable.json"
    ));
    assert_accepted(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/accepted_alias.json"
    ));
    assert_accepted(include_str!(
        "fixtures/core/tests/issue_1606_f8e4m3_scalar_position/accepted_field.json"
    ));
}
