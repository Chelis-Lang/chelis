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

use chelis_surf::desugar::desugar_program;
use chelis_surf::parser::parse_str;
use chelis_surf::resugar::resugar_program;
use chelis_types::{check_ir_program, check_typed_program};
use serde_json::Value;

fn surf_to_deep(source: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("surf parse");
    chelis_macros::expand_program(
        &desugar_program(&decls).expect("Surf fixture must desugar"),
        &chelis_macros::ExpansionOptions::default(),
    )
    .expect("macro expand")
    .into_exprs()
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
        ("ir", check_ir_program(program)),
        ("typed", check_typed_program(program)),
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

/// Rebuild the property def with its fn's `params` node replaced by a
/// structural list, returning the def constructor's rejection: the
/// `property_quantifiers` metadata must match a canonical params node.
fn fn_params_as_bare_list_rejection(
    mut program: Vec<chelis_deep::Expr>,
) -> chelis_deep::node::NodeError {
    let def_index = program
        .iter()
        .position(|expr| expr.tag() == Some(chelis_deep::DeepTag::Def))
        .expect("fixture contains a property def");
    let chelis_deep::Expr::Node(def, _) = program.remove(def_index) else {
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
    let function = chelis_deep::node::Node::try_new(function_tag, function_meta, function_children)
        .expect("a fn node admits a structural params list");
    def_children.push(chelis_deep::Expr::Node(Box::new(function), function_span));
    chelis_deep::node::Node::try_new(def_tag, def_meta, def_children)
        .expect_err("a noncanonical params carrier must fail closed at the def constructor")
}

fn assert_rejected(src: &str, position: &str) {
    for name in ["f8e4m3", "f8e5m2"] {
        let source = src.replace("f8e4m3", name);
        let desugared = surf_to_deep(&source);
        let text = chelis_deep::printer::print_canonical_flat(&desugared);
        let stamped = chelis_deep::parse_and_stamp_file(&text).expect("stamp canonical Deep");
        for (carrier, exprs) in [("desugared", desugared), ("stamped", stamped)] {
            for (entry, result) in [
                ("ir", check_ir_program(&exprs)),
                ("typed", check_typed_program(&exprs)),
            ] {
                let report = result.err().unwrap_or_else(|| {
                    panic!("{name} in {position} must reject at {carrier}/{entry}: {source}")
                });
                assert!(
                    report.errors.iter().any(|error| {
                        error.message.contains(name)
                            && error.message.contains("spec/04-type-system.md §1.1.1")
                    }),
                    "{carrier}/{entry}/{position}: {:?}",
                    report.errors
                );
            }
        }
    }
}

fn assert_forbidden_surf_binder(source: &str, name: &str, position: &str) {
    let error = parse_str(source).expect_err("reserved dtype vocabulary cannot be rebound");
    let message = error.to_string();
    assert!(
        message.contains(&format!("`{name}`"))
            && message.contains("cannot be a declaration binder"),
        "wrong forbidden-binder diagnostic for `{name}` in {position}: {message}"
    );
}

/// The expected sequence names one declaration-owned diagnostic per spelling.
fn assert_one_report_per_site(source: &str, names: &[&str]) {
    let program = surf_to_deep(source);
    let mut search_start = 0;
    let expected = names
        .iter()
        .map(|name| {
            let relative = source[search_start..]
                .find(name)
                .unwrap_or_else(|| panic!("expected `{name}` after byte {search_start}: {source}"));
            let offset = search_start + relative;
            search_start = offset + name.len();
            (
                *name,
                offset,
                format!("source:{offset}..{}", offset + name.len()),
            )
        })
        .collect::<Vec<_>>();
    for (entry, result) in [
        ("ir", check_ir_program(&program)),
        ("typed", check_typed_program(&program)),
    ] {
        let report = result.expect_err("reserved dtype sites must reject");
        assert_eq!(
            report.errors.len(),
            expected.len(),
            "{entry}: {source}: {:?}",
            report.errors
        );
        for (error, (name, offset, span)) in report.errors.iter().zip(&expected) {
            assert!(
                matches!(
                    error.kind,
                    chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
                ),
                "{entry}: {error:?}"
            );
            assert!(error.message.contains(name), "{entry}: {error:?}");
            assert_eq!(error.span_offset, Some(*offset), "{entry}: {error:?}");
            assert_eq!(
                error.span_id.as_deref(),
                Some(span.as_str()),
                "{entry}: {error:?}"
            );
        }
    }
}

fn surf_with_legacy_tensor_precision(source: &str, name: &str) -> Vec<chelis_deep::Expr> {
    let decls = parse_str(source).expect("parse current Surf AST");
    let mut encoded = serde_json::to_value(decls).expect("serialize Surf AST");
    let precision = &mut encoded[0]["TypeAlias"]["ty"]["Tensor"][1];
    assert_eq!(
        precision["name"],
        Value::String(name.to_string()),
        "fixture must select the tensor precision child"
    );
    *precision = Value::String(name.to_string());
    let legacy: Vec<chelis_surf::ast::Decl> =
        serde_json::from_value(encoded).expect("decode legacy precision string");
    desugar_program(&legacy).expect("Surf fixture must desugar")
}

#[test]
fn a_reserved_parameter_site_reports_once_at_both_entries() {
    for name in ["f8e4m3", "f8e5m2"] {
        assert_one_report_per_site(&format!("def classify(x: {name}) -> i32 = 0i32"), &[name]);
    }
}

#[test]
fn no_clause_inline_tensor_precision_is_rejected_as_undeclared() {
    let source = "def inspect(x: tensor[3, p]) -> tensor[3, p] = x";
    let program = surf_to_deep(source);
    for result in [check_ir_program(&program), check_typed_program(&program)] {
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

    let explicit = surf_to_deep("def inspect[p](x: tensor[3, p]) -> tensor[3, p] = x");
    for result in [check_ir_program(&explicit), check_typed_program(&explicit)] {
        result.expect("an explicit matching clause remains valid");
    }

    let excluded = surf_to_deep("def inspect[q](x: tensor[3, p]) -> tensor[3, p] = x");
    for result in [check_ir_program(&excluded), check_typed_program(&excluded)] {
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
        for ty in [name.to_string(), format!("tensor[3, {name}]")] {
            assert_one_report_per_site(
                &format!("@property classify forall(x: {ty}):\n  true"),
                &[name],
            );
        }
    }
}

#[test]
fn hand_authored_property_parameter_contract_remains_independent_from_defsig() {
    let program = hand_authored_deep_property("f64", "f32");
    for (entry, result) in [
        ("ir", check_ir_program(&program)),
        ("typed", check_typed_program(&program)),
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
        ("ir", check_ir_program(&program)),
        ("typed", check_typed_program(&program)),
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
                ("ir", check_ir_program(&program)),
                ("typed", check_typed_program(&program)),
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
                ("ir", check_ir_program(&program)),
                ("typed", check_typed_program(&program)),
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
            ("ir", check_ir_program(&invalid)),
            ("typed", check_typed_program(&invalid)),
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

        // A noncanonical params carrier has no admitted spelling: the def
        // constructor checks `property_quantifiers` against the fn's params.
        let noncanonical = fn_params_as_bare_list_rejection(hand_authored_deep_property_slots(
            std::slice::from_ref(&reserved),
            std::slice::from_ref(&reserved),
            "(lit {} true)",
        ));
        assert!(
            noncanonical.to_string().contains("property_quantifiers"),
            "{name}: {noncanonical}"
        );

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
fn property_quantifier_recovery_keeps_an_independent_body_error() {
    use chelis_types::errors::CheckErrorKind;

    let program = surf_to_deep("@property classify forall(x: f8e4m3):\n  0i32");
    for result in [check_ir_program(&program), check_typed_program(&program)] {
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
        let source = format!("type Rejected = tensor[3, {name}]");
        let program = surf_with_legacy_tensor_precision(&source, name);
        for (entry, result) in [
            ("ir", check_ir_program(&program)),
            ("typed", check_typed_program(&program)),
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
        "def left(x: f8e4m3) -> i32 = 0i32\ndef right(x: f8e5m2) -> i32 = 0i32",
        &["f8e4m3", "f8e5m2"],
    );
}

#[test]
fn repeated_reserved_spelling_in_distinct_declarations_is_not_deduplicated() {
    assert_one_report_per_site(
        "def left(x: f8e4m3) -> i32 = 0i32\ndef right(x: f8e4m3) -> i32 = 0i32",
        &["f8e4m3", "f8e4m3"],
    );
}

#[test]
fn calls_propagate_the_failed_signature_without_a_second_report() {
    assert_one_report_per_site(
        "def classify(x: f8e4m3) -> i32 = 0i32\nresult = classify(1i32)",
        &["f8e4m3"],
    );
}

#[test]
fn one_signature_keys_reserved_diagnostics_by_spelling() {
    assert_one_report_per_site(
        "def classify(x: f8e4m3, y: f8e5m2) -> i32 = 0i32",
        &["f8e4m3", "f8e5m2"],
    );
    assert_one_report_per_site(
        "def classify(x: f8e4m3, y: f8e4m3) -> i32 = 0i32",
        &["f8e4m3"],
    );
}

#[test]
fn parameter_and_return_sites_each_report_once() {
    assert_one_report_per_site(
        "def classify(x: f8e4m3) -> f8e5m2 = x",
        &["f8e4m3", "f8e5m2"],
    );
}

#[test]
fn nested_type_components_each_report_once() {
    for source in [
        "def classify(x: (f8e4m3, f8e5m2)) -> i32 = 0i32",
        "def classify(x: (f8e4m3) -> f8e5m2) -> i32 = 0i32",
        "def classify(x: Dict[f8e4m3, f8e5m2]) -> i32 = 0i32",
    ] {
        assert_one_report_per_site(source, &["f8e4m3", "f8e5m2"]);
    }
}

#[test]
fn a_failed_signature_does_not_hide_an_independent_body_site() {
    let source = "def classify(x: f8e4m3) -> i32 = cast(0i32, f8e5m2)";
    let program = surf_to_deep(source);
    for result in [check_ir_program(&program), check_typed_program(&program)] {
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
        let start = source.find("cast(").expect("cast site");
        let expected = format!("surf:{start}..{}", source.len());
        assert_eq!(report.errors[1].span_offset, Some(start));
        assert_eq!(report.errors[1].span_id.as_deref(), Some(expected.as_str()));
    }
}

#[test]
fn declaration_ownership_does_not_absorb_a_same_spelling_cast_failure() {
    let source = "def classify(x: f8e4m3) -> i32 = cast(0i32, f8e4m3)";
    let program = surf_to_deep(source);
    for result in [check_ir_program(&program), check_typed_program(&program)] {
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
        let start = source.find("cast(").expect("cast site");
        let expected = format!("surf:{start}..{}", source.len());
        assert_eq!(report.errors[1].span_offset, Some(start));
        assert_eq!(report.errors[1].span_id.as_deref(), Some(expected.as_str()));
    }
}

#[test]
fn a_rejected_tensor_precision_preserves_dimensions_for_the_body() {
    use chelis_types::errors::CheckErrorKind;

    for name in ["f8e4m3", "f8e5m2"] {
        let source = format!("def inspect(x: tensor[3, {name}]) -> i64 = shape(x, 1i32)");
        let program = surf_to_deep(&source);
        let offset = source.find(name).expect("reserved precision site");
        let span = format!("source:{offset}..{}", offset + name.len());

        for (entry, result) in [
            ("ir", check_ir_program(&program)),
            ("typed", check_typed_program(&program)),
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
            assert!(
                dimensions[0]
                    .message
                    .contains("shape axis 1 is out of bounds for rank 1 tensor"),
                "{entry}/{name}: {dimensions:?}"
            );
        }
    }
}

#[test]
fn a_tensor_element_reserved_site_has_one_located_owner() {
    for name in ["f8e4m3", "f8e5m2"] {
        for source in [
            format!("def classify(x: tensor[3, {name}]) -> i32 = 0i32"),
            format!("def classify(x: tensor[3,   {name}  ]) -> i32 = 0i32"),
        ] {
            assert_one_report_per_site(&source, &[name]);
        }
    }
}

#[test]
fn deep_surf_deep_roundtrip_preserves_tensor_precision_diagnostic_span() {
    for name in ["f8e4m3", "f8e5m2"] {
        let source = format!("def classify(x: tensor[3, {name}]) -> i32 = 0i32");
        let original = desugar_program(&parse_str(&source).expect("parse source"))
            .expect("Surf fixture must desugar");
        let restored =
            desugar_program(&resugar_program(&original).expect("resugar direct Deep program"))
                .expect("Surf fixture must desugar");
        let offset = source.find(name).expect("reserved precision site");
        let expected_span = format!("source:{offset}..{}", offset + name.len());

        for (carrier, program) in [
            ("original", original.as_slice()),
            ("restored", restored.as_slice()),
        ] {
            for (entry, result) in [
                ("ir", check_ir_program(program)),
                ("typed", check_typed_program(program)),
            ] {
                let report = result.expect_err("reserved precision must reject");
                assert_eq!(
                    report.errors.len(),
                    1,
                    "{carrier}/{entry}/{name}: {:?}",
                    report.errors
                );
                let error = &report.errors[0];
                assert!(
                    matches!(
                        error.kind,
                        chelis_types::errors::CheckErrorKind::UnsupportedTensorPrecision
                    ),
                    "{carrier}/{entry}/{name}: {error:?}"
                );
                assert_eq!(
                    error.span_offset,
                    Some(offset),
                    "{carrier}/{entry}/{name}: {error:?}"
                );
                assert_eq!(
                    error.span_id.as_deref(),
                    Some(expected_span.as_str()),
                    "{carrier}/{entry}/{name}: {error:?}"
                );
            }
        }
    }
}

#[test]
fn valid_parts_of_a_failed_signature_still_constrain_the_body() {
    use chelis_types::errors::CheckErrorKind;
    for (source, expected_kind) in [
        (
            "def classify(x: f8e4m3) -> i32 = true",
            CheckErrorKind::TypeMismatch,
        ),
        (
            "def classify(x: f8e4m3, y: i32) -> i32 = add(y, true)",
            CheckErrorKind::PrecisionMismatch,
        ),
    ] {
        let program = surf_to_deep(source);
        for result in [check_ir_program(&program), check_typed_program(&program)] {
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
        "def classify[p: Float](x: f8e4m3, y: p) -> p = cast(y, p)",
        &["f8e4m3"],
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
        for result in [check_ir_program(&program), check_typed_program(&program)] {
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
        "def left(x: f8e4m3, n: i32) -> i32 = if n == 0 then 0i32 else right(x, n - 1)\n\
         def right(x: f8e5m2, n: i32) -> i32 = if n == 0 then 0i32 else left(x, n - 1)",
        &["f8e4m3", "f8e5m2"],
    );
}

#[test]
fn a_bad_cast_target_is_checked_even_when_its_operand_already_failed() {
    let program = surf_to_deep("def classify(x: f8e4m3) -> i32 = cast(x, f8e5m2)");
    for result in [check_ir_program(&program), check_typed_program(&program)] {
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
        for result in [check_ir_program(&program), check_typed_program(&program)] {
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
    for source in [
        "def identity(x: f32) -> f32 = x",
        "def add_self(x: &tensor[3, f32]) -> tensor[3, f32] = add(x, x)",
    ] {
        let program = surf_to_deep(source);
        for result in [check_ir_program(&program), check_typed_program(&program)] {
            let checked = result.expect("valid declared parameter");
            let text = chelis_deep::printer::print_canonical_flat(checked.exprs());
            assert!(
                !text.contains("type: (t-var {} _)"),
                "unresolved parameter hole: {text}"
            );
            if source.contains("&tensor") {
                assert!(
                    text.contains(
                        "(x {type: (t-ref {} (t-tensor {} (d-lit {} 3) (t-prim {} f32)))})"
                    ),
                    "{text}"
                );
            } else {
                assert!(text.contains("(x {type: (t-prim {} f32)})"), "{text}");
            }
        }
    }
}

#[test]
fn a_canonical_parameter_hole_does_not_erase_declared_shape_evidence() {
    let source = "def f(x: tensor[1, 3, 8, 8, f32], k: tensor[8, 3, 3, 3, f32]) -> tensor[1, 8, 6, 6, f32] = conv(&x, &k, [0i64, 0i64], [(0i64, 0i64), (0i64, 0i64)])";
    let desugared = surf_to_deep(source);
    let text = chelis_deep::printer::print_canonical_flat(&desugared);
    let stamped = chelis_deep::parse_and_stamp_file(&text).expect("stamp canonical Deep");
    for (carrier, expressions) in [("desugared", desugared), ("stamped", stamped)] {
        let report = check_ir_program(&expressions)
            .expect_err("the zero-stride validator must retain declared tensor shapes");
        assert_eq!(report.errors.len(), 1, "{carrier}: {:?}", report.errors);
        assert!(
            report.errors[0].message.contains("positive stride"),
            "{carrier}: {:?}",
            report.errors
        );
        assert!(
            !report.errors[0]
                .message
                .contains("requires concrete tensor argument metadata"),
            "{carrier}: {:?}",
            report.errors
        );
    }
}

#[test]
fn source_annotation_presence_and_read_only_inference_survive_transport() {
    let program = surf_to_deep("def readonly(x, y: tensor[4, f32]) = add(x, y)");
    let text = chelis_deep::printer::print_canonical_flat(&program);
    let printed = chelis_deep::parse_and_stamp_file(&text).expect("parse printed input");
    let encoded = serde_json::to_vec(&program).expect("encode input");
    let decoded: Vec<chelis_deep::Expr> = serde_json::from_slice(&encoded).expect("decode input");
    for expressions in [&program, &printed, &decoded] {
        for result in [
            check_ir_program(expressions),
            check_typed_program(expressions),
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
    for (body, accepted) in [("0i32", true), ("true", false)] {
        let program = surf_to_deep(&format!(
            "sig classify: i32 -> i32\ndef classify(x: _) = {body}"
        ));
        for result in [check_ir_program(&program), check_typed_program(&program)] {
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
            "def classify(x: f8e4m3, y: f8e5m2) -> i32 = 0i32",
            &["f8e4m3", "f8e5m2"][..],
        ),
        (
            "def classify(x: f8e4m3, y: f8e4m3) -> i32 = 0i32",
            &["f8e4m3"][..],
        ),
        (
            "def classify(x: f8e4m3) -> f8e5m2 = x",
            &["f8e4m3", "f8e5m2"][..],
        ),
        (
            "def classify(x: f8e4m3) -> i32 = cast(0i32, f8e5m2)",
            &["f8e4m3", "f8e5m2"][..],
        ),
    ] {
        let program = surf_to_deep(source);
        let text = chelis_deep::printer::print_canonical_flat(&program);
        let printed = chelis_deep::parse_and_stamp_file(&text).expect("parse printed input");
        let encoded = serde_json::to_vec(&program).expect("encode input");
        let decoded: Vec<chelis_deep::Expr> =
            serde_json::from_slice(&encoded).expect("decode input");
        for expressions in [&program, &printed, &decoded] {
            for result in [
                check_ir_program(expressions),
                check_typed_program(expressions),
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
    for (binders, accepted) in [("", false), ("[p]", true), ("[q]", false)] {
        let program = surf_to_deep(&format!(
            "def inspect{binders}(x: tensor[3, p]) -> i32 = 0i32"
        ));
        for result in [check_ir_program(&program), check_typed_program(&program)] {
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
    let program = surf_to_deep("def classify(x: f8e4m3) -> i32 = 0i32");
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
            check_ir_program(expressions),
            check_typed_program(expressions),
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
    let program = surf_to_deep("def classify[p](x: (tensor[3, p], f8e4m3)) -> i32 = 0i32");
    for result in [check_ir_program(&program), check_typed_program(&program)] {
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
        for result in [check_ir_program(&exprs), check_typed_program(&exprs)] {
            let report = result.expect_err("reserved primitive must reject");
            assert!(
                report.errors.iter().any(|error| {
                    error.message.contains(name)
                        && error.message.contains("spec/04-type-system.md §1.1.1")
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
        for result in [check_ir_program(&exprs), check_typed_program(&exprs)] {
            let report = result.expect_err("reserved type variable must reject");
            assert!(
                report.errors.iter().any(|error| {
                    error.message.contains(name)
                        && error.message.contains("cannot be a `defsig` binder")
                }),
                "{:?}",
                report.errors
            );
        }
    }
}

#[test]
fn aliases_fields_and_value_annotations_reject_fp8() {
    assert_rejected("type ReservedAlias = f8e4m3", "a type alias");
    assert_rejected("type Holder = | Holder { value: f8e4m3 }", "an ADT field");
    assert_rejected("value: f8e4m3 = 1.0", "a value annotation");
}

fn assert_accepted(source: &str) {
    let desugared = surf_to_deep(source);
    let text = chelis_deep::printer::print_canonical_flat(&desugared);
    let stamped = chelis_deep::parse_and_stamp_file(&text).expect("stamp canonical Deep");
    for exprs in [&desugared, &stamped] {
        for result in [check_ir_program(exprs), check_typed_program(exprs)] {
            assert!(result.is_ok(), "{source}: {result:?}");
        }
    }
}

#[test]
fn scalar_parameter_and_return_rejected() {
    assert_rejected(
        "def f(x: f8e4m3) -> f8e4m3 = x",
        "a scalar parameter and return",
    );
}

#[test]
fn standalone_sig_rejected() {
    assert_rejected("sig f: f8e4m3 -> f8e4m3\ndef f(x) = x", "a standalone sig");
}

#[test]
fn explicit_binder_scalar_rejected() {
    for name in ["f8e4m3", "f8e5m2"] {
        assert_forbidden_surf_binder(
            &format!("def f[{name}](x: {name}) -> {name} = x"),
            name,
            "an explicit binder (scalar)",
        );
    }
}

#[test]
fn explicit_binder_tensor_rejected() {
    for name in ["f8e4m3", "f8e5m2"] {
        assert_forbidden_surf_binder(
            &format!("def f[{name}](x: tensor[3, {name}]) -> tensor[3, {name}] = x"),
            name,
            "an explicit binder (tensor)",
        );
    }
}

#[test]
fn reference_type_rejected() {
    assert_rejected("def f(x: &f8e4m3) -> i32 = 0i32", "a reference type");
}

#[test]
fn tuple_element_rejected() {
    assert_rejected("def f(x: (f8e4m3, i32)) -> i32 = 0i32", "a tuple element");
}

#[test]
fn arrow_parameter_rejected() {
    assert_rejected(
        "def f(g: (f8e4m3) -> i32) -> i32 = 0i32",
        "an arrow parameter",
    );
}

#[test]
fn list_element_rejected() {
    assert_rejected("def f(x: List[f8e4m3]) -> i32 = 0i32", "a List element");
}

/// DISPOSITION LOCK. The pre-existing locks (`f8e4m3_rejection.rs`) must
/// keep rejecting after this repair: the tensor slot and the cast target.
#[test]
fn tensor_slot_and_cast_target_still_rejected() {
    assert_rejected(
        "def f(x: tensor[3, f8e4m3]) -> tensor[3, f8e4m3] = x",
        "a tensor element slot",
    );
    assert_rejected("def main() -> f32 = cast(1.0, f8e4m3)", "a cast target");
}

/// DISPOSITION LOCK. Every currently-active float dtype must keep checking
/// clean; the admissibility check added for `f8e4m3` must not widen.
#[test]
fn every_active_float_is_still_accepted() {
    for name in ["f32", "f64", "f16", "bf16"] {
        let src = format!("def f(x: {name}) -> {name} = x");
        assert_accepted(&src);
    }
}

/// DISPOSITION LOCK. An ordinary explicitly listed lowercase binder still
/// checks clean; the reserved-name routing must not widen to catch it.
#[test]
fn an_ordinary_type_variable_still_checks_clean() {
    assert_accepted("def f[a](x: a) -> a = x");
    assert_accepted("type ScalarAlias = f32\nvalue: ScalarAlias = 1.0");
    assert_accepted("type Holder = | Holder { value: f32 }");
}
