//! chelis#1854 acceptance at both checker ingresses.

use chelis_deep::{Expr, parser::parse_str as parse_deep, printer::print_canonical_flat};

use chelis_types::{check_ir_program, check_typed_program};

const UNKNOWN_DTYPES: [(&str, Option<&str>); 5] = [
    ("float32", Some("f32")),
    ("fp32", Some("f32")),
    ("int33", Some("i32")),
    ("double", Some("f64")),
    ("half", Some("f16")),
];
const RESERVED_DTYPES: [&str; 2] = ["f8e4m3", "f8e5m2"];

fn parse_core_fixture(source: &str) -> Vec<Expr> {
    if source.trim_start().starts_with('[') {
        serde_json::from_str(source).expect("lossless AST fixture")
    } else {
        chelis_deep::parser::parse_str(source).expect("Deep fixture")
    }
}

#[derive(Debug, PartialEq, Eq)]
struct Diagnostic {
    kind: String,
    message: String,
    suggestions: Vec<String>,
    severity_bits: u64,
    expected: Option<String>,
    got: Option<String>,
    span_offset: Option<usize>,
    span_id: Option<String>,
}

fn diagnostics(
    result: Result<chelis_types::CheckedProgram, chelis_types::InferResult>,
) -> Vec<Diagnostic> {
    result.err().map_or_else(Vec::new, |report| {
        report
            .errors
            .into_iter()
            .map(|error| Diagnostic {
                kind: error.kind.diagnostic_name().to_string(),
                message: error.message,
                suggestions: error.suggestions,
                severity_bits: error.severity.to_bits(),
                expected: error.expected,
                got: error.got,
                span_offset: error.span_offset,
                span_id: error.span_id,
            })
            .collect()
    })
}

fn assert_both_ingresses_reject_once(program: &[Expr], name: &str, nearest: Option<&str>) {
    let ir = diagnostics(check_ir_program(program));
    let typed = diagnostics(check_typed_program(program));
    assert_eq!(
        typed, ir,
        "checker ingresses must report the same ordered diagnostics for `{name}`"
    );
    assert_eq!(
        ir.len(),
        1,
        "`{name}` must have one owning diagnostic: {ir:?}"
    );
    assert!(
        ir[0].message.contains(&format!("`{name}`")),
        "the diagnostic must name the unknown spelling: {ir:?}"
    );
    if let Some(nearest) = nearest {
        assert!(
            ir[0]
                .suggestions
                .iter()
                .any(|suggestion| suggestion.contains(&format!("`{nearest}`"))),
            "the diagnostic for `{name}` must suggest nearest dtype `{nearest}`: {ir:?}"
        );
    }
}

#[test]
fn unknown_surf_scalar_names_reject_once_at_both_checker_ingresses() {
    for (name, nearest) in UNKNOWN_DTYPES {
        assert_both_ingresses_reject_once(
            &parse_core_fixture(&format!(
                r###"(defsig {{}} ident (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))

(def {{}}
  ident
  (fn {{}}
    (params {{}} (x {{type: (t-var {{}} _)}}))
    (var {{span: "surf:35..36"}} x)))
"###
            )),
            name,
            nearest,
        );
    }
    for name in RESERVED_DTYPES {
        assert_both_ingresses_reject_once(
            &parse_core_fixture(&format!(
                r###"(defsig {{}} ident (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))

(def {{}}
  ident
  (fn {{}}
    (params {{}} (x {{type: (t-var {{}} _)}}))
    (var {{span: "surf:35..36"}} x)))
"###
            )),
            name,
            None,
        );
    }
}

#[test]
fn distinct_unknown_surf_scalar_names_keep_distinct_diagnostics() {
    let program = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_1854_explicit_type_binders/distinct_unknown_surf_scalar_names_keep_distinct_diagnostics_100.json"
    ));
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(
        typed, ir,
        "checker ingresses must preserve diagnostic order"
    );
    assert_eq!(
        ir.len(),
        2,
        "distinct invalid names need distinct owners: {ir:?}"
    );
    for name in ["float32", "fp32"] {
        assert_eq!(
            ir.iter()
                .filter(|diagnostic| diagnostic.message.contains(&format!("`{name}`")))
                .count(),
            1,
            "`{name}` must have exactly one diagnostic: {ir:?}"
        );
    }
}

fn assert_unknown_spelling_order(program: &[Expr], expected: &[&str]) {
    let ir = diagnostics(check_ir_program(program));
    let typed = diagnostics(check_typed_program(program));
    assert_eq!(
        typed, ir,
        "checker ingresses must preserve declaration/name ownership and order"
    );
    let actual = ir
        .iter()
        .filter_map(|diagnostic| {
            expected
                .iter()
                .copied()
                .find(|name| diagnostic.message.contains(&format!("`{name}`")))
        })
        .collect::<Vec<_>>();
    assert_eq!(actual, expected, "wrong unknown-name owners: {ir:?}");
    assert_eq!(
        ir.len(),
        expected.len(),
        "no extra diagnostic may preempt the ownership contract: {ir:?}"
    );
}

fn assert_unknown_spelling_order_amid_other_errors(program: &[Expr], expected: &[&str]) {
    let ir = diagnostics(check_ir_program(program));
    let typed = diagnostics(check_typed_program(program));
    assert_eq!(
        typed, ir,
        "checker ingresses must preserve module-scoped ownership and order"
    );
    let actual = ir
        .iter()
        .filter(|diagnostic| diagnostic.message.contains("unknown primitive type"))
        .filter_map(|diagnostic| {
            expected
                .iter()
                .copied()
                .find(|name| diagnostic.message.contains(&format!("`{name}`")))
        })
        .collect::<Vec<_>>();
    assert_eq!(
        actual, expected,
        "unknown-name owners must remain isolated across modules: {ir:?}"
    );
}

#[test]
fn standalone_signature_and_matching_inline_annotation_share_one_unknown_owner() {
    for source in [
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/standalone_signature_and_matching_inline_annotation_share_one_unknown_owner_173.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/standalone_signature_and_matching_inline_annotation_share_one_unknown_owner_175.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/standalone_signature_and_matching_inline_annotation_share_one_unknown_owner_177.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/standalone_signature_and_matching_inline_annotation_share_one_unknown_owner_179.json"
        ),
    ] {
        assert_unknown_spelling_order(&parse_core_fixture(source), &["float32"]);
    }

    for source in [
        "(defsig {} ident
           (t-fn {} (t-prim {} float32) (t-prim {} float32)))
         (def {} ident
           (fn {} (params {} (x {type: (t-prim {} float32)}))
             (var {} x)))",
        "(def {} ident
           (fn {} (params {} (x {type: (t-prim {} float32)}))
             (var {} x)))
         (defsig {} ident
           (t-fn {} (t-prim {} float32) (t-prim {} float32)))",
    ] {
        let program = parse_deep(source).expect("Deep fixture must parse");
        assert_unknown_spelling_order(&program, &["float32"]);
    }
}

#[test]
fn explicit_binders_share_reserved_primitive_diagnostics() {
    for name in RESERVED_DTYPES {
        assert_both_ingresses_reject_once(
            &parse_core_fixture(&format!(
                r###"(defsig {{}} ident (a) (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))

(def {{}}
  ident
  (fn {{}}
    (params {{}} (x {{type: (t-prim {{}} {name})}}))
    (var {{span: "surf:68..69"}} x)))
"###
            )),
            name,
            None,
        );
    }
}

#[test]
fn unknown_primitive_owners_preserve_spelling_declaration_order_and_module_scope() {
    assert_unknown_spelling_order(
        &parse_core_fixture(include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/unknown_primitive_owners_preserve_spelling_declaration_order_and_module_scope_220.json"
        )),
        &["float32", "fp32"],
    );
    for (source, expected) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/unknown_primitive_owners_preserve_spelling_declaration_order_and_module_scope_227.json"
            ),
            ["fp32", "float32"],
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/unknown_primitive_owners_preserve_spelling_declaration_order_and_module_scope_234.json"
            ),
            ["float32", "fp32"],
        ),
    ] {
        assert_unknown_spelling_order(&parse_core_fixture(source), &expected);
    }
    assert_unknown_spelling_order(
        &parse_core_fixture(include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/unknown_primitive_owners_preserve_spelling_declaration_order_and_module_scope_245.json"
        )),
        &["float32", "float32"],
    );

    let modules = parse_deep(
        "(module {} Left
           (defsig {} ident
             (t-fn {} (t-prim {} float32) (t-prim {} float32)))
           (def {} ident
             (fn {} (params {} (x {type: (t-prim {} float32)}))
               (var {} x))))
         (module {} Right
           (defsig {} ident
             (t-fn {} (t-prim {} float32) (t-prim {} float32)))
           (def {} ident
             (fn {} (params {} (x {type: (t-prim {} float32)}))
               (var {} x))))",
    )
    .expect("module fixture must parse");
    assert_unknown_spelling_order_amid_other_errors(&modules, &["float32", "float32"]);
}

#[test]
fn unknown_surf_tensor_precision_names_reject_once_at_both_checker_ingresses() {
    for (name, nearest) in UNKNOWN_DTYPES
        .into_iter()
        .chain(RESERVED_DTYPES.map(|name| (name, None)))
    {
        assert_both_ingresses_reject_once(
            &parse_core_fixture(&format!(
                r###"(defsig {{}}
  ident
  (t-fn {{}} (t-tensor {{}} (d-lit {{}} 2) (t-prim {{}} {name})) (t-prim {{}} f32)))

(def {{}}
  ident
  (fn {{}}
    (params {{}} x)
    (lit {{span: "surf:52..58", surf_literal_style: "explicit", type: (t-prim {{}} f32)}} 0.0)))
"###
            )),
            name,
            nearest,
        );
    }
}

#[test]
fn undeclared_surf_dimension_and_rank_variables_reject_at_both_ingresses() {
    for (source, name, needle) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/undeclared_surf_dimension_and_rank_variables_reject_at_both_ingresses_291.json"
            ),
            "n",
            "undeclared dimension variable",
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/undeclared_surf_dimension_and_rank_variables_reject_at_both_ingresses_296.json"
            ),
            "r",
            "undeclared rank variable",
        ),
    ] {
        let program = parse_core_fixture(source);
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "Surf ingress parity for undeclared `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong Surf undeclared-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn repeated_undeclared_type_dimension_and_rank_names_share_declaration_owners() {
    for (name, slot, needle) in [
        ("a", "(t-var {} a)", "undeclared type variable"),
        (
            "n",
            "(t-tensor {} (d-var {} n) (t-prim {} f32))",
            "undeclared dimension variable",
        ),
        (
            "r",
            "(t-tensor {} (d-rank {} r) (t-prim {} f32))",
            "undeclared rank variable",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} ident (t-fn {{}} {slot} {slot}))
             (def {{}} ident
               (fn {{}} (params {{}} (x {{type: {slot}}})) (var {{}} x)))"
        ))
        .expect("repeated undeclared-name fixture must parse");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "ingress parity for repeated undeclared `{name}`");
        assert_eq!(
            ir.len(),
            1,
            "`{name}` must have one declaration owner: {ir:?}"
        );
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong repeated undeclared-name diagnostic: {ir:?}"
        );
    }
}

#[test]
fn one_spelling_in_distinct_name_resolution_classes_keeps_distinct_diagnostics() {
    let program = parse_deep(
        "(defsig {} classify
           (t-fn {}
             (t-tuple {}
               (t-var {} q)
               (t-tensor {} (d-var {} q) (t-prim {} f32))
               (t-tensor {} (d-rank {} q) (t-prim {} f32)))
             (t-prim {} f32)))
         (def {} classify
           (fn {} (params {} x) (lit {type: (t-prim {} f32)} 0.0)))",
    )
    .expect("diagnostic-class fixture must parse");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(typed, ir, "diagnostic-class ingress parity");
    assert_eq!(
        ir.len(),
        3,
        "distinct name-resolution classes must remain: {ir:?}"
    );
    for needle in [
        "undeclared type variable",
        "undeclared dimension variable",
        "undeclared rank variable",
    ] {
        assert_eq!(
            ir.iter()
                .filter(|diagnostic| {
                    diagnostic.message.contains(needle) && diagnostic.message.contains("`q`")
                })
                .count(),
            1,
            "missing distinct `{needle}` owner: {ir:?}"
        );
    }
}

#[test]
fn undeclared_dimension_owners_are_isolated_by_declaration_and_module() {
    let slot = "(t-tensor {} (d-var {} n) (t-prim {} f32))";
    let declaration_program = parse_deep(&format!(
        "(defsig {{}} first (t-fn {{}} {slot} {slot}))
         (def {{}} first (fn {{}} (params {{}} x) (var {{}} x)))
         (defsig {{}} second (t-fn {{}} {slot} {slot}))
         (def {{}} second (fn {{}} (params {{}} x) (var {{}} x)))"
    ))
    .expect("distinct declaration fixture must parse");
    let module_program = parse_deep(&format!(
        "(module {{}} Left
           (defsig {{}} ident (t-fn {{}} {slot} {slot}))
           (def {{}} ident (fn {{}} (params {{}} x) (var {{}} x))))
         (module {{}} Right
           (defsig {{}} ident (t-fn {{}} {slot} {slot}))
           (def {{}} ident (fn {{}} (params {{}} x) (var {{}} x))))"
    ))
    .expect("module-isolation fixture must parse");

    for (label, program) in [
        ("declarations", declaration_program.as_slice()),
        ("modules", module_program.as_slice()),
    ] {
        let ir = diagnostics(check_ir_program(program));
        let typed = diagnostics(check_typed_program(program));
        assert_eq!(typed, ir, "{label} ingress parity");
        let owners = ir
            .iter()
            .filter(|diagnostic| {
                diagnostic.message.contains("undeclared dimension variable")
                    && diagnostic.message.contains("`n`")
            })
            .collect::<Vec<_>>();
        assert_eq!(
            owners.len(),
            2,
            "{label} must retain two dimension owners: {ir:?}"
        );
        assert!(
            label != "modules"
                || ir
                    .iter()
                    .any(|diagnostic| diagnostic.kind == "DuplicateDefinition"),
            "module isolation must not hide the independent duplicate-definition class: {ir:?}"
        );
        assert!(
            owners[0].span_offset < owners[1].span_offset,
            "{label} owners must remain in declaration order: {ir:?}"
        );
    }
}

#[test]
fn explicit_surf_binders_and_active_primitives_check_clean() {
    for source in [
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/explicit_surf_binders_and_active_primitives_check_clean_444.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/explicit_surf_binders_and_active_primitives_check_clean_445.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/explicit_surf_binders_and_active_primitives_check_clean_446.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/explicit_surf_binders_and_active_primitives_check_clean_447.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/explicit_surf_binders_and_active_primitives_check_clean_448.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/explicit_surf_binders_and_active_primitives_check_clean_449.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/explicit_surf_binders_and_active_primitives_check_clean_450.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/explicit_surf_binders_and_active_primitives_check_clean_452.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/explicit_surf_binders_and_active_primitives_check_clean_453.json"
        ),
    ] {
        let program = parse_core_fixture(source);
        check_ir_program(&program).unwrap_or_else(|report| {
            panic!("IR ingress must accept `{source}`: {:?}", report.errors)
        });
        check_typed_program(&program).unwrap_or_else(|report| {
            panic!("typed ingress must accept `{source}`: {:?}", report.errors)
        });
    }
}

#[test]
fn declaration_body_annotations_preserve_one_explicit_binder_scope() {
    for source in [
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/declaration_body_annotations_preserve_one_explicit_binder_scope_468.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/declaration_body_annotations_preserve_one_explicit_binder_scope_469.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/declaration_body_annotations_preserve_one_explicit_binder_scope_470.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/declaration_body_annotations_preserve_one_explicit_binder_scope_471.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/declaration_body_annotations_preserve_one_explicit_binder_scope_472.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/declaration_body_annotations_preserve_one_explicit_binder_scope_473.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/declaration_body_annotations_preserve_one_explicit_binder_scope_475.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/declaration_body_annotations_preserve_one_explicit_binder_scope_477.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/declaration_body_annotations_preserve_one_explicit_binder_scope_479.json"
        ),
    ] {
        let legacy_program = parse_core_fixture(source);
        let canonical_source = print_canonical_flat(&legacy_program);
        let canonical_program =
            parse_deep(&canonical_source).expect("desugared body-binder Deep must parse");
        for (carrier, program) in [
            ("Surf-desugared Deep", legacy_program.as_slice()),
            ("canonical Deep", canonical_program.as_slice()),
        ] {
            check_ir_program(program).unwrap_or_else(|report| {
                panic!(
                    "normalized IR ingress must accept body binder use through {carrier} in \
                     `{source}`: {:?}",
                    report.errors
                )
            });
            check_typed_program(program).unwrap_or_else(|report| {
                panic!(
                    "typed ingress must accept body binder use through {carrier} in `{source}`: \
                     {:?}",
                    report.errors
                )
            });
        }
    }
}

#[test]
fn expression_tensor_ascriptions_preserve_rigid_dimension_identity() {
    let source = include_str!(
        "fixtures/core/tests/issue_1854_explicit_type_binders/expression_tensor_ascriptions_preserve_rigid_dimension_identity_511.json"
    );
    let legacy_program = parse_core_fixture(source);
    let canonical_source = print_canonical_flat(&legacy_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared expression ascription must parse");

    for (carrier, program) in [
        ("Surf-desugared Deep", legacy_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        let ir = diagnostics(check_ir_program(program));
        let typed = diagnostics(check_typed_program(program));
        assert_eq!(typed, ir, "{carrier} checker ingress parity");
        assert!(
            ir.iter().any(|diagnostic| {
                diagnostic.kind == "DimensionMismatch"
                    && diagnostic.message.contains("`n`")
                    && diagnostic.message.contains("`m`")
                    && diagnostic.message.contains("[04-INF-6]")
            }),
            "{carrier} must reject an expression ascription that collapses distinct rigid \
             declaration dimensions: {ir:?}\n{canonical_source}"
        );
    }
}

#[test]
fn contextual_literal_copies_share_one_unknown_precision_owner() {
    for extent in [1, 3] {
        let mut values = "(var {} Nil)".to_string();
        for value in (1..=extent).rev() {
            values = format!(
                "(app {{}} (var {{}} Cons) (lit {{surf_literal_style: \"unsuffixed\", type: (t-prim {{}} p)}} {value}.0) {values})"
            );
        }
        let source = format!(
            r#"
(defsig {{}} main (t-fn {{}} (t-tensor {{}} (d-lit {{}} {extent}) (t-prim {{}} f32))))
(def {{}} main (fn {{}} (params {{}})
  (let {{span: "surf:31..76"}}
    (bind {{}} xs
      (app {{surf_binding_type: "explicit", type: (t-tensor {{}} (d-lit {{}} {extent}) (t-prim {{}} p))}}
        (var {{}} to_tensor) {values}))
    (var {{span: "surf:72..74"}} xs))))
"#
        );
        let program = parse_core_fixture(&source);
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "checker ingress parity for `{source}`");
        assert_eq!(
            ir.len(),
            1,
            "generated contextual literal copies must not multiply the declaration-owned \
             diagnostic: {ir:?}"
        );
        assert!(
            ir[0].kind == "TypeMismatch"
                && ir[0]
                    .message
                    .contains("unknown primitive type `p` in type annotation")
                && ir[0].span_offset.is_some()
                && ir[0].span_id.is_some()
                && ir[0].suggestions.iter().any(|suggestion| {
                    suggestion.contains("nearest active dtype")
                        && suggestion.contains("declare `p`")
                }),
            "wrong unknown-precision owner: {ir:?}"
        );
    }
}

#[test]
fn defsig_less_function_metadata_shares_the_declaration_owner() {
    let tensor = "(t-tensor {} (d-lit {} 2) (t-prim {} madeup))";
    let program = parse_deep(&format!(
        "(def {{}} helper
           (fn {{type: (t-fn {{}} {tensor} {tensor})}}
             (params {{}} (x {{type: {tensor}}}))
             (var {{}} x)))"
    ))
    .expect("defsig-less function metadata fixture must parse");

    assert_unknown_spelling_order(&program, &["madeup"]);
}

#[test]
fn body_only_surf_binders_check_at_both_ingresses() {
    for source in [
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_surf_binders_check_at_both_ingresses_589.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_surf_binders_check_at_both_ingresses_590.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_surf_binders_check_at_both_ingresses_591.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_surf_binders_check_at_both_ingresses_592.json"
        ),
    ] {
        let program = parse_core_fixture(source);
        check_ir_program(&program).unwrap_or_else(|report| {
            panic!(
                "normalized IR ingress must accept body-only binder in `{source}`: {:?}",
                report.errors
            )
        });
        check_typed_program(&program).unwrap_or_else(|report| {
            panic!(
                "typed ingress must accept body-only binder in `{source}`: {:?}",
                report.errors
            )
        });
    }
}

fn assert_non_function_binder_narrowing_rejects(source: &str, declaration: &str, binders: &[&str]) {
    let legacy_program = parse_core_fixture(source);
    let canonical_source = print_canonical_flat(&legacy_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared non-function Deep must parse");
    for (carrier, program) in [
        ("Surf-desugared Deep", legacy_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        let ir = diagnostics(check_ir_program(program));
        let typed = diagnostics(check_typed_program(program));
        assert_eq!(
            typed, ir,
            "typed-Deep and normalized-IR ingresses must agree for `{declaration}` via {carrier}"
        );
        assert!(
            !ir.is_empty(),
            "`{declaration}` must reject concrete narrowing of its declared scheme via {carrier}"
        );
        for binder in binders {
            assert!(
                ir.iter().any(|diagnostic| {
                    diagnostic.message.contains(declaration)
                        && diagnostic.message.contains(&format!("`{binder}`"))
                        && diagnostic.message.contains("[04-INF-6]")
                }),
                "`{declaration}` must report concrete narrowing of binder `{binder}` via \
                 {carrier}: {ir:?}\n{canonical_source}"
            );
        }
    }
}

#[test]
fn non_function_declaration_binders_reject_concrete_narrowing_at_both_ingresses() {
    for (source, declaration, binders) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/non_function_declaration_binders_reject_concrete_narrowing_at_both_ingresses_647.json"
            ),
            "type_value",
            &["p"][..],
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/non_function_declaration_binders_reject_concrete_narrowing_at_both_ingresses_653.json"
            ),
            "dimension_value",
            &["n"][..],
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/non_function_declaration_binders_reject_concrete_narrowing_at_both_ingresses_659.json"
            ),
            "rank_value",
            &["r"][..],
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/non_function_declaration_binders_reject_concrete_narrowing_at_both_ingresses_665.json"
            ),
            "matrix_value",
            &["rows", "cols"][..],
        ),
    ] {
        assert_non_function_binder_narrowing_rejects(source, declaration, binders);
    }
}

#[test]
fn unconstrained_polymorphic_non_function_value_remains_valid_at_both_ingresses() {
    let legacy_program = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_1854_explicit_type_binders/unconstrained_polymorphic_non_function_value_remains_valid_at_both_ingresses_680.json"
    ));
    let canonical_source = print_canonical_flat(&legacy_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared polymorphic value must parse");
    for (carrier, program) in [
        ("Surf-desugared Deep", legacy_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        check_ir_program(program).unwrap_or_else(|report| {
            panic!(
                "normalized-IR ingress must preserve an unconstrained polymorphic value through \
                 {carrier}: {:?}",
                report.errors
            )
        });
        check_typed_program(program).unwrap_or_else(|report| {
            panic!(
                "typed-Deep ingress must preserve an unconstrained polymorphic value through \
                 {carrier}: {:?}",
                report.errors
            )
        });
    }
}

#[test]
fn body_only_type_binder_is_generalized_and_remains_rigid() {
    let polymorphic = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_type_binder_is_generalized_and_remains_rigid_708.json"
    ));
    check_ir_program(&polymorphic)
        .expect("normalized IR ingress must generalize the body-only type binder");
    check_typed_program(&polymorphic)
        .expect("typed ingress must generalize the body-only type binder");

    let narrowed = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_type_binder_is_generalized_and_remains_rigid_717.json"
    ));
    let ir = diagnostics(check_ir_program(&narrowed));
    let typed = diagnostics(check_typed_program(&narrowed));
    assert_eq!(typed, ir, "body-only binder rigidity ingress parity");
    assert!(
        ir.iter().any(|diagnostic| {
            diagnostic.message.contains("declared type parameter `p`")
                && diagnostic.message.contains("[04-INF-6]")
        }),
        "a body-only authored binder must remain rigid: {ir:?}"
    );
}

fn assert_body_only_shape_binder_pin_rejects(
    source: &str,
    declaration: &str,
    binder: &str,
    role: &str,
) {
    let legacy_program = parse_core_fixture(source);
    let canonical_source = print_canonical_flat(&legacy_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared canonical Deep must parse");

    for (carrier, program) in [
        ("Surf-desugared Deep", legacy_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        let ir = diagnostics(check_ir_program(program));
        let typed = diagnostics(check_typed_program(program));
        assert_eq!(
            typed, ir,
            "{carrier} checker ingresses must agree for body-only {role} binder `{binder}`"
        );
        assert!(
            ir.iter().any(|diagnostic| {
                diagnostic.message.contains(declaration)
                    && diagnostic.message.contains(&format!("`{binder}`"))
                    && diagnostic.message.contains("[04-INF-6]")
            }),
            "{carrier} must reject concrete narrowing of body-only {role} binder `{binder}`: \
             {ir:?}\n{canonical_source}"
        );
    }
}

#[test]
fn body_only_dimension_and_rank_binders_cannot_be_pinned_to_concrete_shapes() {
    for (source, binder, role) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_dimension_and_rank_binders_cannot_be_pinned_to_concrete_shapes_767.json"
            ),
            "n",
            "dimension",
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_dimension_and_rank_binders_cannot_be_pinned_to_concrete_shapes_775.json"
            ),
            "r",
            "rank",
        ),
    ] {
        let declaration = if role == "dimension" {
            "narrowed_dimension"
        } else {
            "narrowed_rank"
        };
        assert_body_only_shape_binder_pin_rejects(source, declaration, binder, role);
    }
}

#[test]
fn recursive_body_only_dimension_and_rank_binders_remain_rigid() {
    for (source, declaration, binder, role) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/recursive_body_only_dimension_and_rank_binders_remain_rigid_796.json"
            ),
            "recursive_dimension",
            "n",
            "dimension",
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/recursive_body_only_dimension_and_rank_binders_remain_rigid_805.json"
            ),
            "recursive_rank",
            "r",
            "rank",
        ),
    ] {
        assert_body_only_shape_binder_pin_rejects(source, declaration, binder, role);
    }
}

fn assert_body_only_dimension_cannot_collapse_into_return_only(source: &str, declaration: &str) {
    let legacy_program = parse_core_fixture(source);
    let canonical_source = print_canonical_flat(&legacy_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared canonical Deep must parse");

    for (carrier, program) in [
        ("Surf-desugared Deep", legacy_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        let ir = diagnostics(check_ir_program(program));
        let typed = diagnostics(check_typed_program(program));
        assert_eq!(
            typed, ir,
            "{carrier} checker ingresses must agree for `{declaration}`"
        );
        assert!(
            ir.iter().any(|diagnostic| {
                diagnostic.message.contains(declaration)
                    && diagnostic.message.contains("`n`")
                    && diagnostic.message.contains("`m`")
                    && diagnostic.message.contains("[04-INF-6]")
            }),
            "{carrier} must reject a rigid body-only `m` collapsed into return-only `n`: \
             {ir:?}\n{canonical_source}"
        );
    }
}

#[test]
fn body_only_dimension_cannot_collapse_into_a_distinct_return_only_binder() {
    for (source, declaration) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_dimension_cannot_collapse_into_a_distinct_return_only_binder_851.json"
            ),
            "mixed_return_only",
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_dimension_cannot_collapse_into_a_distinct_return_only_binder_858.json"
            ),
            "recursive_mixed_return_only",
        ),
    ] {
        assert_body_only_dimension_cannot_collapse_into_return_only(source, declaration);
    }
}

#[test]
fn distinct_return_only_dimensions_may_share_one_body_inferred_output() {
    let source = include_str!(
        "fixtures/core/tests/issue_1854_explicit_type_binders/distinct_return_only_dimensions_may_share_one_body_inferred_output_874.json"
    );
    let legacy_program = parse_core_fixture(source);
    let canonical_source = print_canonical_flat(&legacy_program);
    let canonical_program =
        parse_deep(&canonical_source).expect("desugared canonical Deep must parse");

    for (carrier, program) in [
        ("Surf-desugared Deep", legacy_program.as_slice()),
        ("canonical Deep", canonical_program.as_slice()),
    ] {
        check_ir_program(program).unwrap_or_else(|report| {
            panic!(
                "normalized IR ingress must preserve return-only output inference through \
                 {carrier}: {:?}",
                report.errors
            )
        });
        check_typed_program(program).unwrap_or_else(|report| {
            panic!(
                "typed ingress must preserve return-only output inference through {carrier}: {:?}",
                report.errors
            )
        });
    }
}

#[test]
fn body_only_surf_annotations_cannot_introduce_undeclared_binders() {
    for (source, name, needle) in [
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_surf_annotations_cannot_introduce_undeclared_binders_908.json"
            ),
            "p",
            "unknown primitive type",
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_surf_annotations_cannot_introduce_undeclared_binders_913.json"
            ),
            "n",
            "undeclared dimension variable",
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_surf_annotations_cannot_introduce_undeclared_binders_918.json"
            ),
            "p",
            "unknown primitive type",
        ),
        (
            include_str!(
                "fixtures/core/tests/issue_1854_explicit_type_binders/body_only_surf_annotations_cannot_introduce_undeclared_binders_923.json"
            ),
            "r",
            "undeclared rank variable",
        ),
    ] {
        let program = parse_core_fixture(source);
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "undeclared Surf body-binder parity for `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong undeclared Surf body-binder diagnostic: {ir:?}"
        );
        assert_eq!(
            ir[0].kind, "TypeMismatch",
            "the shared type resolver must own undeclared body binders: {ir:?}"
        );
        if source.contains("tensor[2, p]") {
            assert!(
                ir[0].span_offset.is_some()
                    && ir[0].span_id.is_some()
                    && ir[0]
                        .suggestions
                        .iter()
                        .any(|suggestion| suggestion.contains("declare `p`")),
                "the undeclared precision owner must explain the explicit binder repair: {ir:?}"
            );
        }
    }
}

#[test]
fn canonical_deep_body_only_binders_fill_a_wildcard_result_at_both_ingresses() {
    for (binder, annotation) in [
        ("p", "(t-var {} p)"),
        ("n", "(t-tensor {} (d-var {} n) (t-prim {} f32))"),
        ("r", "(t-tensor {} (d-rank {} r) (t-prim {} f32))"),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} maker ({binder}) (t-fn {{}} (t-var {{}} _)))
             (def {{}}
               maker
               (fn {{}}
                 (params {{}})
                 (fn {{}}
                   (params {{}} (x {{type: {annotation}}}))
                   (var {{}} x))))"
        ))
        .expect("canonical Deep fixture must parse");
        check_ir_program(&program).unwrap_or_else(|report| {
            panic!(
                "normalized IR ingress must let body-only `{binder}` fill the wildcard: {:?}",
                report.errors
            )
        });
        check_typed_program(&program).unwrap_or_else(|report| {
            panic!(
                "typed ingress must let body-only `{binder}` fill the wildcard: {:?}",
                report.errors
            )
        });
    }
}

#[test]
fn body_annotations_cannot_introduce_undeclared_binders_at_either_ingress() {
    for (name, annotation, needle) in [
        ("p", "(t-var {} p)", "undeclared type variable"),
        (
            "p",
            "(t-tensor {} (d-lit {} 2) (t-prim {} p))",
            "unknown primitive type",
        ),
        (
            "n",
            "(t-tensor {} (d-var {} n) (t-prim {} f32))",
            "undeclared dimension variable",
        ),
        (
            "r",
            "(t-tensor {} (d-rank {} r) (t-prim {} f32))",
            "undeclared rank variable",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} maker (t-fn {{}} (t-var {{}} _)))
             (def {{}}
               maker
               (fn {{}}
                 (params {{}})
                 (fn {{}}
                   (params {{}} (x {{type: {annotation}}}))
                   (var {{}} x))))"
        ))
        .expect("canonical Deep fixture must parse");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(
            typed, ir,
            "undeclared body-binder ingress parity for `{name}`"
        );
        assert_eq!(ir.len(), 1, "`{name}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong undeclared body-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn explicit_deep_t_var_nodes_remain_valid_binders() {
    let program = parse_deep(
        "(defsig {} ident (float32)
           (t-fn {} (t-var {} float32) (t-var {} float32)))
         (def {} ident
           (fn {} (params {} (x {type: (t-var {} float32)})) (var {} x)))",
    )
    .expect("Deep fixture must parse");
    check_ir_program(&program).expect("an authored Deep `t-var` is explicit");
    check_typed_program(&program).expect("typed ingress must agree");
}

#[test]
fn forbidden_dtype_names_reject_once_at_the_deep_binder_list() {
    for (name, ty, body) in [
        (
            "f32",
            "(t-fn {} (t-tensor {} (d-var {} f32) (t-prim {} i32)) (t-prim {} i32))",
            "(fn {} (params {} x) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "f8e4m3",
            "(t-fn {} (t-tensor {} (d-var {} f8e4m3) (t-prim {} i32)) (t-prim {} i32))",
            "(fn {} (params {} x) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "i32",
            "(t-fn {} (t-tensor {} (d-rank {} i32) (t-prim {} f32)) (t-prim {} i32))",
            "(fn {} (params {} x) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "f8e5m2",
            "(t-fn {} (t-tensor {} (d-rank {} f8e5m2) (t-prim {} f32)) (t-prim {} i32))",
            "(fn {} (params {} x) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "bool",
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "u8",
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "int32",
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
        (
            "complex64",
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} ident ({name})
               {ty})
             (def {{}} ident {body})"
        ))
        .expect("Deep fixture must parse");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "forbidden-binder rejection parity for `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must report once: {ir:?}");
        assert!(
            ir[0].message.contains("cannot be a `defsig` binder")
                && ir[0].message.contains(&format!("`{name}`")),
            "wrong forbidden-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn explicit_deep_unknown_primitives_reject_like_desugared_surf() {
    for (name, nearest) in UNKNOWN_DTYPES {
        let program = parse_deep(&format!(
            "(defsig {{}} ident
               (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))
             (def {{}} ident
               (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("Deep fixture must parse");
        assert_both_ingresses_reject_once(&program, name, nearest);
    }
    for name in RESERVED_DTYPES {
        let program = parse_deep(&format!(
            "(defsig {{}} ident
               (t-fn {{}} (t-prim {{}} {name}) (t-prim {{}} {name})))
             (def {{}} ident
               (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("Deep fixture must parse");
        assert_both_ingresses_reject_once(&program, name, None);
    }
}

#[test]
fn distinct_unknown_deep_scalar_names_keep_distinct_diagnostics() {
    let program = parse_deep(
        "(defsig {} convert
           (t-fn {} (t-prim {} float32) (t-prim {} fp32)))
         (def {} convert (fn {} (params {} x) (var {} x)))",
    )
    .expect("Deep fixture must parse");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(
        typed, ir,
        "checker ingresses must preserve diagnostic order"
    );
    assert_eq!(
        ir.len(),
        2,
        "distinct invalid names need distinct owners: {ir:?}"
    );
    for name in ["float32", "fp32"] {
        assert_eq!(
            ir.iter()
                .filter(|diagnostic| diagnostic.message.contains(&format!("`{name}`")))
                .count(),
            1,
            "`{name}` must have exactly one diagnostic: {ir:?}"
        );
    }
}

#[test]
fn arbitrary_non_dtype_names_remain_legal_deep_binders() {
    for (ty, body) in [
        (
            "(t-fn {} (t-tensor {} (d-var {} float32) (t-prim {} f32)) \
                       (t-tensor {} (d-var {} float32) (t-prim {} f32)))",
            "(fn {} (params {} x) (var {} x))",
        ),
        (
            "(t-fn {} (t-tensor {} (d-rank {} float32) (t-prim {} f32)) \
                       (t-tensor {} (d-rank {} float32) (t-prim {} f32)))",
            "(fn {} (params {} x) (var {} x))",
        ),
        (
            "(t-fn {} (t-prim {} i32))",
            "(fn {} (params {}) (lit {type: (t-prim {} i32)} 0))",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} ident (float32) {ty})
             (def {{}} ident {body})"
        ))
        .expect("Deep fixture must parse");
        check_ir_program(&program).expect("arbitrary Deep binder must remain legal");
        check_typed_program(&program).expect("typed ingress must agree");
    }
}

#[test]
fn undeclared_deep_type_dimension_and_rank_variables_reject() {
    for (name, ty, needle) in [
        (
            "a",
            "(t-fn {} (t-var {} a) (t-prim {} f32))",
            "undeclared type variable",
        ),
        (
            "n",
            "(t-fn {} (t-tensor {} (d-var {} n) (t-prim {} f32)) (t-prim {} f32))",
            "undeclared dimension variable",
        ),
        (
            "r",
            "(t-fn {} (t-tensor {} (d-rank {} r) (t-prim {} f32)) (t-prim {} f32))",
            "undeclared rank variable",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} f {ty})
             (def {{}} f (fn {{}} (params {{}} x) (lit {{type: (t-prim {{}} f32)}} 0.0)))"
        ))
        .expect("Deep fixture must parse");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "ingress parity for `{name}`");
        assert_eq!(ir.len(), 1, "`{name}` must report once: {ir:?}");
        assert!(
            ir[0].message.contains(needle) && ir[0].message.contains(&format!("`{name}`")),
            "wrong undeclared-binder diagnostic: {ir:?}"
        );
    }
}

#[test]
fn empty_deep_binder_list_is_rejected_at_both_checker_ingresses() {
    let program = parse_deep(
        "(defsig {} f () (t-fn {} (t-prim {} f32) (t-prim {} f32)))
         (def {} f (fn {} (params {} x) (lit {type: (t-prim {} f32)} 0.0)))",
    )
    .expect("structural fixture parses before semantic validation");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(typed, ir, "empty-list rejection must have ingress parity");
    assert_eq!(ir.len(), 1, "empty binder list must have one owner: {ir:?}");
    assert!(
        ir[0].message.contains("empty list must be omitted"),
        "wrong empty-list diagnostic: {ir:?}"
    );
}

#[test]
fn malformed_and_duplicate_deep_binder_lists_reject_at_both_ingresses() {
    for (binders, needle) in [
        ("(a a)", "duplicate `defsig` binder `a`"),
        (
            "(a 1)",
            "expected a nonempty structural list of distinct symbol names",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} f {binders} (t-fn {{}} (t-var {{}} a) (t-var {{}} a)))
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("structural fixture parses before checker validation");
        let ir = diagnostics(check_ir_program(&program));
        let typed = diagnostics(check_typed_program(&program));
        assert_eq!(typed, ir, "malformed-list ingress parity for `{binders}`");
        assert_eq!(ir.len(), 1, "`{binders}` must have one owner: {ir:?}");
        assert!(
            ir[0].message.contains(needle),
            "wrong malformed-list diagnostic for `{binders}`: {ir:?}"
        );
    }
}

#[test]
fn declared_deep_type_dimension_and_rank_variables_check_clean() {
    for (binders, ty) in [
        ("(a)", "(t-fn {} (t-var {} a) (t-var {} a))"),
        (
            "(n)",
            "(t-fn {} (t-tensor {} (d-var {} n) (t-prim {} f32)) \
                      (t-tensor {} (d-var {} n) (t-prim {} f32)))",
        ),
        (
            "(r)",
            "(t-fn {} (t-tensor {} (d-rank {} r) (t-prim {} f32)) \
                      (t-tensor {} (d-rank {} r) (t-prim {} f32)))",
        ),
    ] {
        let program = parse_deep(&format!(
            "(defsig {{}} f {binders} {ty})
             (def {{}} f (fn {{}} (params {{}} x) (var {{}} x)))"
        ))
        .expect("Deep fixture must parse");
        check_ir_program(&program).expect("declared Deep binder must check");
        check_typed_program(&program).expect("typed ingress must agree");
    }
}

#[test]
fn declared_but_unused_deep_binder_is_a_vacuous_quantifier() {
    let program = parse_deep(
        "(defsig {} constant (a) (t-fn {} (t-prim {} i32)))
         (def {} constant
           (fn {} (params {}) (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("Deep fixture must parse");
    check_ir_program(&program).expect("IR ingress must accept an unused unbounded binder");
    check_typed_program(&program).expect("typed ingress must accept the same binder policy");
}

#[test]
fn declared_but_unused_bounded_deep_binder_rejects_at_both_ingresses() {
    let program = parse_deep(
        "(defsig {dtype_bounds: {a: numeric}} constant (a)
           (t-fn {} (t-prim {} i32)))
         (def {} constant
           (fn {} (params {}) (lit {type: (t-prim {} i32)} 1)))",
    )
    .expect("Deep fixture must parse");
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(
        typed, ir,
        "bounded-unused rejection must have ingress parity"
    );
    assert_eq!(
        ir.len(),
        1,
        "bounded-unused binder must report once: {ir:?}"
    );
    assert!(
        ir[0].message.contains("does not occur") && ir[0].message.contains("`a`"),
        "wrong bounded-unused diagnostic: {ir:?}"
    );
}

#[test]
fn explicit_property_binders_check_at_both_ingresses() {
    for source in [
        include_str!("fixtures/core/tests/issue_1854_explicit_type_binders/shared_1333.json"),
        include_str!("fixtures/core/tests/issue_1854_explicit_type_binders/shared_1334.json"),
    ] {
        let program = parse_core_fixture(source);
        check_ir_program(&program).expect("IR ingress must accept the property binder");
        check_typed_program(&program).expect("typed ingress must accept the property binder");
    }
}

#[test]
fn incomplete_property_binder_lists_reject_at_both_ingresses() {
    let program = parse_core_fixture(include_str!(
        "fixtures/core/tests/issue_1854_explicit_type_binders/shared_1344.json"
    ));
    let ir = diagnostics(check_ir_program(&program));
    let typed = diagnostics(check_typed_program(&program));
    assert_eq!(typed, ir, "incomplete property binders need ingress parity");
    assert_eq!(ir.len(), 1, "undeclared `n` must have one owner: {ir:?}");
    assert!(
        ir[0].message.contains("undeclared dimension variable") && ir[0].message.contains("`n`"),
        "wrong incomplete-property-binder diagnostic: {ir:?}"
    );
}

#[test]
fn an_unlisted_property_scalar_name_remains_an_unknown_primitive() {
    assert_both_ingresses_reject_once(
        &parse_core_fixture(include_str!(
            "fixtures/core/tests/issue_1854_explicit_type_binders/shared_1358.json"
        )),
        "float32",
        Some("f32"),
    );
}
