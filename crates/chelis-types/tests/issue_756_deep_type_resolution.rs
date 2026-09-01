//! Chelis-Lang/chelis#756 and chelis#731 Phase 2: Deep type expressions are
//! resolved by one fail-closed boundary. Only declared or explicitly implicit
//! binders may allocate variables; malformed/unknown types report exactly once
//! and cannot seed a reusable library context.
//!
//! Both polarities are intentional. Positive cases cover the legal binder and
//! nominal-reference paths. Negative cases cover every type/dimension form
//! whose historical fallback was a fresh variable, wildcard, or silently
//! dropped dimension.
//!
//! Spec authority: spec/03-deep-syntax.md §2.5.1;
//! spec/04-type-system.md §10 [04-TOT-2];
//! spec/design/checker_totality.md §C3.1.

mod support;

use chelis_types::errors::CheckErrorKind;
use chelis_types::types::Type;
use chelis_types::{build_type_env_from_library, check_ir_program, check_ir_with_context};

fn parse(source: &str) -> Vec<chelis_deep::Expr> {
    // Deliberately lenient: the checker owns the fail-closed diagnostic for an
    // unknown tag nested in a type position, even when strict Deep validation
    // would reject the same input earlier.
    chelis_deep::parser::parse_str(source).expect("Deep fixture must parse")
}

fn assert_accepts(source: &str, label: &str) {
    let exprs = parse(source);
    if let Err(result) = check_ir_program(&exprs) {
        let messages: Vec<_> = result.errors.iter().map(|error| &error.message).collect();
        panic!("{label}: expected a clean check, got {messages:?}");
    }
}

fn assert_surf_accepts(source: &str, label: &str) {
    let declarations = chelis_surf::parser::parse_str(source)
        .unwrap_or_else(|error| panic!("{label}: Surf fixture must parse: {error}"));
    let exprs = chelis_surf::desugar::desugar_program(&declarations);
    let fitness = chelis_types::check_program(&exprs);
    assert!(
        fitness.errors.is_empty(),
        "{label}: fitness/check_program must be clean, got {:?}",
        fitness.errors
    );
    assert!(
        fitness.score >= 0.9,
        "{label}: fitness/check_program scored {}, expected >= 0.9",
        fitness.score
    );
    if let Err(result) = check_ir_program(&exprs) {
        let messages: Vec<_> = result.errors.iter().map(|error| &error.message).collect();
        panic!("{label}: expected a clean check, got {messages:?}");
    }
}

fn assert_one_type_resolution_error(source: &str, needle: &str, label: &str) {
    let exprs = match chelis_deep::parser::parse_str(source) {
        Ok(exprs) => exprs,
        Err(error) => {
            let message = error.to_string();
            assert!(
                message.contains(needle),
                "{label}: stamped-boundary diagnostic must name `{needle}`, got {message:?}"
            );
            return;
        }
    };
    let result =
        check_ir_program(&exprs).expect_err("malformed/unknown Deep type must fail the checker");
    assert_eq!(
        result.errors.len(),
        1,
        "{label}: a nested type failure must report once without a cascade: {:?}",
        result.errors
    );
    let error = &result.errors[0];
    assert!(
        matches!(
            error.kind,
            CheckErrorKind::TypeMismatch | CheckErrorKind::MalformedForm
        ),
        "{label}: expected a type-resolution diagnostic, got {error:?}"
    );
    assert!(
        error.message.contains(needle),
        "{label}: diagnostic must name `{needle}`, got {error:?}"
    );
    assert!(
        error.span_offset.is_some() || error.span_id.is_some(),
        "{label}: the owning Deep type use site must provide a stable location: {error:?}"
    );
}

fn field_type(type_expr: &str) -> String {
    format!(
        "(deftype {{}} Holder ()\n\
           (variant {{}} Hold (field {{}} value {type_expr})))"
    )
}

#[test]
fn valid_primitives_self_and_forward_nominals_are_accepted() {
    assert_accepts(
        "(deftype {} Scalars ()
           (variant {} Scalars
             (field {} a (t-prim {} f32))
             (field {} b (t-prim {} int64))
             (field {} c (t-prim {} bool))))
         (deftype {} Node ()
           (variant {} Node (field {} next (t-adt {} Node))))
         (deftype {} First ()
           (variant {} First (field {} second (t-adt {} Second))))
         (deftype {} Second () (variant {} Second))",
        "primitive, self-recursive, and forward nominal references",
    );
}

#[test]
fn explicit_deftype_rank_binder_is_rejected() {
    assert_one_type_resolution_error(
        "(deftype {} Boxed (p n r)
           (variant {} Boxed
             (field {} value
               (t-tensor {} (d-var {} n) (d-rank {} r) (t-var {} p)))))",
        "cannot be used as a rank spread",
        "rank-kinded nominal parameter",
    );
}

#[test]
fn explicit_kinded_deftype_field_rejects_inference_hole() {
    assert_one_type_resolution_error(
        "(deftype {} Hole (a)
           (variant {} Hole (field {} value (t-var {} _))))",
        "undeclared type variable `_`",
        "explicit kinded deftype field inference hole",
    );
}

#[test]
fn explicit_kinded_typealias_body_rejects_inference_hole() {
    assert_one_type_resolution_error(
        "(typealias {} Hole (a) (t-var {} _))",
        "undeclared type variable `_`",
        "explicit kinded typealias body inference hole",
    );
}

#[test]
fn parameterized_typealias_type_and_dimension_binders_are_accepted() {
    assert_surf_accepts(
        "type Matrix[p, rows] = tensor[rows, p]\n\
         def keep[p, rows](value: Matrix[p, rows]) -> Matrix[p, rows] = value",
        "parameterized typealias type/dimension binders",
    );
}

#[test]
fn typealias_self_and_forward_nominal_headers_are_visible() {
    assert_accepts(
        "(typealias {} Recursive (a)
           (t-tuple {} (t-var {} a) (t-adt {} Recursive (t-var {} a))))
         (typealias {} First () (t-adt {} Second))
         (typealias {} Second () (t-tuple {} (t-prim {} f32)))",
        "typealias self and forward nominal references",
    );
}

#[test]
fn documented_zero_parameter_symbolic_dimension_alias_is_accepted() {
    // Exact executable example shared by packages/chelis-std/SKILL.md and
    // docs/book/src/surf-reference.md. With no `[n]` parameter, `n` is a
    // concrete symbolic dimension (`d-name`), not an undeclared `d-var`.
    assert_surf_accepts(
        "type Weights = tensor[n, f32]\n\n\
         def keep(w: Weights) -> Weights = w",
        "documented Weights alias",
    );
}

#[test]
fn implicit_defsig_binders_and_legal_metadata_hole_are_accepted() {
    assert_accepts(
        "(defsig {} polymorphic
           (t-fn {}
             (t-tensor {} (d-var {} n) (d-rank {} r) (t-var {} p))
             (t-tensor {} (d-var {} n) (d-rank {} r) (t-var {} p))))
         (def {} polymorphic
           (fn {} (params {} value) (var {} value)))
         (def {} inferred (lit {type: (t-var {} _)} 1))",
        "implicit signature binders and metadata inference hole",
    );
}

#[test]
fn enclosing_defsig_binder_is_legal_in_nested_ascription_metadata() {
    assert_accepts(
        "(defsig {} keep
           (t-fn {}
             (t-tensor {} (d-var {} n) (t-prim {} f32))
             (t-tensor {} (d-var {} n) (t-prim {} f32))))
         (def {} keep
           (fn {} (params {} x)
             (let {} (bind {} y
               (var {type: (t-tensor {} (d-var {} n) (t-prim {} f32))} x))
               (var {} y))))",
        "nested annotation using an enclosing signature binder",
    );
}

#[test]
fn annotation_pass_preserves_nested_defsig_binder_result_type() {
    let exprs = parse(
        "(defsig {} keep
           (t-fn {}
             (t-tensor {} (d-var {} n) (t-prim {} f32))
             (t-tensor {} (d-var {} n) (t-prim {} f32))))
         (def {} keep
           (fn {} (params {} x)
             (let {} (bind {} y
               (var {type: (t-tensor {} (d-var {} n) (t-prim {} f32))} x))
               (var {} y))))",
    );
    let checked = check_ir_program(&exprs).expect("program must check");
    let rendered = chelis_deep::printer::print_canonical(checked.annotated_exprs());
    assert!(
        !rendered.contains("(t-var {} _)"),
        "authoritative owner writeback must retain the tensor result instead of silently stamping an error hole:\n{rendered}"
    );
}

#[test]
fn checked_nominal_metadata_resolves_against_precollected_headers() {
    let exprs = parse(
        "(deftype {} Marker () (variant {} Marker))
         (defsig {} make (t-fn {} (t-adt {} Marker)))
         (def {} make (fn {} (params {}) (var {} Marker)))",
    );
    let checked = check_ir_program(&exprs).expect("program must check");
    let signature = checked
        .signature_inference()
        .functions
        .get("make")
        .expect("checked nominal signature must not disappear during metadata resolution");
    assert!(matches!(
        &signature.checked_signature,
        Type::Fn(args, ret)
            if args.is_empty() && matches!(ret.as_ref(), Type::Adt(name, args) if name == "Marker" && args.is_empty())
    ));
}

#[test]
fn checked_metadata_resolves_nominals_imported_from_library_context() {
    let library = parse("(deftype {} LibMarker () (variant {} LibMarker))");
    let context = build_type_env_from_library(&library).expect("library context must check");
    let new_exprs = parse(
        "(defsig {} make (t-fn {} (t-adt {} LibMarker)))
         (def {} make (fn {} (params {}) (var {} LibMarker)))",
    );
    let checked = check_ir_with_context(&context, &new_exprs).expect("new code must check");
    let signature = checked
        .signature_inference()
        .functions
        .get("make")
        .expect("an imported nominal must not make checked signature metadata disappear");
    let rendered = chelis_deep::printer::print_canonical(checked.annotated_exprs());
    assert!(
        matches!(
            &signature.checked_signature,
            Type::Fn(args, ret)
                if args.is_empty() && matches!(ret.as_ref(), Type::Adt(name, args) if name == "LibMarker" && args.is_empty())
        ),
        "unexpected imported-nominal signature: {:?}\n{rendered}",
        signature.checked_signature
    );
}

#[test]
fn checked_metadata_resolves_aliases_imported_from_library_context() {
    let library = parse(
        "(deftype {} LibMarker () (variant {} LibMarker))
         (typealias {} LibAlias () (t-adt {} LibMarker))",
    );
    let context = build_type_env_from_library(&library).expect("library context must check");
    let new_exprs = parse(
        "(defsig {} make (t-fn {} (t-adt {} LibAlias)))
         (def {} make (fn {} (params {}) (var {} LibMarker)))",
    );
    check_ir_with_context(&context, &new_exprs)
        .expect("a validated alias header must survive stacked context construction");
}

#[test]
fn unknown_primitive_adt_and_wrong_nominal_arity_report_once() {
    let cases = [
        (
            field_type("(t-prim {} madeup)"),
            "madeup",
            "unknown primitive",
        ),
        (field_type("(t-adt {} Missing)"), "Missing", "unknown ADT"),
        (field_type("(t-adt {} Option)"), "Option", "wrong ADT arity"),
    ];

    for (source, needle, label) in cases {
        assert_one_type_resolution_error(&source, needle, label);
    }
}

#[test]
fn undeclared_type_dimension_and_rank_variables_report_once() {
    let cases = [
        (field_type("(t-var {} a)"), "a", "undeclared type variable"),
        (
            field_type("(t-tensor {} (d-var {} n) (t-prim {} f32))"),
            "n",
            "undeclared dimension variable",
        ),
        (
            field_type("(t-tensor {} (d-rank {} r) (t-prim {} f32))"),
            "r",
            "undeclared rank variable",
        ),
    ];

    for (source, needle, label) in cases {
        assert_one_type_resolution_error(&source, needle, label);
    }
}

#[test]
fn invalid_alias_body_does_not_make_its_declared_header_unknown_downstream() {
    let source = "(typealias {} Broken ()
           (t-tensor {} (d-var {} missing) (t-prim {} f32)))
         (defsig {} keep (t-fn {} (t-adt {} Broken) (t-adt {} Broken)))
         (def {} keep (fn {} (params {} x) (var {} x)))";
    assert_one_type_resolution_error(
        source,
        "undeclared dimension variable `missing` in typealias body",
        "invalid alias with a downstream annotation",
    );
}

#[test]
fn unknown_nominal_in_alias_body_reports_once_without_downstream_cascade() {
    let source = "(typealias {} Broken () (t-adt {} Missing))
         (defsig {} keep (t-fn {} (t-adt {} Broken) (t-adt {} Broken)))
         (def {} keep (fn {} (params {} x) (var {} x)))";
    assert_one_type_resolution_error(
        source,
        "unknown nominal type `Missing` in typealias body",
        "unknown nominal in alias body",
    );
}

#[test]
fn malformed_nested_type_and_dimension_forms_report_once_without_drops() {
    let cases = [
        (field_type("madeup"), "madeup", "bare field type"),
        (
            field_type("(t-prim {} f32 extra)"),
            "t-prim",
            "t-prim extra child",
        ),
        (field_type("(t-adt {})"), "t-adt", "t-adt missing name"),
        (field_type("(t-var {})"), "t-var", "t-var missing name"),
        (field_type("(t-fn {})"), "t-fn", "empty function type"),
        (field_type("(t-ref {})"), "t-ref", "empty reference type"),
        (
            field_type("(t-ref {} (t-prim {} f32) (t-prim {} f32))"),
            "t-ref",
            "reference type with extra child",
        ),
        (field_type("(t-tensor {})"), "t-tensor", "empty tensor type"),
        (
            field_type("(t-tensor {} (t-tuple {}))"),
            "tensor precision",
            "non-precision tensor tail",
        ),
        (
            field_type("(t-tensor {} (d-lit {} nope) (t-prim {} f32))"),
            "d-lit",
            "malformed dimension child",
        ),
        (
            field_type("(t-tensor {} (d-name {}) (t-prim {} f32))"),
            "d-name",
            "dimension name missing child",
        ),
        (
            field_type("(t-tensor {} (d-var {}) (t-prim {} f32))"),
            "d-var",
            "dimension variable missing child",
        ),
        (
            field_type("(t-tensor {} (d-rank {}) (t-prim {} f32))"),
            "d-rank",
            "rank variable missing child",
        ),
        (
            field_type("(t-tensor {} (unknown-dim {}) (t-prim {} f32))"),
            "unknown-dim",
            "unknown dimension tag",
        ),
        (
            field_type("(t-unit {} extra)"),
            "t-unit",
            "unit type with extra child",
        ),
        (
            field_type("(unknown-type {})"),
            "unknown-type",
            "unknown nested type tag",
        ),
    ];

    for (source, needle, label) in cases {
        assert_one_type_resolution_error(&source, needle, label);
    }
}

#[test]
fn malformed_canonical_primitive_reports_once_at_every_type_consumer() {
    let cases = [
        (field_type("(t-prim {} f32 extra)"), "deftype field"),
        (
            "(typealias {} Bad () (t-prim {} f32 extra))".to_string(),
            "typealias body",
        ),
        (
            "(defsig {} bad (t-fn {} (t-prim {} f32 extra)))".to_string(),
            "defsig",
        ),
        (
            "(def {} bad (lit {type: (t-prim {} f32 extra)} 1.0))".to_string(),
            "type annotation",
        ),
        (
            "(def {} bad
               (let {} (bind {} value
                 (lit {type: (t-prim {} f32 extra)} 1.0))
                 (var {} value)))"
                .to_string(),
            "let-binding ascription",
        ),
        (
            "(def {} x (lit {type: (t-prim {} f32)} 1.0))
             (def {} bad (cast {} (var {} x) (t-prim {} f32 extra)))"
                .to_string(),
            "cast target",
        ),
    ];

    for (source, label) in cases {
        assert_one_type_resolution_error(&source, "t-prim", label);
    }
}

#[test]
fn independent_rhs_and_malformed_let_ascription_each_report_once() {
    let exprs = support::parse_unchecked_legacy(
        "(def {} bad
           (let {} (bind {} value
             (var {type: (t-prim {} f32 extra)} missing))
             (lit {} 0)))",
    );
    let result = check_ir_program(&exprs)
        .expect_err("the independent RHS and ascription roots must both fail the checker");
    assert_eq!(
        result.errors.len(),
        2,
        "independent roots must each report once without suppressing or spraying: {:?}",
        result.errors
    );
    assert_eq!(
        result
            .errors
            .iter()
            .filter(|error| error.message.contains("missing"))
            .count(),
        1,
        "the RHS root must report exactly once: {:?}",
        result.errors
    );
    assert_eq!(
        result
            .errors
            .iter()
            .filter(|error| error.message.contains("t-prim"))
            .count(),
        1,
        "the malformed ascription root must report exactly once: {:?}",
        result.errors
    );
}

#[test]
fn prebound_failure_is_owned_by_its_exact_duplicate_name_declaration() {
    let exprs = support::parse_unchecked_legacy(
        "(def {} duplicate (var {} missing))
         (def {} duplicate (lit {type: (t-prim {} f32 extra)} 1.0))",
    );
    let result = check_ir_program(&exprs)
        .expect_err("duplicate declarations with independent roots must fail the checker");
    assert_eq!(
        result.errors.len(),
        3,
        "the duplicate plus both independent roots must each report once: {:?}",
        result.errors
    );
    for needle in ["duplicate definition", "missing", "t-prim"] {
        assert_eq!(
            result
                .errors
                .iter()
                .filter(|error| error.message.contains(needle))
                .count(),
            1,
            "`{needle}` must belong to exactly one declaration/root: {:?}",
            result.errors
        );
    }
}

#[test]
fn undeclared_variable_in_closed_top_level_metadata_reports_once() {
    assert_one_type_resolution_error(
        "(def {} x (lit {type: (t-var {} rogue)} 1))",
        "rogue",
        "closed top-level annotation",
    );
}

#[test]
fn invalid_declaration_cannot_seed_a_reusable_context() {
    let exprs = parse(&field_type("(t-adt {} Missing)"));
    let result = build_type_env_from_library(&exprs)
        .expect_err("an invalid declaration must not produce a cacheable TypeEnv");
    assert_eq!(
        result.errors.len(),
        1,
        "context build must retain the one witnessed resolution error: {:?}",
        result.errors
    );
    assert!(result.errors[0].message.contains("Missing"));
}

#[test]
fn nested_failure_propagates_through_parent_type_without_error_spray() {
    assert_one_type_resolution_error(
        &field_type("(t-ref {} (t-adt {} Missing))"),
        "Missing",
        "nested nominal failure",
    );
}
