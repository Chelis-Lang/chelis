//! chelis#1134 / [04-INF-4]: top-level eager-value scope is identical at the
//! stamped typed and serialized-IR checker ingresses.

use chelis_deep::{parse_and_stamp, parse_and_stamp_file};

use chelis_types::{
    TypeEnv, build_type_env_from_library, check_ir_program, check_ir_with_context,
    check_typed_program,
};

type Diagnostics = Vec<(String, String)>;

fn deep_program(source: &str) -> Vec<chelis_deep::Expr> {
    parse_and_stamp(source).expect("Deep fixture must parse and stamp")
}

/// Declaration-level Deep parse. `parse_and_stamp` stamps expression forms;
/// a `(module {} ...)` wrapper is only valid at declaration position, so the
/// matrix's wrapped rows need this entry point.
fn deep_file_program(source: &str) -> Vec<chelis_deep::Expr> {
    parse_and_stamp_file(source).expect("Deep file fixture must parse and stamp")
}

fn legacy_program(source: &str) -> Vec<chelis_deep::Expr> {
    serde_json::from_str(source).expect("lossless legacy fixture")
}

fn diagnostics(program: &[chelis_deep::Expr]) -> (Diagnostics, Diagnostics) {
    let summarize = |result: Result<_, chelis_types::InferResult>| match result {
        Ok(_) => Vec::new(),
        Err(result) => result
            .errors
            .iter()
            .map(|error| {
                (
                    error.kind.diagnostic_name().to_string(),
                    error.message.clone(),
                )
            })
            .collect(),
    };
    (
        summarize(check_ir_program(
            program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )),
        summarize(check_typed_program(
            program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )),
    )
}

fn assert_accepts_at_both_ingresses(program: &[chelis_deep::Expr], label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
    assert!(ir.is_empty(), "{label}: expected acceptance, got {ir:#?}");
}

fn assert_rejects_identically(program: &[chelis_deep::Expr], expected_kind: &str, label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
    assert!(
        ir.iter().any(|(kind, _)| kind == expected_kind),
        "{label}: expected {expected_kind}, got {ir:#?}"
    );
}

fn assert_only_cycle_at_both_ingresses(program: &[chelis_deep::Expr], label: &str) {
    let (ir, typed) = diagnostics(program);
    assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
    assert!(!ir.is_empty(), "{label}: expected CycleDetected");
    assert!(
        ir.iter().all(|(kind, _)| kind == "CycleDetected"),
        "{label}: expected only CycleDetected, got {ir:#?}"
    );
}

#[test]
fn defsig_less_forward_value_reference_rejects_at_both_ingresses() {
    let program = deep_program(
        "(def {} use_base (var {} base))\n\n\
         (def {} base (lit {type: (t-prim {} i32)} 7))\n",
    );
    assert_rejects_identically(&program, "UnboundVariable", "defsig-less forward value");
}

#[test]
fn backward_value_reference_remains_legal_at_both_ingresses() {
    let program = deep_program(
        "(def {} base (lit {type: (t-prim {} i32)} 7))\n\n\
         (def {} use_base (var {} base))\n",
    );
    assert_accepts_at_both_ingresses(&program, "backward value");
}

#[test]
fn module_wrapped_forward_generic_helper_has_the_same_verdict() {
    let program = legacy_program(include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/module_wrapped_forward_generic_helper_has_the_same_verdict_1.json"
    ));
    assert_accepts_at_both_ingresses(&program, "module generic forward helper");
}

#[test]
fn ascribed_external_input_self_reference_is_legal_at_both_ingresses() {
    let program = legacy_program(include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/ascribed_external_input_self_reference_is_legal_at_both_ingresses_1.json"
    ));
    assert_accepts_at_both_ingresses(&program, "external input self-reference");
}

#[test]
fn declaration_typed_self_reference_is_an_external_input_at_both_ingresses() {
    let program = legacy_program(include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/declaration_typed_self_reference_is_an_external_input_at_both_ingresses_1.json"
    ));
    assert_accepts_at_both_ingresses(&program, "declaration-typed external input");
}

#[test]
fn bare_self_reference_is_not_an_external_input_at_either_ingress() {
    let program = legacy_program(include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/bare_self_reference_is_not_an_external_input_at_either_ingress_1.json"
    ));
    assert_rejects_identically(&program, "CycleDetected", "bare self-reference");
}

#[test]
fn type_stamped_deep_self_reference_is_an_explicit_external_input() {
    let program = deep_program("(def {} x (var {type: (t-prim {} i32)} x))\n");
    assert_accepts_at_both_ingresses(&program, "typed Deep external input");
}

/// An external input whose declared type does not resolve must report the
/// same diagnostics at both ingresses. The serialized-IR ingress resolves the
/// body stamp in its prebind pass and must not resolve it a second time when
/// the declaration's own external-input binding is installed.
#[test]
fn a_malformed_external_input_type_reports_identically_at_both_ingresses() {
    for (label, source) in [
        (
            "ascribed",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_malformed_external_input_type_reports_identically_at_both_ingresses_1.json"
            ),
        ),
        (
            "declaration-typed",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_malformed_external_input_type_reports_identically_at_both_ingresses_2.json"
            ),
        ),
    ] {
        let program = legacy_program(source);
        let (ir, typed) = diagnostics(&program);
        assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
        assert!(!ir.is_empty(), "{label}: an unknown type must be rejected");
        assert_eq!(
            ir.len(),
            ir.iter().collect::<std::collections::BTreeSet<_>>().len(),
            "{label}: a diagnostic must be reported once: {ir:#?}"
        );
    }
}

/// chelis#1487 / [04-INF-7]: higher-order spelling does not weaken the
/// complete eager-reference closure. Returning a lambda and immediately
/// applying it, or storing it in a top-level function binding before a later
/// eager application, must both close the value cycle.
#[test]
fn higher_order_lambda_cycles_reject_identically_at_both_ingresses() {
    for (label, source) in [
        (
            "annotated helper-returned lambda",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/higher_order_lambda_cycles_reject_identically_at_both_ingresses_1.json"
            ),
        ),
        (
            "unannotated helper-returned lambda",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/higher_order_lambda_cycles_reject_identically_at_both_ingresses_2.json"
            ),
        ),
        (
            "annotated stored closure applied by later value",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/higher_order_lambda_cycles_reject_identically_at_both_ingresses_3.json"
            ),
        ),
        (
            "unannotated stored closure applied by later value",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/higher_order_lambda_cycles_reject_identically_at_both_ingresses_4.json"
            ),
        ),
    ] {
        assert_only_cycle_at_both_ingresses(&legacy_program(source), label);
    }
}

/// The top-level-lambda/function classification remains the negative control:
/// merely storing a lambda is lazy, and following it to an earlier eager value
/// through a callee cannot manufacture an initialization cycle.
#[test]
fn stored_top_level_lambdas_and_earlier_reads_remain_legal() {
    for (label, source) in [
        (
            "stored top-level lambda",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/stored_top_level_lambdas_and_earlier_reads_remain_legal_1.json"
            ),
        ),
        (
            "lambda reaches earlier eager value",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/stored_top_level_lambdas_and_earlier_reads_remain_legal_2.json"
            ),
        ),
    ] {
        assert_accepts_at_both_ingresses(&legacy_program(source), label);
    }
}

/// chelis#1485 / [04-INF-7]: every full-reference component containing an
/// eager value is rejected as one initialization cycle at both ingresses.
///
/// The three spellings exercise a complete header, a lambda-mediated read,
/// and a `defsig`-less reader. Their header shapes may change scheduling edges,
/// but must not change the component verdict. In particular, provisional
/// bindings used while co-inferring the component must prevent the historical
/// `UnboundVariable` split without suppressing `CycleDetected`.
#[test]
fn a_value_naming_a_function_that_reads_it_back_is_a_recorded_stall() {
    for (label, source) in [
        (
            "signed reader",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_value_naming_a_function_that_reads_it_back_is_a_recorded_stall_1.json"
            ),
        ),
        (
            "lambda naming a signed reader",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_value_naming_a_function_that_reads_it_back_is_a_recorded_stall_2.json"
            ),
        ),
        (
            "defsig-less reader",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_value_naming_a_function_that_reads_it_back_is_a_recorded_stall_3.json"
            ),
        ),
    ] {
        let program = legacy_program(source);
        let (ir, typed) = diagnostics(&program);
        assert_eq!(ir, typed, "{label}: ingress diagnostics diverged");
        assert!(
            ir.iter().any(|(kind, _)| kind == "CycleDetected"),
            "{label}: expected CycleDetected, got {ir:#?}"
        );
        assert!(
            ir.iter().all(|(kind, _)| kind == "CycleDetected"),
            "{label}: component co-inference leaked a non-cycle diagnostic: {ir:#?}"
        );
    }
    let stamped = deep_file_program(
        "(module {} MirrorEscapeStamped\n  \
           (defsig {} anchor (t-fn {} (t-prim {} i32)))\n  \
           (def {} anchor (fn {} (params {}) (lit {type: (t-prim {} i32)} 1)))\n  \
           (def {} carried (app {type: (t-fn {} (t-prim {} i32) (t-prim {} i32))} (var {} wrap) (var {} f)))\n  \
           (def {} wrap (fn {} (params {} g) (var {} g)))\n  \
           (defsig {} f (t-fn {} (t-prim {} i32) (t-prim {} i32)))\n  \
           (def {} f (fn {} (params {} (n {type: (t-prim {} i32)}))\n    \
             (if {} (app {} (var {} lte) (var {} n) (lit {type: (t-prim {} i32)} 0))\n      \
               (lit {type: (t-prim {} i32)} 0)\n      \
               (app {} (var {} carried) (app {} (var {} sub) (var {} n) (lit {type: (t-prim {} i32)} 1)))))))\n",
    );
    let (ir, typed) = diagnostics(&stamped);
    assert_eq!(ir, typed, "stamped reader: ingress diagnostics diverged");
    assert!(
        ir.iter().any(|(kind, _)| kind == "CycleDetected"),
        "stamped reader: expected CycleDetected, got {ir:#?}"
    );
    assert!(
        ir.iter().all(|(kind, _)| kind == "CycleDetected"),
        "stamped reader: component co-inference leaked a non-cycle diagnostic: {ir:#?}"
    );
}

/// Assemble one unit from ordered declarations, with or without the module
/// wrapper that gives the planner a region.
fn layout_source(module: Option<&str>, declarations: &[&str]) -> String {
    use chelis_deep::{Atom, DeepTag, Expr, List, Metadata, Span};
    let mut program: Vec<Expr> = declarations
        .iter()
        .flat_map(|source| legacy_program(source))
        .collect();
    if let Some(name) = module {
        let span = Span { offset: 0, len: 0 };
        let mut elements = vec![
            Expr::Atom(Atom::Tag(DeepTag::Module), span),
            Expr::Map(Metadata::default(), span),
            Expr::Atom(Atom::Name(name.into()), span),
        ];
        elements.append(&mut program);
        program.push(Expr::List(List { elements }, span));
    }
    serde_json::to_string(&program).expect("serializable legacy declarations")
}

/// A partial or generic header must not be instantiated before its body
/// narrows it (chelis#1486). `def f(n: i32) = ...` synthesizes a `defsig`
/// with a wildcard result, which [04-INF-5] makes an inference hole whose type
/// is whatever the body determines; `def f[a](x: a) -> a` declares an authored
/// binder, which [04-INF-6] makes rigid. The two atoms close the defect at
/// opposite ends: the hole edge defers the READER past the body, and the rigid
/// check rejects the DECLARATION.
///
/// The three layouts matter because the mirror edge that used to defer the
/// reader was bounded to the planner's region. Below the hoist floor and in a
/// bare unit it never applied, so the same spelling was accepted or rejected
/// by position alone. The hole edge carries no such bound, so every layout now
/// rejects, identically at both ingresses.
///
/// Regression test. Before this change the two below-floor layouts and the two
/// bare layouts of the partial program were ACCEPTED, and the generic program
/// rejected at the reader rather than at the declaration.
#[test]
fn a_partial_or_generic_header_is_not_instantiated_before_its_body_narrows_it() {
    let partial_reader = include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/a_partial_or_generic_header_is_not_instantiated_before_its_body_narrows_it_1.json"
    );
    let partial_fn = include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/a_partial_or_generic_header_is_not_instantiated_before_its_body_narrows_it_2.json"
    );
    let generic_reader = include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/a_partial_or_generic_header_is_not_instantiated_before_its_body_narrows_it_3.json"
    );
    // [04-INF-6]: the body pins the authored `a` to `i32`, so `f` itself is
    // the rejection and the reader never gets to matter.
    let generic_fn = include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/a_partial_or_generic_header_is_not_instantiated_before_its_body_narrows_it_4.json"
    );
    let anchor = include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/a_partial_or_generic_header_is_not_instantiated_before_its_body_narrows_it_5.json"
    );
    let carried = include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/a_partial_or_generic_header_is_not_instantiated_before_its_body_narrows_it_6.json"
    );

    for (program, reader, function) in [
        ("PartialHeader", partial_reader, partial_fn),
        ("GenericHeader", generic_reader, generic_fn),
    ] {
        for (layout, declarations) in [
            // Today's layout: the reader sits at or after the hoist floor.
            ("after the floor", vec![anchor, reader, carried, function]),
            // Below the floor: no mirror edge ever reached this reader.
            ("below the floor", vec![reader, carried, anchor, function]),
        ] {
            let source = layout_source(Some(program), &declarations);
            assert_rejects_identically(
                &legacy_program(&source),
                "TypeMismatch",
                &format!("{program} {layout}\n{source}"),
            );
        }
        // A bare unit has no planner region at all.
        let source = layout_source(None, &[reader, carried, function]);
        assert_rejects_identically(
            &legacy_program(&source),
            "TypeMismatch",
            &format!("{program} bare unit\n{source}"),
        );
    }
}

/// [04-INF-6]: an authored type binder is rigid in the body, so a body that
/// pins it to a concrete type or collapses it onto a sibling binder is a type
/// error AT THE DECLARATION. No reader is present in any of these programs:
/// the rejection is a property of the declaration alone, which is what
/// separates this atom from the scheduling half of chelis#1486.
///
/// Both declaration spellings of an authored binder are covered: an inline
/// `def f[a](..)` list and a standalone `sig f[a]: ...` list, which the
/// resolver records identically.
///
/// Regression test. Every row was ACCEPTED before this change, and the first
/// two compiled a body typed at `i32` behind a signature promising `forall
/// a. (a) -> a`.
#[test]
fn a_body_that_narrows_an_authored_type_binder_rejects_at_the_declaration() {
    for (label, source) in [
        (
            "explicit binder pinned to i32",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_body_that_narrows_an_authored_type_binder_rejects_at_the_declaration_1.json"
            ),
        ),
        (
            "standalone sig binder pinned to i32",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_body_that_narrows_an_authored_type_binder_rejects_at_the_declaration_2.json"
            ),
        ),
        (
            "two binders collapsed onto each other",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_body_that_narrows_an_authored_type_binder_rejects_at_the_declaration_3.json"
            ),
        ),
        (
            "bounded binder pinned by an unsuffixed literal",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_body_that_narrows_an_authored_type_binder_rejects_at_the_declaration_4.json"
            ),
        ),
    ] {
        assert_rejects_identically(&legacy_program(source), "TypeMismatch", label);
    }
}

/// The failure twin of the rigid-binder rejections: a body that leaves every
/// authored binder unconstrained, or constrains it only through the §P10
/// `cast(<literal>, <binder>)` override, stays accepted.
///
/// The bounded row is the shape the ten stdlib repairs took. Without it the
/// rejection above could be satisfied by rejecting every bounded binder, which
/// would be a different and much worse rule.
///
/// Disposition lock. Every row was green before this change too; the job is to
/// bound the rejection so it cannot grow into the polymorphic bodies the
/// language is for.
#[test]
fn a_body_that_keeps_its_authored_type_binders_polymorphic_stays_accepted() {
    for (label, source) in [
        (
            "identity with an explicit result",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_body_that_keeps_its_authored_type_binders_polymorphic_stays_accepted_1.json"
            ),
        ),
        (
            "identity whose result slot is a hole",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_body_that_keeps_its_authored_type_binders_polymorphic_stays_accepted_2.json"
            ),
        ),
        (
            "bounded binder written with the cast override",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_body_that_keeps_its_authored_type_binders_polymorphic_stays_accepted_3.json"
            ),
        ),
        (
            "two binders kept distinct",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_body_that_keeps_its_authored_type_binders_polymorphic_stays_accepted_4.json"
            ),
        ),
    ] {
        assert_accepts_at_both_ingresses(&legacy_program(source), label);
    }
}

#[test]
fn a_later_external_input_is_not_visible_to_an_earlier_declaration() {
    for (label, source) in [
        (
            "later ascribed external input",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_later_external_input_is_not_visible_to_an_earlier_declaration_1.json"
            ),
        ),
        (
            "later declaration-typed external input",
            include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/a_later_external_input_is_not_visible_to_an_earlier_declaration_2.json"
            ),
        ),
    ] {
        let program = legacy_program(source);
        assert_rejects_identically(&program, "UnboundVariable", label);
    }
}

#[test]
fn a_declaration_typed_value_is_not_visible_before_its_source_position() {
    let program = legacy_program(include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/a_declaration_typed_value_is_not_visible_before_its_source_position_1.json"
    ));
    assert_rejects_identically(&program, "UnboundVariable", "later declared value");
}

#[test]
fn a_separated_defsig_does_not_publish_a_future_eager_value() {
    let program = deep_program(
        "(defsig {} later (t-prim {} i32))\n\n\
         (def {} capture (var {} later))\n\n\
         (def {} later (lit {type: (t-prim {} i32)} 7))\n",
    );
    assert_rejects_identically(&program, "UnboundVariable", "separated future eager defsig");
}

#[test]
fn a_separated_defsig_preserves_a_prior_context_binding_until_its_def() {
    let library = deep_program("(def {} value (lit {type: (t-prim {} f32)} 1.0))\n");
    let context = build_type_env_from_library(
        &library,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect("library context checks cleanly");
    let program = deep_program(
        "(defsig {} value (t-prim {} i32))\n\n\
         (defsig {} capture (t-prim {} f32))\n\n\
         (def {} capture (var {} value))\n\n\
         (def {} value (lit {type: (t-prim {} i32)} 7))\n",
    );
    check_ir_with_context(
        &context,
        &program,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect("the prior f32 binding must remain visible until the new value def");
}

#[test]
fn context_check_cannot_prebind_a_later_external_input_globally() {
    for source in [
        include_str!(
            "fixtures/core/tests/issue_1134_forward_reference_parity/context_check_cannot_prebind_a_later_external_input_globally_1.json"
        ),
        include_str!(
            "fixtures/core/tests/issue_1134_forward_reference_parity/context_check_cannot_prebind_a_later_external_input_globally_2.json"
        ),
    ] {
        let program = legacy_program(source);
        let result = check_ir_with_context(
            &TypeEnv::empty(),
            &program,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        )
        .expect_err("later external input must remain unavailable in context checks");
        assert!(
            result.errors.iter().any(|error| matches!(
                error.kind,
                chelis_types::errors::CheckErrorKind::UnboundVariable { .. }
            ) && error.message.contains("x")),
            "expected x to remain unbound: {result:#?}"
        );
    }
}

#[test]
fn sequential_let_shadow_does_not_create_a_top_level_dependency() {
    let program = legacy_program(include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/sequential_let_shadow_does_not_create_a_top_level_dependency_1.json"
    ));
    assert_accepts_at_both_ingresses(&program, "sequential let shadow");
}

#[test]
fn missing_top_level_name_rejects_identically() {
    let program = deep_program("(def {} use_missing (var {} missing))\n");
    assert_rejects_identically(&program, "UnboundVariable", "missing top-level name");
}

#[test]
fn local_let_forward_reference_remains_sequential() {
    let program = deep_program(
        "(def {} local_forward\n\
           (let {}\n\
             (bind {} x (var {} y) y (lit {type: (t-prim {} i32)} 1))\n\
             (var {} x)))\n",
    );
    assert_rejects_identically(&program, "UnboundVariable", "local let forward reference");
}

#[test]
fn eager_top_level_value_cycle_rejects_as_cycle_at_both_ingresses() {
    let program = deep_program(
        "(def {} first (var {} second))\n\n\
         (def {} second (var {} first))\n",
    );
    assert_rejects_identically(&program, "CycleDetected", "eager value cycle");
}

#[test]
fn value_only_diagnostics_remain_in_source_order_without_reaching_the_scheduler() {
    // This program has no function def, so the schedule has no module
    // function to move and no reference edge to honor: it is the identity map.
    // That is deliberate: it isolates the value path. A program that reaches
    // the scheduler is NOT held to source-ordered diagnostics, because a moved
    // function or component genuinely moves its own diagnostics, and no
    // version of the checker ever ordered those by source position.
    let program = deep_program(
        "(def {} first\n\
           (let {}\n\
             (bind {} dependency (var {} later))\n\
             (app {} (var {} missing_first))))\n\n\
         (def {} later (app {} (var {} missing_later)))\n",
    );
    let (ir, typed) = diagnostics(&program);
    assert_eq!(ir, typed, "ingress diagnostics diverged");
    let messages = ir
        .iter()
        .map(|(_, message)| message.as_str())
        .collect::<Vec<_>>();
    let first = messages
        .iter()
        .position(|message| message.contains("missing_first"))
        .expect("missing_first diagnostic");
    let later = messages
        .iter()
        .position(|message| message.contains("missing_later"))
        .expect("missing_later diagnostic");
    assert!(
        first < later,
        "diagnostics must retain source order: {ir:#?}"
    );
}

// ---------------------------------------------------------------------------
// [04-INF-4] generated ordering matrix
//
// This is the verdict oracle for the atom. The named regressions above each
// pin one spelling; they cannot pin the interaction between declaration ORDER
// and the body-inference schedule. `primary_inference_schedule` reorders
// module functions by dependency, so a program's verdict depends on the
// layout as much as on the spelling, and a hand-written case only ever
// samples one layout. The matrix enumerates the layouts instead and derives
// each expected verdict from the atom: a top-level eager value is visible
// from its own `def` onward and nowhere earlier, at both ingresses,
// identically.
//
// A verdict matrix cannot see the emitted ORDER itself, which is where every
// schedule defect has lived. The order invariants are asserted directly on
// the schedule by `crates/chelis-types/src/infer/tests/schedule_invariants.rs`;
// this matrix pins that the resulting verdicts are the atom's.
//
// The `wrapped` axis matters and is easy to lose. Bare declarations carry no
// lexical module key, so no function is hoisted and only a recursive
// component straddling a value can move: an unwrapped program reaches the
// schedule only through that path.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Item {
    /// A module function that reads nothing. Its only job is to exist before
    /// the value, so the planner has an earlier function to hoist toward.
    Anchor,
    Value,
    Reader,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum ReaderKind {
    Function,
    Value,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LegacySpelling {
    Unannotated,
    DeclarationTyped,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum DeepSpelling {
    /// No sibling signature: the `def`'s own body stamp is the only type
    /// evidence, and only the serialized-IR ingress prebinds from it.
    BodyStampOnly,
    /// `(defsig ...)` immediately before the matching `(def ...)`.
    AdjacentSignature,
    /// The signature is declared at the head of the unit and the `def` stays
    /// at its layout position. A signature is metadata about a declaration,
    /// not the declaration, so it must not publish the value early.
    SeparatedSignature,
}

/// Every layout the matrix walks: all orderings of anchor/value/reader, plus
/// the two anchor-free orderings.
fn layouts() -> Vec<Vec<Item>> {
    vec![
        vec![Item::Anchor, Item::Value, Item::Reader],
        vec![Item::Anchor, Item::Reader, Item::Value],
        vec![Item::Value, Item::Anchor, Item::Reader],
        vec![Item::Value, Item::Reader, Item::Anchor],
        vec![Item::Reader, Item::Anchor, Item::Value],
        vec![Item::Reader, Item::Value, Item::Anchor],
        vec![Item::Value, Item::Reader],
        vec![Item::Reader, Item::Value],
    ]
}

/// [04-INF-4]: the reader resolves exactly when the value's `def` precedes it.
fn layout_accepts(layout: &[Item]) -> bool {
    let position = |wanted: Item| layout.iter().position(|item| *item == wanted);
    position(Item::Value) < position(Item::Reader)
}

fn legacy_source(
    layout: &[Item],
    reader: ReaderKind,
    spelling: LegacySpelling,
    wrapped: bool,
) -> String {
    let declarations: Vec<&str> = layout.iter().map(|item| match item {
        Item::Anchor => include_str!("fixtures/core/tests/issue_1134_forward_reference_parity/fragment_anchor.json"),
        Item::Value => match spelling {
            LegacySpelling::Unannotated => include_str!("fixtures/core/tests/issue_1134_forward_reference_parity/fragment_carried.json"),
            LegacySpelling::DeclarationTyped => include_str!("fixtures/core/tests/issue_1134_forward_reference_parity/fragment_typed_carried.json"),
        },
        Item::Reader => match reader {
            ReaderKind::Function => include_str!("fixtures/core/tests/issue_1134_forward_reference_parity/fragment_reader.json"),
            ReaderKind::Value => include_str!("fixtures/core/tests/issue_1134_forward_reference_parity/fragment_echoed.json"),
        },
    }).collect();
    layout_source(wrapped.then_some("OrderingMatrix"), &declarations)
}

fn deep_source(
    layout: &[Item],
    reader: ReaderKind,
    spelling: DeepSpelling,
    wrapped: bool,
) -> String {
    let value_signature = "(defsig {} carried (t-prim {} i32))";
    let value_def = "(def {} carried (lit {type: (t-prim {} i32)} 7))";
    let mut declarations: Vec<String> = Vec::new();
    if spelling == DeepSpelling::SeparatedSignature {
        declarations.push(value_signature.to_string());
    }
    for item in layout {
        match item {
            Item::Anchor => declarations.push(
                "(def {} anchor (fn {} (params {}) (lit {type: (t-prim {} i32)} 1)))".to_string(),
            ),
            Item::Value => match spelling {
                DeepSpelling::BodyStampOnly | DeepSpelling::SeparatedSignature => {
                    declarations.push(value_def.to_string())
                }
                DeepSpelling::AdjacentSignature => {
                    declarations.push(value_signature.to_string());
                    declarations.push(value_def.to_string());
                }
            },
            Item::Reader => declarations.push(match reader {
                ReaderKind::Function => {
                    "(def {} reader (fn {} (params {}) (var {} carried)))".to_string()
                }
                ReaderKind::Value => "(def {} echoed (var {} carried))".to_string(),
            }),
        }
    }
    if wrapped {
        format!(
            "(module {{}} OrderingMatrix\n  {})\n",
            declarations.join("\n  ")
        )
    } else {
        format!("{}\n", declarations.join("\n\n"))
    }
}

/// Check one generated program and report the disagreement, if any.
fn ordering_row_failure(
    program: &[chelis_deep::Expr],
    accepts: bool,
    label: &str,
) -> Option<String> {
    let (ir, typed) = diagnostics(program);
    if ir != typed {
        return Some(format!(
            "{label}: ingress diagnostics diverged\n  ir:    {ir:#?}\n  typed: {typed:#?}"
        ));
    }
    match (accepts, ir.is_empty()) {
        (true, false) => Some(format!("{label}: expected acceptance, got {ir:#?}")),
        (false, true) => Some(format!("{label}: expected UnboundVariable, got acceptance")),
        (false, false) if !ir.iter().any(|(kind, _)| kind == "UnboundVariable") => {
            Some(format!("{label}: expected UnboundVariable, got {ir:#?}"))
        }
        _ => None,
    }
}

fn report(failures: Vec<String>, rows: usize) {
    assert!(
        failures.is_empty(),
        "{}/{rows} ordering rows disagree with [04-INF-4]:\n\n{}",
        failures.len(),
        failures.join("\n\n")
    );
}

#[test]
fn legacy_ordering_matrix_matches_04_inf_4_at_both_ingresses() {
    let mut failures = Vec::new();
    let mut rows = 0;
    for layout in layouts() {
        for reader in [ReaderKind::Function, ReaderKind::Value] {
            for spelling in [
                LegacySpelling::Unannotated,
                LegacySpelling::DeclarationTyped,
            ] {
                for wrapped in [true, false] {
                    rows += 1;
                    let source = legacy_source(&layout, reader, spelling, wrapped);
                    let label = format!(
                        "surf layout={layout:?} reader={reader:?} spelling={spelling:?} \
                         wrapped={wrapped}\n{source}"
                    );
                    let program = legacy_program(&source);
                    let accepts = layout_accepts(&layout);
                    if let Some(failure) = ordering_row_failure(&program, accepts, &label) {
                        failures.push(failure);
                    }
                }
            }
        }
    }
    report(failures, rows);
}

#[test]
fn deep_ordering_matrix_matches_04_inf_4_at_both_ingresses() {
    let mut failures = Vec::new();
    let mut rows = 0;
    for layout in layouts() {
        for reader in [ReaderKind::Function, ReaderKind::Value] {
            for spelling in [
                DeepSpelling::BodyStampOnly,
                DeepSpelling::AdjacentSignature,
                DeepSpelling::SeparatedSignature,
            ] {
                for wrapped in [true, false] {
                    rows += 1;
                    let source = deep_source(&layout, reader, spelling, wrapped);
                    let label = format!(
                        "deep layout={layout:?} reader={reader:?} spelling={spelling:?} \
                         wrapped={wrapped}\n{source}"
                    );
                    let program = deep_file_program(&source);
                    let accepts = layout_accepts(&layout);
                    if let Some(failure) = ordering_row_failure(&program, accepts, &label) {
                        failures.push(failure);
                    }
                }
            }
        }
    }
    report(failures, rows);
}

// ---------------------------------------------------------------------------
// [04-INF-4] recursive-component ordering matrix
//
// The matrix above cannot reach this interaction. Its only function-shaped
// items never call each other, so `FunctionInferencePlan` never builds a
// multi-member recursive component and `primary_inference_groups`'
// whole-component emission path is never taken. A component is inferred as
// one unit at the first member the schedule reaches, so the schedule
// contracts it to one vertex and an edge that one member earns constrains
// every member. This matrix pins the verdicts that contraction produces.
//
// The value spelling matters here too. `carried = 7` desugars to a
// type-stamped literal, which the serialized-IR body-stamp prebind can read,
// so a lost edge shows up as an ingress DIVERGENCE. `carried = seed()` has no
// header of any kind, so the same loss shows up as an over-rejection at both
// ingresses. Both spellings are enumerated.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RecursiveItem {
    /// The recursive peer that reads nothing.
    Ping,
    Value,
    /// The recursive peer that reads the value.
    Pong,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum RecursiveValueSpelling {
    /// Desugars to a type-stamped literal the IR prebind can read.
    Stamped,
    /// No header anywhere: the type exists only once its own `def` is inferred.
    Computed,
}

fn recursive_layouts() -> Vec<Vec<RecursiveItem>> {
    use RecursiveItem::{Ping, Pong, Value};
    vec![
        vec![Ping, Value, Pong],
        vec![Ping, Pong, Value],
        vec![Value, Ping, Pong],
        vec![Value, Pong, Ping],
        vec![Pong, Ping, Value],
        vec![Pong, Value, Ping],
    ]
}

/// [04-INF-4]: the reading peer resolves exactly when the value precedes it.
fn recursive_layout_accepts(layout: &[RecursiveItem]) -> bool {
    let position = |wanted: RecursiveItem| layout.iter().position(|item| *item == wanted);
    position(RecursiveItem::Value) < position(RecursiveItem::Pong)
}

fn recursive_legacy_source(
    layout: &[RecursiveItem],
    spelling: RecursiveValueSpelling,
    wrapped: bool,
) -> String {
    let mut declarations = vec![include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/fragment_seed.json"
    )];
    declarations.extend(layout.iter().map(|item| match item {
        RecursiveItem::Ping => include_str!("fixtures/core/tests/issue_1134_forward_reference_parity/fragment_ping.json"),
        RecursiveItem::Pong => include_str!("fixtures/core/tests/issue_1134_forward_reference_parity/fragment_pong.json"),
        RecursiveItem::Value => match spelling {
            RecursiveValueSpelling::Stamped => include_str!("fixtures/core/tests/issue_1134_forward_reference_parity/fragment_carried.json"),
            RecursiveValueSpelling::Computed => include_str!("fixtures/core/tests/issue_1134_forward_reference_parity/fragment_computed_carried.json"),
        },
    }));
    layout_source(wrapped.then_some("RecursiveOrderingMatrix"), &declarations)
}

#[test]
fn recursive_component_ordering_matrix_matches_04_inf_4_at_both_ingresses() {
    let mut failures = Vec::new();
    let mut rows = 0;
    for layout in recursive_layouts() {
        for spelling in [
            RecursiveValueSpelling::Stamped,
            RecursiveValueSpelling::Computed,
        ] {
            for wrapped in [true, false] {
                rows += 1;
                let source = recursive_legacy_source(&layout, spelling, wrapped);
                let label = format!(
                    "recursive layout={layout:?} spelling={spelling:?} wrapped={wrapped}\n{source}"
                );
                let program = legacy_program(&source);
                let accepts = recursive_layout_accepts(&layout);
                if let Some(failure) = ordering_row_failure(&program, accepts, &label) {
                    failures.push(failure);
                }
            }
        }
    }
    report(failures, rows);
}

// ---------------------------------------------------------------------------
// Function-visibility non-regression matrix
//
// [04-INF-4] governs eager values. It says nothing about when a `defsig`-less
// FUNCTION's inferred scheme becomes available, which [04-INF-2]/[04-INF-3]
// and the planner own. This change must therefore leave that axis exactly
// where it was, and an earlier attempt did not: replacing the hoist order
// (every module function spliced at the earliest module-function ordinal) with
// a bare dependency order silently dropped the guarantee the hoist supplied,
// that a later declaration can always resolve a module function. The schedule
// now keeps the hoist order as its priority and adds only reference edges, so
// these rows hold by construction; they stay as the pin in both directions, so
// a future ordering change cannot widen the axis either.
//
// Expected verdicts are derived, not measured: a module-wrapped unit resolves a
// function from any declaration at or after the first function, because that is
// where the planner's region begins; a bare unit has no planner and resolves
// only backwards.
// ---------------------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum FnDepItem {
    /// `defsig`-less leaf whose scheme exists only after its body is inferred.
    Helper,
    /// `defsig`-less function that calls `Helper`.
    Caller,
    /// An eager value that reads `Caller`, i.e. a non-function declaration
    /// depending on a function's inferred scheme.
    Consumer,
}

fn fn_dep_layouts() -> Vec<Vec<FnDepItem>> {
    use FnDepItem::{Caller, Consumer, Helper};
    vec![
        vec![Helper, Caller, Consumer],
        vec![Helper, Consumer, Caller],
        vec![Caller, Helper, Consumer],
        vec![Caller, Consumer, Helper],
        vec![Consumer, Helper, Caller],
        vec![Consumer, Caller, Helper],
    ]
}

fn fn_dep_layout_accepts(layout: &[FnDepItem], wrapped: bool) -> bool {
    let position = |wanted: FnDepItem| {
        layout
            .iter()
            .position(|item| *item == wanted)
            .expect("every layout carries every item")
    };
    let helper = position(FnDepItem::Helper);
    let caller = position(FnDepItem::Caller);
    let consumer = position(FnDepItem::Consumer);
    if wrapped {
        // The planner orders functions by dependency, so `caller` always sees
        // `helper`; a reader resolves once the planner's region has begun.
        consumer > helper.min(caller)
    } else {
        // No module key, no planner: both edges are plain backward references.
        helper < caller && caller < consumer
    }
}

fn fn_dep_source(layout: &[FnDepItem], wrapped: bool) -> String {
    let declarations: Vec<&str> = layout
        .iter()
        .map(|item| match item {
            FnDepItem::Helper => include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/fragment_helper.json"
            ),
            FnDepItem::Caller => include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/fragment_caller.json"
            ),
            FnDepItem::Consumer => include_str!(
                "fixtures/core/tests/issue_1134_forward_reference_parity/fragment_consumer.json"
            ),
        })
        .collect();
    layout_source(wrapped.then_some("FnDepMatrix"), &declarations)
}

#[test]
fn function_visibility_matrix_is_unchanged_by_eager_value_scope() {
    let mut failures = Vec::new();
    let mut rows = 0;
    for layout in fn_dep_layouts() {
        for wrapped in [true, false] {
            rows += 1;
            let source = fn_dep_source(&layout, wrapped);
            let label = format!("fn-dep layout={layout:?} wrapped={wrapped}\n{source}");
            let program = legacy_program(&source);
            if let Some(failure) =
                ordering_row_failure(&program, fn_dep_layout_accepts(&layout, wrapped), &label)
            {
                failures.push(failure);
            }
        }
    }
    report(failures, rows);
}

/// An eager value whose initializer reaches, through a call, a value declared
/// after it. Every reference in it obeys [04-INF-4] and [04-INF-2] on its own,
/// so the scope rule must not be what rejects it; the binding-cycle detector
/// owns that verdict and must reach it identically at both ingresses. The
/// compiled read of a not-yet-assigned global in this shape is chelis#1339's
/// indirect residue and is owned there, not here.
#[test]
fn an_initialization_cycle_leaves_the_schedule_total_at_both_ingresses() {
    let program = legacy_program(include_str!(
        "fixtures/core/tests/issue_1134_forward_reference_parity/an_initialization_cycle_leaves_the_schedule_total_at_both_ingresses_1.json"
    ));
    // The cycle detector owns the verdict; this test owns the property that the
    // scheduler reaches it at all, identically at both ingresses.
    assert_rejects_identically(&program, "CycleDetected", "initialization cycle");
}
