use chelis_deep::parser::parse_str;
use chelis_types::{
    CheckedProgram, build_compiled_library_context, check_ir_with_signature_context,
    check_typed_program,
};

fn checked_core(source: &str) -> CheckedProgram {
    let decls = parse_str(source).expect("Deep fixture parse");
    let deep = decls;
    check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked).expect("type check")
}

#[test]
fn unannotated_read_only_tensor_param_gets_display_ref() {
    let checked = checked_core(include_str!(
        "fixtures/core/tests/signature_inference/unannotated_read_only_tensor_param_gets_display_ref_17.deep"
    ));

    let func = checked
        .signature_inference()
        .functions
        .get("readonly")
        .expect("readonly metadata");
    assert!(!func.recursive_cycle);
    assert_eq!(func.params.len(), 2);
    assert!(!func.params[0].written);
    assert!(func.params[0].inferred_read_only);
    assert!(matches!(
        func.params[0].display_type,
        chelis_types::types::Type::Ref(_)
    ));
}

#[test]
fn consuming_param_use_stays_owned_for_display() {
    let checked = checked_core(include_str!(
        "fixtures/core/tests/signature_inference/consuming_param_use_stays_owned_for_display_40.deep"
    ));

    let func = checked
        .signature_inference()
        .functions
        .get("consume")
        .expect("consume metadata");
    assert!(!func.params[0].written);
    assert!(!func.params[0].inferred_read_only);
    assert!(!matches!(
        func.params[0].display_type,
        chelis_types::types::Type::Ref(_)
    ));
}

#[test]
fn written_param_annotation_is_authoritative() {
    let checked = checked_core(include_str!(
        "fixtures/core/tests/signature_inference/written_param_annotation_is_authoritative_64.deep"
    ));

    let func = checked
        .signature_inference()
        .functions
        .get("annotated")
        .expect("annotated metadata");
    assert!(func.params[0].written);
    assert!(!func.params[0].inferred_read_only);
    assert!(!matches!(
        func.params[0].display_type,
        chelis_types::types::Type::Ref(_)
    ));
}

#[test]
fn earlier_inferred_signature_makes_later_call_read_only() {
    let checked = checked_core(include_str!(
        "fixtures/core/tests/signature_inference/earlier_inferred_signature_makes_later_call_read_only_85.deep"
    ));

    let helper = checked
        .signature_inference()
        .functions
        .get("helper")
        .expect("helper metadata");
    let caller = checked
        .signature_inference()
        .functions
        .get("caller")
        .expect("caller metadata");
    assert!(helper.params[0].inferred_read_only);
    assert!(caller.params[0].inferred_read_only);
}

/// [04-INF-5]: a reference to a hole-signature helper is typed at the helper's
/// body-determined signature wherever the reference sits, so a forward
/// reference in a bare unit reads exactly what a backward one reads.
///
/// `def helper(x, y: tensor[4, f32])` carries one annotation, so the desugarer
/// synthesizes a `defsig` with a wildcard for `x` and for the result. The
/// schedule's hole edge infers that body before any reader, in every region.
///
/// This test previously asserted the opposite, as
/// `later_helper_does_not_retroactively_make_earlier_unconstrained_call_read_only`:
/// in a BARE unit the forward reference saw nothing, `caller`'s `a` stayed
/// unconstrained, and read-only inference defaulted it to owned, while the
/// module-wrapped spelling of the same program inferred it read-only. That
/// layout dependence is the chelis#1486 defect [04-INF-5] decides away, and
/// `earlier_inferred_signature_makes_later_call_read_only` above already pins
/// the backward half.
///
/// Regression test: before this change `caller`'s `a` was inferred owned here
/// and read-only in the other three layouts of the same program.
#[test]
fn a_forward_reference_to_a_hole_signature_helper_reads_its_body_signature() {
    let checked = checked_core(include_str!(
        "fixtures/core/tests/signature_inference/a_forward_reference_to_a_hole_signature_helper_reads_its_body_signature_130.deep"
    ));

    let helper = checked
        .signature_inference()
        .functions
        .get("helper")
        .expect("helper metadata");
    let caller = checked
        .signature_inference()
        .functions
        .get("caller")
        .expect("caller metadata");
    assert!(helper.params[0].inferred_read_only);
    assert!(
        caller.params[0].inferred_read_only,
        "`a` is passed to `helper`'s read-only `x`, and [04-INF-5] types that \
         reference at `helper`'s body-determined signature whether `helper` is \
         declared before or after `caller`"
    );
}

#[test]
fn module_wrapped_helper_signature_visible_to_caller_annotation() {
    // Pre-#181, module-wrapped multi-decl programs had NO inference
    // pre-populated during annotation (a silent no-op), so the
    // wrapped vs unwrapped flavors of the same source produced
    // different signature-inference results. PR #183 fixed
    // `annotate_ir_program{,_with_context}` to descend through
    // `(module {} name ...)` wrappers, which means the helper IS
    // visible at caller-annotation time in the wrapped case.
    //
    // This test pins the post-fix behavior: caller's `a` becomes
    // read-only because helper's read-only-`x` signature is visible by
    // the time caller is annotated.
    //
    // This comment used to record an intentional asymmetry between the
    // bare-decl and module-wrapped spellings of this program. There is
    // none any more: [04-INF-5] types every reference to a
    // hole-signature declaration at its body-determined signature
    // wherever the reference sits, and the schedule's hole edge is
    // bounded by no region, so the bare sibling above now agrees with
    // this one (chelis#1486).
    let checked = checked_core(include_str!(
        "fixtures/core/tests/signature_inference/module_wrapped_helper_signature_visible_to_caller_annotation_180.deep"
    ));

    let helper = checked
        .signature_inference()
        .functions
        .get("helper")
        .expect("helper metadata");
    let caller = checked
        .signature_inference()
        .functions
        .get("caller")
        .expect("caller metadata");
    assert!(helper.params[0].inferred_read_only);
    assert!(
        caller.params[0].inferred_read_only,
        "module-wrapped caller's `a` SHOULD become read-only: helper's signature \
         is visible at caller-annotation time after the #183 module-descent fix"
    );
}

#[test]
fn module_wrapped_backward_helper_keeps_the_same_primary_signature() {
    let checked = checked_core(include_str!(
        "fixtures/core/tests/signature_inference/module_wrapped_backward_helper_keeps_the_same_primary_signature_211.deep"
    ));
    let caller = checked
        .signature_inference()
        .functions
        .get("caller")
        .expect("caller metadata");
    assert!(
        caller.params[0].inferred_read_only,
        "dependency scheduling must agree in both source orders"
    );
}

#[test]
fn module_wrapped_unknown_helper_reports_exactly_one_unbound_root() {
    let decls = parse_str(
        include_str!("fixtures/core/tests/signature_inference/module_wrapped_unknown_helper_reports_exactly_one_unbound_root_234.deep"),
    )
    .expect("Deep fixture parse");
    let deep = decls;
    let result = check_typed_program(&deep, chelis_types::ProgramOrigin::Unlinked)
        .expect_err("unknown helper must reject");
    assert_eq!(result.errors.len(), 1, "unknown helper owns one diagnostic");
    assert!(matches!(
        result.errors[0].kind,
        chelis_types::errors::CheckErrorKind::UnboundVariable { .. }
    ));
}

#[test]
fn recursive_cycle_member_does_not_infer_read_only_param() {
    let checked = checked_core(include_str!(
        "fixtures/core/tests/signature_inference/recursive_cycle_member_does_not_infer_read_only_param_252.deep"
    ));

    let recur = checked
        .signature_inference()
        .functions
        .get("recur")
        .expect("recur metadata");
    assert!(recur.recursive_cycle);
    assert!(!recur.params[0].written);
    assert!(!recur.params[0].inferred_read_only);
}

#[test]
fn mutual_recursive_cycle_members_do_not_infer_read_only_params() {
    let checked = checked_core(include_str!(
        "fixtures/core/tests/signature_inference/mutual_recursive_cycle_members_do_not_infer_read_only_params_271.deep"
    ));

    let ping = checked
        .signature_inference()
        .functions
        .get("ping")
        .expect("ping metadata");
    let pong = checked
        .signature_inference()
        .functions
        .get("pong")
        .expect("pong metadata");
    assert!(ping.recursive_cycle);
    assert!(pong.recursive_cycle);
    assert!(!ping.params[0].inferred_read_only);
    assert!(!pong.params[0].inferred_read_only);
}

#[test]
fn context_signature_metadata_makes_imported_call_read_only() {
    let library_decls = parse_str(
        include_str!("fixtures/core/tests/signature_inference/context_signature_metadata_makes_imported_call_read_only_299.deep"),
    )
    .expect("library Deep fixture parse");
    let library_deep = library_decls;
    let (type_env, library_checked) =
        build_compiled_library_context(&library_deep, chelis_types::ProgramOrigin::Unlinked)
            .expect("library context");

    let app_decls = parse_str(
        include_str!("fixtures/core/tests/signature_inference/context_signature_metadata_makes_imported_call_read_only_309.deep"),
    )
    .expect("application Deep fixture parse");
    let app_deep = app_decls;
    let checked = check_ir_with_signature_context(
        &type_env,
        library_checked.signature_inference(),
        &app_deep,
        chelis_types::ProgramOrigin::Unlinked,
    )
    .expect("app check");

    let caller = checked
        .signature_inference()
        .functions
        .get("caller")
        .expect("caller metadata");
    assert!(caller.params[0].inferred_read_only);
}

#[test]
fn imported_rank_generic_adt_signatures_instantiate_independently() {
    // chelis#968: applying the global substitution through a scheme's
    // quantified dimension made the instantiated body dimension appear free
    // in the environment. The exported scheme then omitted that dimension
    // quantifier, so the first imported call concretized every later call.
    let library_decls = parse_str(
        include_str!("fixtures/core/tests/signature_inference/imported_rank_generic_adt_signatures_instantiate_independently_340.deep"),
    )
    .expect("library Deep fixture parse");
    let library_deep = library_decls;
    let (type_env, library_checked) =
        build_compiled_library_context(&library_deep, chelis_types::ProgramOrigin::Unlinked)
            .expect("library context");

    for (first_extent, second_extent) in [(3, 2), (2, 3)] {
        let vector = |extent| {
            let mut values = "(var {} Nil)".to_string();
            for index in (0..extent).rev() {
                values = format!(
                    "(app {{}} (var {{}} Cons) (cast {{}} (lit {{surf_literal_style: \"unsuffixed\", type: (t-prim {{}} f32)}} {index}.0) (t-prim {{}} f32)) {values})"
                );
            }
            values
        };
        let app_source = format!(
            r#"
(defsig {{}} first (t-fn {{}} (t-prim {{}} f32)))
(def {{}} first (fn {{}} (params {{}})
  (app {{}} (var {{}} value)
    (app {{}} (var {{}} make)
      (app {{}} (var {{}} to_tensor) {first_values})))))
(defsig {{}} second (t-fn {{}} (t-prim {{}} f32)))
(def {{}} second (fn {{}} (params {{}})
  (app {{}} (var {{}} value)
    (app {{}} (var {{}} make)
      (app {{}} (var {{}} to_tensor) {second_values})))))
"#,
            first_values = vector(first_extent),
            second_values = vector(second_extent),
        );
        let app_decls = parse_str(&app_source).expect("application Deep fixture parse");
        let app_deep = app_decls;
        check_ir_with_signature_context(
            &type_env,
            library_checked.signature_inference(),
            &app_deep,
            chelis_types::ProgramOrigin::Unlinked,
        )
        .unwrap_or_else(|err| {
            panic!(
                "imported generic signatures must instantiate independently for \
                 {first_extent} then {second_extent}: {:?}",
                err.errors
            )
        });
    }
}

#[test]
fn imported_fixed_rank_signature_still_rejects_a_different_extent() {
    // Negative parity for chelis#968: independent generic instantiation must
    // not weaken a genuinely fixed rank/extent contract.
    let library_decls = parse_str(
        include_str!("fixtures/core/tests/signature_inference/imported_fixed_rank_signature_still_rejects_a_different_extent_395.deep"),
    )
    .expect("library Deep fixture parse");
    let library_deep = library_decls;
    let (type_env, library_checked) =
        build_compiled_library_context(&library_deep, chelis_types::ProgramOrigin::Unlinked)
            .expect("library context");
    let app_decls = parse_str(
        include_str!("fixtures/core/tests/signature_inference/imported_fixed_rank_signature_still_rejects_a_different_extent_405.deep"),
    )
    .expect("application Deep fixture parse");
    let app_deep = app_decls;
    let err = check_ir_with_signature_context(
        &type_env,
        library_checked.signature_inference(),
        &app_deep,
        chelis_types::ProgramOrigin::Unlinked,
    )
    .expect_err("fixed tensor extent must remain enforced");
    assert!(
        err.errors
            .iter()
            .any(|error| error.message.contains("Lit(3) vs Lit(2)")),
        "fixed-rank negative control must fail for the actual extent mismatch: {:?}",
        err.errors
    );
}
