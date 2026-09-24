//! [05-OP-62], spec/04 §4.5, spec/05 §4.2: actual concat capability,
//! not result-annotation spelling, decides whether a helper has a DAG.
use chelis_ir::host::{HostLoweringSession, host_def_kernel};

fn checked(src: &str) -> chelis_types::CheckedProgram {
    let expanded = chelis_deep::parser::parse_str(src).expect("Deep fixture");
    let typed = chelis_types::check_ir_program(
        &expanded,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect("type check");
    let effected =
        chelis_effects::check_program(&typed, &mut chelis_types::NativeServices::new(None))
            .expect("effect check");
    chelis_types::check_linearity(&effected, &mut chelis_types::NativeServices::new(None))
        .expect("linearity")
}

fn decision(source: &str, name: &str, kernel: bool) {
    let program = checked(source);
    let result = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        host_def_kernel(
            &mut HostLoweringSession::new(&program),
            name,
            None,
            runtime,
            &mut chelis_types::NativeServices::new(None),
        )
    })
    .map(|value| value.is_some())
    .map_err(|error| error.to_string());
    assert_eq!(result, Ok(kernel), "{source}");
}

#[test]
fn annotated_dynamic_helper_is_host_before_lowering() {
    decision(
        include_str!(
            "fixtures/core/tests/issue_1906_concat_admission/annotated_dynamic_helper_is_host_before_lowering.dp"
        ),
        "probabilities",
        false,
    );
}

#[test]
fn annotated_dynamic_direct_is_host_before_lowering() {
    decision(
        include_str!(
            "fixtures/core/tests/issue_1906_concat_admission/annotated_dynamic_direct_is_host_before_lowering_2.dp"
        ),
        "probabilities",
        false,
    );
}

#[test]
fn static_direct_concat_remains_kernel() {
    decision(
        include_str!(
            "fixtures/core/tests/issue_1906_concat_admission/static_direct_concat_remains_kernel_3.dp"
        ),
        "probabilities",
        true,
    );
}

#[test]
fn static_actuals_through_wildcard_helper_remain_kernel() {
    decision(
        include_str!(
            "fixtures/core/tests/issue_1906_concat_admission/static_actuals_through_wildcard_helper_remain_kernel.dp"
        ),
        "probabilities",
        true,
    );
}

#[test]
fn static_lexical_alias_and_list_remain_kernel() {
    decision(
        include_str!(
            "fixtures/core/tests/issue_1906_concat_admission/static_lexical_alias_and_list_remain_kernel_5.dp"
        ),
        "probabilities",
        true,
    );
}

#[test]
fn dynamic_lexical_alias_is_host() {
    decision(
        include_str!(
            "fixtures/core/tests/issue_1906_concat_admission/dynamic_lexical_alias_is_host_6.dp"
        ),
        "probabilities",
        false,
    );
}

#[test]
fn distinct_calls_do_not_reuse_first_shape_decision() {
    let source = include_str!(
        "fixtures/core/tests/issue_1906_concat_admission/distinct_calls_do_not_reuse_first_shape_decision.dp"
    );
    decision(source, "fixed", true);
    decision(source, "dynamic", false);
    decision(source, "fixed", true);
    decision(source, "dynamic", false);
}

#[test]
fn same_activation_static_then_dynamic_is_host() {
    decision(
        include_str!(
            "fixtures/core/tests/issue_1906_concat_admission/same_activation_static_then_dynamic_is_host.dp"
        ),
        "run",
        false,
    );
}

#[test]
fn same_activation_dynamic_then_static_is_host() {
    decision(
        include_str!(
            "fixtures/core/tests/issue_1906_concat_admission/same_activation_dynamic_then_static_is_host.dp"
        ),
        "run",
        false,
    );
}

#[test]
fn literal_shadow_does_not_inherit_runtime_input_fact() {
    decision(
        include_str!(
            "fixtures/core/tests/issue_1906_concat_admission/literal_shadow_does_not_inherit_runtime_input_fact_10.dp"
        ),
        "run",
        true,
    );
}

#[test]
fn swapped_actuals_with_fresh_formals_preserve_static_kernel() {
    let helper = include_str!(
        "fixtures/core/tests/issue_1906_concat_admission/swapped_actuals_with_fresh_formals_preserve_static_kernel_11.dp"
    );
    for (first_width, second_width, kernel) in [
        ("(d-name {} *)", "(d-lit {} 2)", false),
        ("(d-lit {} 2)", "(d-name {} *)", true),
    ] {
        decision(
            &format!(
                "{helper}{}",
                format_args!(
                    r#"(defsig {{}}
              run
              (s)
              (t-fn {{}}
                (t-tensor {{}} (d-var {{}} s) {first_width} (t-prim {{}} f32))
                (t-tensor {{}} (d-var {{}} s) {second_width} (t-prim {{}} f32))
                (t-tensor {{}} (d-var {{}} s) (d-name {{}} *) (t-prim {{}} f32))))
            
            (def {{}}
              run
              (fn {{}}
                (params {{}} (x {{type: (t-var {{}} _)}}) (y {{type: (t-var {{}} _)}}))
                (app {{span: "surf:97..122"}}
                  (var {{span: "surf:97..104"}} softmax)
                  (app {{span: "surf:105..117"}}
                    (var {{span: "surf:105..111"}} second)
                    (var {{span: "surf:112..113"}} y)
                    (var {{span: "surf:115..116"}} x))
                  (app {{span: "surf:119..121"}}
                    (var {{}} neg)
                    (lit {{span: "surf:120..121", type: (t-prim {{}} i32)}} 1)))))
            "#
                )
            ),
            "run",
            kernel,
        );
    }
}

#[test]
fn computed_dynamic_extent_is_not_claimed_as_supported_host_partition() {
    let program = checked(include_str!(
        "fixtures/core/tests/issue_1906_concat_admission/computed_dynamic_extent_is_not_claimed_as_supported_host_partition_13.dp"
    ));
    let error = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        host_def_kernel(
            &mut HostLoweringSession::new(&program),
            "run",
            None,
            runtime,
            &mut chelis_types::NativeServices::new(None),
        )
    })
    .expect_err("computed concat stays outside this admission class");
    assert!(
        error
            .to_string()
            .contains("tensor concat cannot be represented by the static tensor DAG")
    );
}

#[test]
fn forced_dynamic_concat_rejects_instead_of_fabricating_a_load() {
    use chelis_ir::dag::{DimInfo, TensorType};
    use chelis_types::types::Prim;
    use chelis_unord::UnordMap;
    let expr = chelis_deep::parser::parse_str("(app {} (var {} concat) (app {} (var {} Cons) (var {} x) (app {} (var {} Cons) (var {} x) (var {} Nil))) (lit {} 1))").unwrap().remove(0);
    let scope = UnordMap::from([(
        "x".to_string(),
        TensorType {
            dims: vec![DimInfo::Lit(2), DimInfo::Named("*".to_string(), None)],
            precision: Prim::F32,
        },
    )]);
    let error = chelis_ir::lowering_runtime::with_native_lowering_runtime(|runtime| {
        chelis_ir::lower::try_lower_subexpr_program(
            &expr,
            scope,
            UnordMap::new(),
            UnordMap::new(),
            runtime,
            &mut chelis_types::NativeServices::new(None),
        )
    })
    .expect_err("no fabricated input");
    assert!(
        error
            .to_string()
            .contains("tensor concat cannot be represented by the static tensor DAG")
    );
}
