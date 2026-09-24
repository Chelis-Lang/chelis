//! Authored generic contracts entail each operation's dtype requirements.
use chelis_types::{check_ir_program, check_typed_program};

fn template(name: &str, args: &[&str]) -> Vec<chelis_deep::Expr> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/core/tests/issue_1942_declared_operation_bounds")
        .join(format!("{name}.deep"));
    let mut source = std::fs::read_to_string(path).expect("core fixture");
    for (index, value) in args.iter().enumerate() {
        source = source.replace(&format!("__CORE_ARG_{index}__"), value);
    }
    chelis_deep::parser::parse_str(&source).expect("valid parameterized Deep fixture")
}

fn diagnostics(name: &str, args: &[&str], both: bool) -> Vec<String> {
    let program = template(name, args);
    let kinds = |result: Result<_, chelis_types::InferResult>| match result {
        Ok(_) => Vec::new(),
        Err(report) => report
            .errors
            .into_iter()
            .map(|e| format!("{:?}", e.kind))
            .collect::<Vec<_>>(),
    };
    let mut typed = kinds(check_typed_program(
        &program,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    ));
    typed.sort();
    if both {
        let expanded = template(&format!("{name}_expanded"), args);
        let mut ir = kinds(check_ir_program(
            &expanded,
            chelis_types::ProgramOrigin::Unlinked,
            &mut chelis_types::NativeServices::new(None),
        ));
        ir.sort();
        assert_eq!(typed, ir, "checker ingress disagreement: {name}, {args:?}");
    }
    typed
}

fn check(name: &str, args: &[&str], accepted: bool, both: bool) {
    let errors = diagnostics(name, args, both);
    if accepted {
        assert!(errors.is_empty(), "{name}, {args:?}: {errors:?}");
    } else {
        assert!(
            errors.iter().any(|kind| kind == "PrecisionMismatch"),
            "{name}, {args:?}: {errors:?}"
        );
    }
}

#[test]
fn a_deferred_cast_preserves_its_consumers_family_rejection() {
    for (args, expected) in [(&["i32"][..], false), (&["f32"][..], true)] {
        check(
            "a_deferred_cast_preserves_its_consumers_family_rejection_template_0",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn each_family_operation_requires_a_sufficient_authored_bound() {
    check(
        "each_family_operation_requires_a_sufficient_authored_bound_template_0",
        &[],
        false,
        false,
    );
    for (args, expected) in [(&["numeric"][..], false), (&["float"][..], true)] {
        check(
            "each_family_operation_requires_a_sufficient_authored_bound_template_1",
            args,
            expected,
            false,
        );
    }
    check(
        "each_family_operation_requires_a_sufficient_authored_bound_template_2",
        &[],
        false,
        false,
    );
    for (args, expected) in [(&["numeric"][..], false), (&["float"][..], true)] {
        check(
            "each_family_operation_requires_a_sufficient_authored_bound_template_3",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [(&["div"][..], false), (&["trunc_div"][..], false)] {
        check(
            "each_family_operation_requires_a_sufficient_authored_bound_template_4",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["numeric", "div"][..], false),
        (&["float", "div"][..], true),
        (&["numeric", "trunc_div"][..], false),
        (&["int", "trunc_div"][..], true),
    ] {
        check(
            "each_family_operation_requires_a_sufficient_authored_bound_template_5",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["exp"][..], false),
        (&["log"][..], false),
        (&["sin"][..], false),
        (&["cos"][..], false),
        (&["tan"][..], false),
        (&["atan"][..], false),
        (&["sqrt"][..], false),
        (&["relu"][..], false),
        (&["sigmoid"][..], false),
        (&["tanh"][..], false),
        (&["silu"][..], false),
        (&["gelu"][..], false),
        (&["recip"][..], false),
    ] {
        check(
            "each_family_operation_requires_a_sufficient_authored_bound_template_6",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["numeric", "exp"][..], false),
        (&["float", "exp"][..], true),
        (&["numeric", "log"][..], false),
        (&["float", "log"][..], true),
        (&["numeric", "sin"][..], false),
        (&["float", "sin"][..], true),
        (&["numeric", "cos"][..], false),
        (&["float", "cos"][..], true),
        (&["numeric", "tan"][..], false),
        (&["float", "tan"][..], true),
        (&["numeric", "atan"][..], false),
        (&["float", "atan"][..], true),
        (&["numeric", "sqrt"][..], false),
        (&["float", "sqrt"][..], true),
        (&["numeric", "relu"][..], false),
        (&["float", "relu"][..], true),
        (&["numeric", "sigmoid"][..], false),
        (&["float", "sigmoid"][..], true),
        (&["numeric", "tanh"][..], false),
        (&["float", "tanh"][..], true),
        (&["numeric", "silu"][..], false),
        (&["float", "silu"][..], true),
        (&["numeric", "gelu"][..], false),
        (&["float", "gelu"][..], true),
        (&["numeric", "recip"][..], false),
        (&["float", "recip"][..], true),
    ] {
        check(
            "each_family_operation_requires_a_sufficient_authored_bound_template_7",
            args,
            expected,
            false,
        );
    }
    check(
        "each_family_operation_requires_a_sufficient_authored_bound_template_8",
        &[],
        false,
        false,
    );
    for (args, expected) in [(&["numeric"][..], false), (&["float"][..], true)] {
        check(
            "each_family_operation_requires_a_sufficient_authored_bound_template_9",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn standalone_signatures_express_the_same_contract() {
    check(
        "standalone_signatures_express_the_same_contract_template_0",
        &[],
        false,
        false,
    );
    check(
        "standalone_signatures_express_the_same_contract_template_1",
        &[],
        true,
        false,
    );
}

#[test]
fn a_wrapper_cannot_silently_narrow_its_authored_contract() {
    check(
        "a_wrapper_cannot_silently_narrow_its_authored_contract_template_0",
        &[],
        false,
        false,
    );
    for (args, expected) in [(&["numeric"][..], false), (&["float"][..], true)] {
        check(
            "a_wrapper_cannot_silently_narrow_its_authored_contract_template_1",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn bounded_function_values_keep_their_restriction() {
    for (args, expected) in [(&["i32"][..], false), (&["f32"][..], true)] {
        check(
            "bounded_function_values_keep_their_restriction_template_0",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn local_lambda_and_aggregate_projection_check_the_declared_precision() {
    check(
        "local_lambda_and_aggregate_projection_check_the_declared_precision_template_0",
        &[],
        false,
        false,
    );
    check(
        "local_lambda_and_aggregate_projection_check_the_declared_precision_template_1",
        &[],
        true,
        false,
    );
    for (args, expected) in [(&["i32"][..], false), (&["f32"][..], true)] {
        check(
            "local_lambda_and_aggregate_projection_check_the_declared_precision_template_2",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn inferred_local_operands_still_bind_before_the_declaration_boundary() {
    for (args, expected) in [(&["f32"][..], true), (&["i32"][..], false)] {
        check(
            "inferred_local_operands_still_bind_before_the_declaration_boundary_template_0",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn primitive_scalar_contracts_require_bounds_too() {
    check(
        "primitive_scalar_contracts_require_bounds_too_template_0",
        &[],
        false,
        false,
    );
    for (args, expected) in [(&["numeric"][..], false), (&["float"][..], true)] {
        check(
            "primitive_scalar_contracts_require_bounds_too_template_1",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn numeric_and_integer_operation_contracts_cover_each_spec_family() {
    for (args, expected) in [
        (&["abs"][..], false),
        (&["floor"][..], false),
        (&["ceil"][..], false),
        (&["round"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_0",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["abs"][..], false),
        (&["floor"][..], false),
        (&["ceil"][..], false),
        (&["round"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_1",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["abs"][..], true),
        (&["floor"][..], true),
        (&["ceil"][..], true),
        (&["round"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_2",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["abs"][..], true),
        (&["floor"][..], true),
        (&["ceil"][..], true),
        (&["round"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_3",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [(&["max_elem"][..], false), (&["min_elem"][..], false)] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_4",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [(&["max_elem"][..], false), (&["min_elem"][..], false)] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_5",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [(&["max_elem"][..], true), (&["min_elem"][..], true)] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_6",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [(&["max_elem"][..], true), (&["min_elem"][..], true)] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_7",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["cmplt"][..], false),
        (&["lt"][..], false),
        (&["gt"][..], false),
        (&["gte"][..], false),
        (&["lte"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_8",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["cmplt"][..], false),
        (&["lt"][..], false),
        (&["gt"][..], false),
        (&["gte"][..], false),
        (&["lte"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_9",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["cmplt"][..], true),
        (&["lt"][..], true),
        (&["gt"][..], true),
        (&["gte"][..], true),
        (&["lte"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_10",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["cmplt"][..], true),
        (&["lt"][..], true),
        (&["gt"][..], true),
        (&["gte"][..], true),
        (&["lte"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_11",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["sum"][..], false),
        (&["max_reduce"][..], false),
        (&["min_reduce"][..], false),
        (&["prod_reduce"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_12",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["sum"][..], false),
        (&["max_reduce"][..], false),
        (&["min_reduce"][..], false),
        (&["prod_reduce"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_13",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["sum"][..], true),
        (&["max_reduce"][..], true),
        (&["min_reduce"][..], true),
        (&["prod_reduce"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_14",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["sum"][..], true),
        (&["max_reduce"][..], true),
        (&["min_reduce"][..], true),
        (&["prod_reduce"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_15",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["argmax_reduce"][..], false),
        (&["argmin_reduce"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_16",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["argmax_reduce"][..], false),
        (&["argmin_reduce"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_17",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["argmax_reduce"][..], true),
        (&["argmin_reduce"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_18",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["argmax_reduce"][..], true),
        (&["argmin_reduce"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_19",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["mod"][..], false),
        (&["bitand"][..], false),
        (&["bitor"][..], false),
        (&["bitxor"][..], false),
        (&["shl"][..], false),
        (&["shr"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_20",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["mod"][..], false),
        (&["bitand"][..], false),
        (&["bitor"][..], false),
        (&["bitxor"][..], false),
        (&["shl"][..], false),
        (&["shr"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_21",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["numeric", "mod"][..], false),
        (&["int", "mod"][..], true),
        (&["numeric", "bitand"][..], false),
        (&["int", "bitand"][..], true),
        (&["numeric", "bitor"][..], false),
        (&["int", "bitor"][..], true),
        (&["numeric", "bitxor"][..], false),
        (&["int", "bitxor"][..], true),
        (&["numeric", "shl"][..], false),
        (&["int", "shl"][..], true),
        (&["numeric", "shr"][..], false),
        (&["int", "shr"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_22",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["mod", "numeric"][..], false),
        (&["mod", "int"][..], true),
        (&["bitand", "numeric"][..], false),
        (&["bitand", "int"][..], true),
        (&["bitor", "numeric"][..], false),
        (&["bitor", "int"][..], true),
        (&["bitxor", "numeric"][..], false),
        (&["bitxor", "int"][..], true),
        (&["shl", "numeric"][..], false),
        (&["shl", "int"][..], true),
        (&["shr", "numeric"][..], false),
        (&["shr", "int"][..], true),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_23",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["mod"][..], false),
        (&["bitand"][..], false),
        (&["bitor"][..], false),
        (&["bitxor"][..], false),
        (&["shl"][..], false),
        (&["shr"][..], false),
    ] {
        check(
            "numeric_and_integer_operation_contracts_cover_each_spec_family_template_24",
            args,
            expected,
            false,
        );
    }
    check(
        "numeric_and_integer_operation_contracts_cover_each_spec_family_template_25",
        &[],
        false,
        false,
    );
    check(
        "numeric_and_integer_operation_contracts_cover_each_spec_family_template_26",
        &[],
        true,
        false,
    );
}

#[test]
fn window_reduction_contracts_cover_each_spec_family_on_both_ingresses() {
    for (args, expected) in [
        (&["reduce_window_mean"][..], false),
        (&["reduce_window_sum"][..], false),
        (&["reduce_window_max"][..], false),
        (&["reduce_window_min"][..], false),
    ] {
        check(
            "window_reduction_contracts_cover_each_spec_family_on_both_ingresses_template_0",
            args,
            expected,
            true,
        );
    }
    for (args, expected) in [
        (&["reduce_window_mean"][..], false),
        (&["reduce_window_sum"][..], false),
        (&["reduce_window_max"][..], false),
        (&["reduce_window_min"][..], false),
    ] {
        check(
            "window_reduction_contracts_cover_each_spec_family_on_both_ingresses_template_1",
            args,
            expected,
            true,
        );
    }
    for (args, expected) in [
        (&["numeric", "reduce_window_mean"][..], false),
        (&["float", "reduce_window_mean"][..], true),
        (&["numeric", "reduce_window_sum"][..], true),
        (&["numeric", "reduce_window_max"][..], true),
        (&["numeric", "reduce_window_min"][..], true),
    ] {
        check(
            "window_reduction_contracts_cover_each_spec_family_on_both_ingresses_template_2",
            args,
            expected,
            true,
        );
    }
    for (args, expected) in [
        (&["reduce_window_mean", "numeric"][..], false),
        (&["reduce_window_mean", "float"][..], true),
        (&["reduce_window_sum", "numeric"][..], true),
        (&["reduce_window_max", "numeric"][..], true),
        (&["reduce_window_min", "numeric"][..], true),
    ] {
        check(
            "window_reduction_contracts_cover_each_spec_family_on_both_ingresses_template_3",
            args,
            expected,
            true,
        );
    }
    for (args, expected) in [
        (&["i32", "reduce_window_mean"][..], false),
        (&["f32", "reduce_window_mean"][..], true),
        (&["bool", "reduce_window_sum"][..], false),
        (&["i32", "reduce_window_sum"][..], true),
        (&["bool", "reduce_window_max"][..], false),
        (&["i32", "reduce_window_max"][..], true),
        (&["bool", "reduce_window_min"][..], false),
        (&["i32", "reduce_window_min"][..], true),
    ] {
        check(
            "window_reduction_contracts_cover_each_spec_family_on_both_ingresses_template_4",
            args,
            expected,
            true,
        );
    }
    for (args, expected) in [
        (&["reduce_window_mean", "i32"][..], false),
        (&["reduce_window_mean", "f32"][..], true),
        (&["reduce_window_sum", "bool"][..], false),
        (&["reduce_window_sum", "i32"][..], true),
        (&["reduce_window_max", "bool"][..], false),
        (&["reduce_window_max", "i32"][..], true),
        (&["reduce_window_min", "bool"][..], false),
        (&["reduce_window_min", "i32"][..], true),
    ] {
        check(
            "window_reduction_contracts_cover_each_spec_family_on_both_ingresses_template_5",
            args,
            expected,
            true,
        );
    }
}

#[test]
fn a_lexically_shadowed_window_builtin_keeps_its_function_contract() {
    check(
        "a_lexically_shadowed_window_builtin_keeps_its_function_contract_template_0",
        &[],
        true,
        true,
    );
}

#[test]
fn omitted_signatures_do_not_publish_inferred_dtype_contracts() {
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_0",
        &[],
        true,
        false,
    );
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_1",
        &[],
        true,
        false,
    );
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_2",
        &[],
        false,
        false,
    );
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_3",
        &[],
        false,
        false,
    );
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_4",
        &[],
        true,
        false,
    );
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_5",
        &[],
        true,
        false,
    );
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_6",
        &[],
        true,
        false,
    );
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_7",
        &[],
        false,
        false,
    );
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_8",
        &[],
        true,
        false,
    );
    check(
        "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_9",
        &[],
        false,
        false,
    );
    for (args, expected) in [(&["add"][..], true), (&["trunc_div"][..], true)] {
        check(
            "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_10",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [(&["add"][..], false), (&["trunc_div"][..], false)] {
        check(
            "omitted_signatures_do_not_publish_inferred_dtype_contracts_template_11",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn later_callers_do_not_supply_an_omitted_generic_contract() {
    check(
        "later_callers_do_not_supply_an_omitted_generic_contract_template_0",
        &[],
        true,
        false,
    );
    check(
        "later_callers_do_not_supply_an_omitted_generic_contract_template_1",
        &[],
        false,
        false,
    );
}

#[test]
fn unannotated_wrappers_do_not_acquire_their_callees_bounds() {
    check(
        "unannotated_wrappers_do_not_acquire_their_callees_bounds_template_0",
        &[],
        true,
        false,
    );
    check(
        "unannotated_wrappers_do_not_acquire_their_callees_bounds_template_1",
        &[],
        false,
        false,
    );
}

#[test]
fn escaping_lambdas_need_sufficient_authored_admission() {
    check(
        "escaping_lambdas_need_sufficient_authored_admission_template_0",
        &[],
        true,
        false,
    );
    check(
        "escaping_lambdas_need_sufficient_authored_admission_template_1",
        &[],
        false,
        false,
    );
    check(
        "escaping_lambdas_need_sufficient_authored_admission_template_2",
        &[],
        true,
        false,
    );
    check(
        "escaping_lambdas_need_sufficient_authored_admission_template_3",
        &[],
        false,
        false,
    );
    check(
        "escaping_lambdas_need_sufficient_authored_admission_template_4",
        &[],
        true,
        false,
    );
    check(
        "escaping_lambdas_need_sufficient_authored_admission_template_5",
        &[],
        false,
        false,
    );
}

#[test]
fn inferred_family_lambdas_bind_monomorphically_within_the_declaration() {
    for (args, expected) in [(&["f32", "1.0"][..], true), (&["i32", "1"][..], false)] {
        check(
            "inferred_family_lambdas_bind_monomorphically_within_the_declaration_template_0",
            args,
            expected,
            false,
        );
    }
    check(
        "inferred_family_lambdas_bind_monomorphically_within_the_declaration_template_1",
        &[],
        false,
        false,
    );
}

#[test]
fn contract_transport_and_unconstrained_inference_remain_polymorphic() {
    check(
        "contract_transport_and_unconstrained_inference_remain_polymorphic_template_0",
        &[],
        true,
        false,
    );
    check(
        "contract_transport_and_unconstrained_inference_remain_polymorphic_template_1",
        &[],
        true,
        false,
    );
    check(
        "contract_transport_and_unconstrained_inference_remain_polymorphic_template_2",
        &[],
        true,
        false,
    );
    check(
        "contract_transport_and_unconstrained_inference_remain_polymorphic_template_3",
        &[],
        true,
        false,
    );
    check(
        "contract_transport_and_unconstrained_inference_remain_polymorphic_template_4",
        &[],
        true,
        false,
    );
    check(
        "contract_transport_and_unconstrained_inference_remain_polymorphic_template_5",
        &[],
        false,
        false,
    );
}

#[test]
fn arithmetic_and_matmul_require_their_operand_families() {
    check(
        "arithmetic_and_matmul_require_their_operand_families_template_0",
        &[],
        false,
        false,
    );
    for (args, expected) in [
        (&["float"][..], true),
        (&["int"][..], true),
        (&["numeric"][..], true),
    ] {
        check(
            "arithmetic_and_matmul_require_their_operand_families_template_1",
            args,
            expected,
            false,
        );
    }
    check(
        "arithmetic_and_matmul_require_their_operand_families_template_2",
        &[],
        false,
        false,
    );
    check(
        "arithmetic_and_matmul_require_their_operand_families_template_3",
        &[],
        true,
        false,
    );
}

#[test]
fn a_primitive_function_value_carries_its_admission_requirement() {
    for (args, expected) in [(&["i32"][..], false), (&["f32"][..], true)] {
        check(
            "a_primitive_function_value_carries_its_admission_requirement_template_0",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [(&["i32"][..], false), (&["f32"][..], true)] {
        check(
            "a_primitive_function_value_carries_its_admission_requirement_template_1",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn every_family_primitive_alias_preserves_its_requirement() {
    check(
        "every_family_primitive_alias_preserves_its_requirement_template_0",
        &[],
        false,
        false,
    );
    check(
        "every_family_primitive_alias_preserves_its_requirement_template_1",
        &[],
        true,
        false,
    );
    check(
        "every_family_primitive_alias_preserves_its_requirement_template_2",
        &[],
        false,
        false,
    );
    check(
        "every_family_primitive_alias_preserves_its_requirement_template_3",
        &[],
        true,
        false,
    );
    for (args, expected) in [(&["div"][..], false), (&["trunc_div"][..], false)] {
        check(
            "every_family_primitive_alias_preserves_its_requirement_template_4",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["float", "div"][..], true),
        (&["int", "trunc_div"][..], true),
    ] {
        check(
            "every_family_primitive_alias_preserves_its_requirement_template_5",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["exp"][..], false),
        (&["log"][..], false),
        (&["sin"][..], false),
        (&["cos"][..], false),
        (&["tan"][..], false),
        (&["atan"][..], false),
        (&["sqrt"][..], false),
        (&["relu"][..], false),
        (&["sigmoid"][..], false),
        (&["tanh"][..], false),
        (&["silu"][..], false),
        (&["gelu"][..], false),
        (&["recip"][..], false),
    ] {
        check(
            "every_family_primitive_alias_preserves_its_requirement_template_6",
            args,
            expected,
            false,
        );
    }
    for (args, expected) in [
        (&["exp"][..], true),
        (&["log"][..], true),
        (&["sin"][..], true),
        (&["cos"][..], true),
        (&["tan"][..], true),
        (&["atan"][..], true),
        (&["sqrt"][..], true),
        (&["relu"][..], true),
        (&["sigmoid"][..], true),
        (&["tanh"][..], true),
        (&["silu"][..], true),
        (&["gelu"][..], true),
        (&["recip"][..], true),
    ] {
        check(
            "every_family_primitive_alias_preserves_its_requirement_template_7",
            args,
            expected,
            false,
        );
    }
    check(
        "every_family_primitive_alias_preserves_its_requirement_template_8",
        &[],
        false,
        false,
    );
    check(
        "every_family_primitive_alias_preserves_its_requirement_template_9",
        &[],
        true,
        false,
    );
}

#[test]
fn inferred_family_requirements_intersect_without_narrowing_authored_contracts() {
    for (args, expected) in [(&["f32"][..], true), (&["i32"][..], false)] {
        check(
            "inferred_family_requirements_intersect_without_narrowing_authored_contracts_template_0",
            args,
            expected,
            false,
        );
    }
}

#[test]
fn window_shape_diagnostics_precede_family_admission_on_both_ingresses() {
    for name in [
        "window_shape_diagnostics_precede_family_admission_on_both_ingresses_0",
        "window_shape_diagnostics_precede_family_admission_on_both_ingresses_1",
        "window_shape_diagnostics_precede_family_admission_on_both_ingresses_2",
    ] {
        let errors = diagnostics(name, &[], true);
        assert!(
            errors.iter().any(|kind| kind == "DimensionMismatch"),
            "{name}: {errors:?}"
        );
        assert!(
            !errors.iter().any(|kind| kind == "PrecisionMismatch"),
            "{name}: {errors:?}"
        );
    }
}

#[test]
fn deferred_window_shape_diagnostics_precede_late_family_rejection_on_both_ingresses() {
    for args in [&["bool"][..], &["i32"][..]] {
        let errors = diagnostics(
            "deferred_window_shape_diagnostics_precede_late_family_rejection_on_both_ingresses_template_0",
            args,
            true,
        );
        assert!(
            errors.iter().any(|kind| kind == "DimensionMismatch"),
            "{errors:?}"
        );
        assert!(
            !errors.iter().any(|kind| kind == "PrecisionMismatch"),
            "{errors:?}"
        );
    }
    let errors = diagnostics(
        "deferred_window_shape_diagnostics_precede_late_family_rejection_on_both_ingresses_template_1",
        &[],
        true,
    );
    assert!(
        errors.iter().any(|kind| kind == "PrecisionMismatch"),
        "{errors:?}"
    );
}

#[test]
fn empty_literal_precision_cannot_evade_the_operation_contract() {
    let errors = diagnostics(
        "empty_literal_precision_cannot_evade_the_operation_contract_0",
        &[],
        false,
    );
    assert!(
        errors.iter().any(|kind| kind == "TypeMismatch"),
        "{errors:?}"
    );
    check(
        "empty_literal_precision_cannot_evade_the_operation_contract_1",
        &[],
        false,
        false,
    );
}

#[test]
fn a_failed_family_application_does_not_also_claim_its_operand_was_never_bound() {
    let errors = diagnostics(
        "a_failed_family_application_does_not_also_claim_its_operand_was_never_bound",
        &[],
        false,
    );
    assert!(
        errors.iter().any(|kind| kind == "PrecisionMismatch"),
        "{errors:?}"
    );
    assert_eq!(
        errors
            .iter()
            .filter(|kind| kind.as_str() == "TypeMismatch")
            .count(),
        1,
        "only the still-unbound operand should have a constructor obligation: {errors:?}"
    );
}
