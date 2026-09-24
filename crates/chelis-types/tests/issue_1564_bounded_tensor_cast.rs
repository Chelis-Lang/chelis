//! Bounded tensor casts preserve shape and constrain each instantiation.
use chelis_types::errors::CheckErrorKind;
use chelis_types::{check_ir_program, check_typed_program};

fn template(name: &str, args: &[&str]) -> Vec<chelis_deep::Expr> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/core/tests/issue_1564_bounded_tensor_cast")
        .join(format!("{name}.deep"));
    let mut source = std::fs::read_to_string(path).expect("core fixture");
    for (index, value) in args.iter().enumerate() {
        source = source.replace(&format!("__CORE_ARG_{index}__"), value);
    }
    chelis_deep::parser::parse_str(&source).expect("valid parameterized Deep fixture")
}

fn check(name: &str, args: &[&str], expected: Option<CheckErrorKind>) {
    let program = template(name, args);
    let kinds = |result: Result<_, chelis_types::InferResult>| match result {
        Ok(_) => Vec::new(),
        Err(report) => report
            .errors
            .into_iter()
            .map(|e| format!("{:?}", e.kind))
            .collect::<Vec<_>>(),
    };
    let ir = kinds(check_ir_program(
        &program,
        chelis_types::ProgramOrigin::Unlinked,
    ));
    let typed = kinds(check_typed_program(
        &program,
        chelis_types::ProgramOrigin::Unlinked,
    ));
    assert_eq!(ir, typed, "ingress parity: {name}, {args:?}");
    match expected {
        None => assert!(ir.is_empty(), "{name}, {args:?}: {ir:?}"),
        Some(kind) => assert!(
            ir.contains(&format!("{kind:?}")),
            "{name}, {args:?}: expected {kind:?}, got {ir:?}"
        ),
    }
}

#[test]
fn bounded_tensor_targets_preserve_shape_and_instantiate_independently() {
    for (args, expected) in [
        (&["float", "f32", "f64"][..], None),
        (&["float", "f16", "bf16"][..], None),
        (&["int", "i16", "i64"][..], None),
        (&["int", "i8", "i32"][..], None),
        (&["numeric", "f32", "i64"][..], None),
    ] {
        check(
            "bounded_tensor_targets_preserve_shape_and_instantiate_independently_template_0",
            args,
            expected,
        );
    }
    for (args, expected) in [(&["f32", "f64"][..], None), (&["f16", "bf16"][..], None)] {
        check(
            "bounded_tensor_targets_preserve_shape_and_instantiate_independently_template_1",
            args,
            expected,
        );
    }
    for (args, expected) in [(&["i16", "i64"][..], None), (&["i8", "i32"][..], None)] {
        check(
            "bounded_tensor_targets_preserve_shape_and_instantiate_independently_template_2",
            args,
            expected,
        );
    }
    check(
        "bounded_tensor_targets_preserve_shape_and_instantiate_independently_template_3",
        &[],
        None,
    );
}

#[test]
fn invalid_target_and_family_are_still_type_errors() {
    check(
        "invalid_target_and_family_are_still_type_errors_template_0",
        &[],
        Some(CheckErrorKind::TypeMismatch),
    );
    check(
        "invalid_target_and_family_are_still_type_errors_template_1",
        &[],
        Some(CheckErrorKind::PrecisionMismatch),
    );
    check(
        "invalid_target_and_family_are_still_type_errors_template_2",
        &[],
        Some(CheckErrorKind::PrecisionMismatch),
    );
    check(
        "invalid_target_and_family_are_still_type_errors_template_3",
        &[],
        Some(CheckErrorKind::PrecisionMismatch),
    );
    check(
        "invalid_target_and_family_are_still_type_errors_template_4",
        &[],
        Some(CheckErrorKind::CastNonTensor),
    );
    check(
        "invalid_target_and_family_are_still_type_errors_template_5",
        &[],
        Some(CheckErrorKind::DimensionMismatch),
    );
}

#[test]
fn truncating_bounded_tensor_cast_requires_float_to_integer() {
    check(
        "truncating_bounded_tensor_cast_requires_float_to_integer_template_0",
        &[],
        None,
    );
    check(
        "truncating_bounded_tensor_cast_requires_float_to_integer_template_1",
        &[],
        None,
    );
    for (args, expected) in [
        (&["float"][..], Some(CheckErrorKind::CastNonTensor)),
        (&["int"][..], Some(CheckErrorKind::CastNonTensor)),
    ] {
        check(
            "truncating_bounded_tensor_cast_requires_float_to_integer_template_2",
            args,
            expected,
        );
    }
}

#[test]
fn result_constraints_obey_bounded_target_family() {
    check(
        "result_constraints_obey_bounded_target_family_template_0",
        &[],
        None,
    );
    check(
        "result_constraints_obey_bounded_target_family_template_1",
        &[],
        Some(CheckErrorKind::PrecisionMismatch),
    );
}
