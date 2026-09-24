//! Unsupported f8e4m3 is rejected for scalar casts, tensor casts, and declarations.

use chelis_types::check_ir_program;

fn core_program(source: &str) -> Vec<chelis_deep::Expr> {
    chelis_deep::parse_and_stamp_file(source).expect("Deep fixture")
}

#[test]
fn cast_scalar_to_f8e4m3_rejected() {
    let deep = core_program(include_str!(
        "fixtures/core/tests/f8e4m3_rejection/cast_scalar_to_f8e4m3_rejected_with_spec_diagnostic_1.deep"
    ));
    check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect_err("f8e4m3 must be rejected");
}

#[test]
fn cast_tensor_to_f8e4m3_rejected() {
    let deep = core_program(include_str!(
        "fixtures/core/tests/f8e4m3_rejection/cast_tensor_to_f8e4m3_rejected_with_spec_diagnostic_1.deep"
    ));
    check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect_err("f8e4m3 must be rejected");
}

#[test]
fn declared_tensor_with_f8e4m3_rejected() {
    let deep = core_program(include_str!(
        "fixtures/core/tests/f8e4m3_rejection/declared_tensor_with_f8e4m3_rejected_with_spec_diagnostic_1.deep"
    ));
    check_ir_program(
        &deep,
        chelis_types::ProgramOrigin::Unlinked,
        &mut chelis_types::NativeServices::new(None),
    )
    .expect_err("f8e4m3 must be rejected");
}
