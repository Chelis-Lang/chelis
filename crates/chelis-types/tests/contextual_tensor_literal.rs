//! Inference accepts contextual tensor-literal core programs.

use chelis_types::infer::infer_program;

fn fixture(name: &str) -> Vec<chelis_deep::Expr> {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/core/tests/contextual_tensor_literal")
        .join(format!("{name}.deep"));
    let source = std::fs::read_to_string(path).expect("core fixture");
    chelis_deep::parser::parse_str(&source).expect("valid Deep fixture")
}

#[test]
fn position_1_let_binding_tensor_f64_narrows_float_literals() {
    let result = infer_program(&fixture(
        "position_1_let_binding_tensor_f64_narrows_float_literals",
    ));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn position_1_let_binding_tensor_int64_narrows_int_literals() {
    let result = infer_program(&fixture(
        "position_1_let_binding_tensor_int64_narrows_int_literals",
    ));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn position_1_no_annotation_float_default_is_f32() {
    let result = infer_program(&fixture("position_1_no_annotation_float_default_is_f32"));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn position_1_no_annotation_int_default_is_int32() {
    let result = infer_program(&fixture("position_1_no_annotation_int_default_is_int32"));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn position_4_cast_int_list_to_int8() {
    let result = infer_program(&fixture("position_4_cast_int_list_to_int8"));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn position_4_cast_float_list_to_f64() {
    let result = infer_program(&fixture("position_4_cast_float_list_to_f64"));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn position_3_fn_return_tensor_f64_narrows_body_literals() {
    let result = infer_program(&fixture(
        "position_3_fn_return_tensor_f64_narrows_body_literals",
    ));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn position_2_call_with_tensor_f64_param_narrows_arg_literals() {
    let result = infer_program(&fixture(
        "position_2_call_with_tensor_f64_param_narrows_arg_literals",
    ));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn position_2_call_with_sig_decl_tensor_f64_param_narrows_arg_literals() {
    let result = infer_program(&fixture(
        "position_2_call_with_sig_decl_tensor_f64_param_narrows_arg_literals",
    ));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}

#[test]
fn negative_deep_mixed_dtype_in_to_tensor_body_errors() {
    let exprs = chelis_deep::parser::parse_str("(defsig {} xs (t-tensor {} (d-lit {} 3) (t-prim {} f32)))\n               (def {} xs (app {} (var {} to_tensor)\n                 (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 1.0)\n                   (app {} (var {} Cons) (lit {type: (t-prim {} f64)} 2.0)\n                     (app {} (var {} Cons) (lit {type: (t-prim {} f32)} 3.0)\n                       (var {} Nil))))))").expect("Deep fixture");
    let result = infer_program(&exprs);
    assert!(
        result.errors.iter().any(|e| matches!(
            e.kind,
            chelis_types::errors::CheckErrorKind::TypeMismatch
                | chelis_types::errors::CheckErrorKind::PrecisionMismatch
        )),
        "{:?}",
        result.errors
    );
}

#[test]
fn position_1_block_level_let_with_tensor_type_narrows_literals() {
    let result = infer_program(&fixture(
        "position_1_block_level_let_with_tensor_type_narrows_literals",
    ));
    assert!(result.errors.is_empty(), "{:?}", result.errors);
}
