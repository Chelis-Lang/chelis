//! Spec/04 section 8.6: lexical inputs retain precedence over builtin names.
#[allow(dead_code)]
mod ownership_support;
use chelis_compiler_api::compiler::eval_selected;
use chelis_compiler_api::schema::{EvalRequest, ExecutionValue, SourceKind};
#[path = "../../../tests/support/wire_values.rs"]
mod wire_values;

#[test]
fn builtin_spelled_data_parameters_agree_in_eval_and_native_c() {
    for name in ["mean", "fold", "map", "sum"] {
        for tensor_parameter in [false, true] {
            let source = if tensor_parameter {
                format!(
                    "def shifted[n]({name}: &tensor[n, f32], offset: f32) -> tensor[n, f32] = {{\n\
             bias = insert(scalar_to_tensor(offset), 0i32, shape({name}, 0i32))\n\
             add({name}, bias)\n}}\n\
             answer = shifted(to_tensor([1.0f32, 2.0f32]), 3.0f32)\n"
                )
            } else {
                format!(
                    "def shifted[n](x: &tensor[n, f32], {name}: f32) -> tensor[n, f32] = {{\n\
             bias = insert(scalar_to_tensor({name}), 0i32, shape(x, 0i32))\n\
             add(x, bias)\n}}\n\
             answer = shifted(to_tensor([1.0f32, 2.0f32]), 3.0f32)\n"
                )
            };
            let result = eval_selected(
                EvalRequest {
                    source_kind: SourceKind::Surf,
                    source: source.clone(),
                    bindings: Default::default(),
                },
                &["answer".to_string()],
            )
            .unwrap();
            let ExecutionValue::Tensor { value } = &result.roots[0].value else {
                panic!("tensor result required");
            };
            assert_eq!(value.shape, vec![2]);
            assert_eq!(
                serde_json::to_value(&value.data).unwrap(),
                serde_json::to_value(wire_values::storage_f32(vec![4.0, 5.0])).unwrap()
            );
            let generated = ownership_support::emit(&source, name);
            let (summary, stdout) = ownership_support::run_program(&generated);
            ownership_support::balanced(&summary);
            assert_eq!(stdout, "answer = tensor(shape=[2], data=[4.0, 5.0])\n");
        }
    }
}
