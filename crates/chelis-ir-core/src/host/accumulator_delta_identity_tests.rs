use super::*;
#[test]
fn equal_constructor_spelling_in_another_nominal_is_not_component_evidence() {
    fn find_fn(expr: &Expr) -> Option<&Expr> {
        if let Expr::MetaExpr(meta, _) = expr {
            return find_fn(&meta.expr);
        }
        let node = as_node(expr)?;
        if node.tag() == DeepTag::Fn {
            return Some(expr);
        }
        node.children_slice().iter().find_map(find_fn)
    }
    let source = "type Chain[n] =\n | End { v: tensor[n, f32] }\n | Link { v: tensor[n, f32], next: Chain[n] }\ntype Other[n] =\n | Link { v: tensor[n, f32], next: Chain[n] }\ndef wrong(acc: Chain[3], i: i64) -> Other[3] = Link { v: to_tensor([1.0f32, 2.0f32, 3.0f32]), next: acc }\n";
    let declarations =
        chelis_surf::parser::parse_str(source, &chelis_syntax_host::NativeStack).unwrap();
    let deep =
        chelis_surf::desugar::desugar_program(&declarations, &chelis_syntax_host::NativeStack)
            .unwrap();
    let checked = chelis_types::check_ir_program(&deep).unwrap();
    crate::test_support::with_operation(|operation| {
        assert!(
            accumulator_delta_proof(
                checked.facts().exprs().iter().find_map(find_fn).unwrap(),
                checked.facts().adt_registry(),
                operation,
            )
            .unwrap()
            .is_none()
        );
    });
}
