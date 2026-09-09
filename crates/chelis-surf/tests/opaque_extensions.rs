use chelis_deep::{parse_and_stamp_file, parser::parse_str, printer::print_expr_flat};
use chelis_surf::resugar::{
    normalize_deep_for_surface_roundtrip, resugar_expression, resugar_program,
};

#[test]
fn resugaring_reports_extensions_instead_of_losing_them() {
    for source in [
        "(def {tool_data: {type: \"ablation\"}} f 1)",
        "(def {} f (fn {} (params {} (x {tool_data: 1})) (var {} x)))",
        "(def {property_quantifiers: (params {tool_data: 1})} f 1)",
    ] {
        let program = parse_and_stamp_file(source).unwrap();
        let error = resugar_program(&program).unwrap_err().to_string();
        assert!(
            error.contains("tool_data") && error.contains("preserv"),
            "{error}"
        );
    }
    let fragment = parse_str("(var {tool_data: 1} x)").unwrap();
    assert!(resugar_expression(&fragment[0]).is_err());
    assert!(resugar_program(&parse_and_stamp_file("(def {} f (lit {} 1))").unwrap()).is_ok());
    assert!(resugar_expression(&parse_str("(var {} x)").unwrap()[0]).is_ok());
}

#[test]
fn normalization_preserves_data_while_rewriting_real_expressions() {
    let source = "(def {tool_data: (tuple {span: \"id\"}), property_seed: (tuple {})} f 1)";
    let program = parse_str(source).unwrap();
    let normalized = normalize_deep_for_surface_roundtrip(&program).unwrap();
    let printed = print_expr_flat(&normalized[0]);
    assert!(
        printed.contains("tool_data: (tuple {span: \"id\"})"),
        "{printed}"
    );
    assert!(printed.contains("property_seed: (lit"), "{printed}");
    assert_eq!(
        normalize_deep_for_surface_roundtrip(&normalized).unwrap(),
        normalized
    );
}
