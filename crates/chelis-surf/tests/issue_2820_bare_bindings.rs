//! Bare binding literals use the tensor default from §P10b / §5.6.
use chelis_surf::{desugar::desugar_program, parser::parse_str};
fn deep(source: &str) -> String {
    desugar_program(&parse_str(source).unwrap()).unwrap().iter()
        .map(chelis_deep::printer::print_expr).collect::<Vec<_>>().join("\n")
}
#[test]
fn bare_numeric_bindings_are_tensors_at_default_or_explicit_leaf_widths() {
    for literal in ["[1, 2, 3]", "[1.0, 2.0, 3.0]", "[-1i64, 2i64]", "[[1.0f16, 2.0f16], [3.0f16, 4.0f16]]"] {
        let output = deep(&format!("result = {literal}\n"));
        assert!(output.contains("(var {} to_tensor)"), "{literal}: {output}");
    }
}
#[test]
fn explicit_list_context_and_nonnumeric_bindings_stay_lists() {
    for source in ["result: List[f32] = [1.0, 2.0]\n", "sig result: List[i32]\nresult = [1, 2]\n", "result = [true, false]\n", "result = [\"a\", \"b\"]\n", "result = []\n", "def one() -> i32 = 1\nresult = [one(), one()]\n"] {
        let output = deep(source);
        assert!(!output.contains("(var {} to_tensor)"), "{source}: {output}");
    }
}
#[test]
fn local_bare_binding_uses_tensor_and_local_list_annotation_stays_list() {
    let inferred = deep("def sample() = {\n  xs = [1.0, 2.0]\n  xs\n}\n");
    assert!(inferred.contains("(var {} to_tensor)"), "{inferred}");
    let explicit = deep("def sample() = {\n  xs: List[f32] = [1.0, 2.0]\n  xs\n}\n");
    assert!(!explicit.contains("(var {} to_tensor)"), "{explicit}");
}

#[test]
fn standalone_tensor_signature_supplies_the_declared_literal_context() {
    let output = deep("sig result: tensor[2, f64]\nresult = [1.0, 2.0]\n");
    assert!(output.contains("(var {} to_tensor)"), "{output}");
    assert!(output.contains("(t-prim {} f64)"), "{output}");
    assert!(!output.contains("(t-prim {} f32)"), "{output}");
}
