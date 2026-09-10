//! [04-SHAPE-1]/[05-OP-33]: remaining host copies consume checked metadata.
const DAG: &str = include_str!("../src/emit.rs");
fn method<'a>(source: &'a str, name: &str) -> &'a str {
    let tail = &source[source.find(&format!("fn {name}(")).unwrap()..];
    &tail[..tail.find("\n    fn ").unwrap_or(tail.len())]
}
#[test]
fn dimension_rendering_has_no_unchecked_arithmetic_variant() {
    assert!(!DAG.contains("fn emit_dim_expr("));
    let body = method(DAG, "emit_dim_info");
    assert!(body.contains("&DimInfo"));
    assert!(!body.contains("DimExpr") && !body.contains(" * ") && !body.contains(" / "));
}
fn checked_literal(source: &str) -> bool {
    let body = method(source, "emit_const_tensor");
    let before = body.find("chelis_tensor_check_literal(");
    let allocation = body.find("emit_slot_wrapper(");
    before.zip(allocation).is_some_and(|(a, b)| a < b)
        && body.contains("storage.prim() != ty.precision")
        && body.contains("i64::try_from(storage.len())")
        && body.contains("chelis_tensor_write_literal(")
        && !body.contains("memcpy(")
        && !body.contains("_data)[")
        && !body.contains(" * sizeof(")
        && !body.contains("to_f64_lossy_vec(")
}
#[test]
fn constant_copy_validates_literal_domain_before_storage() {
    assert!(checked_literal(DAG));
}
#[test]
fn literal_ingress_bypass_controls_fail() {
    for required in [
        "chelis_tensor_check_literal(",
        "storage.prim() != ty.precision",
        "i64::try_from(storage.len())",
        "chelis_tensor_write_literal(",
    ] {
        assert!(
            !checked_literal(&DAG.replace(required, "REMOVED")),
            "{required}"
        );
    }
    let late = DAG.replace("self.emit_slot_wrapper(id, ty);", "").replace(
        "chelis_tensor_check_literal(",
        "emit_slot_wrapper( chelis_tensor_check_literal(",
    );
    assert!(!checked_literal(&late));
}
