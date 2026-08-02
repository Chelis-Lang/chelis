//! Structural locks for chelis#729 Phase 2's IR evaluator and static fold.

#[test]
fn ir_elementwise_arithmetic_cannot_reintroduce_raw_numeric_closures() {
    let source = include_str!("../src/eval.rs");
    for forbidden in [
        "impl Fn(f64, f64) -> f64",
        "impl Fn(f64) -> f64",
        "try_binary_elementwise",
        "convert_cast_data",
    ] {
        assert!(
            !source.contains(forbidden),
            "chelis#729 section C5 forbids `{forbidden}` in the IR arithmetic boundary; \
             map the RISC operation to a closed typed kernel enum instead"
        );
    }
    for required in [
        "enum ElementwiseBinOp",
        "enum ElementwiseUnOp",
        "int_tensor_binop",
        "float_tensor_binop",
        "compare_tensors",
    ] {
        assert!(
            source.contains(required),
            "chelis#729 section C5 requires the IR evaluator to retain `{required}`"
        );
    }
}

#[test]
fn static_condition_fold_keeps_values_sealed() {
    let source = include_str!("../src/lower.rs");
    let start = source
        .find("fn fold_static_cond")
        .expect("static condition fold must remain present");
    let end = source[start..]
        .find("fn fold_shape_derived_static_size")
        .map(|offset| start + offset)
        .expect("fold boundary marker must remain present");
    let fold = &source[start..end];

    assert!(fold.contains("HashMap<NodeId, ScalarValue>"));
    assert!(fold.contains("cast_scalar"));
    assert!(fold.contains("compare_scalars"));
    assert!(
        !fold.contains("HashMap<NodeId, f64>"),
        "chelis#729 section C5 forbids a lossy f64 memo in the static condition fold"
    );
}
