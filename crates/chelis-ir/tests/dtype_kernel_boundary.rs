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

fn source_slice<'a>(source: &'a str, start: &str, end: &str) -> &'a str {
    let start = source
        .find(start)
        .unwrap_or_else(|| panic!("missing `{start}`"));
    let end = source[start..]
        .find(end)
        .map(|offset| start + offset)
        .unwrap_or_else(|| panic!("missing `{end}` after slice start"));
    &source[start..end]
}

#[test]
fn ir_reductions_can_only_plan_groups_and_call_typed_kernels() {
    let source = include_str!("../src/eval.rs");
    let window = source_slice(source, "fn reduce_window(", "fn reduce_window_grad(");
    let window_grad = source_slice(source, "fn reduce_window_grad(", "fn for_each_window_pos(");
    let axis = source_slice(source, "fn axis_reduction_groups(", "fn reshape(");
    let reduction_boundary = format!("{window}\n{window_grad}\n{axis}");

    for required in [
        "reduce_tensor_groups",
        "reduce_window_grad_tensor_groups",
        "arg_reduce_tensor_groups",
    ] {
        assert!(
            reduction_boundary.contains(required),
            "chelis#729 Phase 2 requires `{required}` inside the IR reduction boundary"
        );
    }
    for required in [
        "TensorReduceOp::Sum",
        "TensorReduceOp::ReduceWindowSum",
        "ArgReduceOp::Argmax",
        "ArgReduceOp::Argmin",
    ] {
        assert!(
            source.contains(required),
            "chelis#729 Phase 2 requires the IR dispatch to map `{required}`"
        );
    }
    for required in [
        "ReduceWindowGradOp::Sum",
        "ReduceWindowGradOp::Mean",
        "ReduceWindowGradOp::Max",
        "ReduceWindowGradOp::Min",
    ] {
        assert!(
            window_grad.contains(required),
            "chelis#729 Phase 2 requires the window-adjoint dispatch to map `{required}`"
        );
    }
    for forbidden in [
        ".to_f64_lossy_vec()",
        ".to_i64_exact_vec()",
        ".to_raw()",
        "checked_add",
        "checked_mul",
        "sum_lanes",
        "best_value",
        "f64::max",
        "f64::min",
    ] {
        assert!(
            !reduction_boundary.contains(forbidden),
            "chelis#729 Phase 2 forbids `{forbidden}` inside IR reducers; only ordered index-group planning belongs there"
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
