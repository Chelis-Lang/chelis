//! Structural locks for chelis#729 Phase 2's typed host-runtime boundary.

use chelis_types::{
    ArgReduceOp, FloatBinOp, FloatUnOp, IntBinOp, IntUnOp, NumericKernelError, ReduceWindowGradOp,
    ScalarValue, TensorReduceOp, TensorStorage, arg_reduce_tensor_groups, float_binop,
    float_tensor_binop, float_tensor_unop, float_unop, int_binop, int_tensor_binop,
    int_tensor_unop, int_unop, reduce_tensor_groups, reduce_window_grad_tensor_groups,
};

type ReduceKernel =
    fn(TensorReduceOp, &TensorStorage, &[Vec<usize>]) -> Result<TensorStorage, NumericKernelError>;
type ArgReduceKernel =
    fn(ArgReduceOp, &TensorStorage, &[Vec<usize>]) -> Result<TensorStorage, NumericKernelError>;
type ReduceWindowGradKernel = fn(
    ReduceWindowGradOp,
    &TensorStorage,
    &TensorStorage,
    &[Vec<usize>],
) -> Result<TensorStorage, NumericKernelError>;

#[test]
fn closed_kernel_signatures_are_the_public_arithmetic_boundary() {
    let _: fn(IntBinOp, ScalarValue, ScalarValue) -> Result<ScalarValue, NumericKernelError> =
        int_binop;
    let _: fn(IntUnOp, ScalarValue) -> Result<ScalarValue, NumericKernelError> = int_unop;
    let _: fn(FloatBinOp, ScalarValue, ScalarValue) -> Result<ScalarValue, NumericKernelError> =
        float_binop;
    let _: fn(FloatUnOp, ScalarValue) -> Result<ScalarValue, NumericKernelError> = float_unop;
    let _: fn(
        IntBinOp,
        &TensorStorage,
        &TensorStorage,
    ) -> Result<TensorStorage, NumericKernelError> = int_tensor_binop;
    let _: fn(IntUnOp, &TensorStorage) -> Result<TensorStorage, NumericKernelError> =
        int_tensor_unop;
    let _: fn(
        FloatBinOp,
        &TensorStorage,
        &TensorStorage,
    ) -> Result<TensorStorage, NumericKernelError> = float_tensor_binop;
    let _: fn(FloatUnOp, &TensorStorage) -> Result<TensorStorage, NumericKernelError> =
        float_tensor_unop;
    let _: ReduceKernel = reduce_tensor_groups;
    let _: ArgReduceKernel = arg_reduce_tensor_groups;
    let _: ReduceWindowGradKernel = reduce_window_grad_tensor_groups;
}

#[test]
fn host_runtime_cannot_reintroduce_raw_numeric_closure_helpers() {
    let source = include_str!("../src/runtime/host_ops.rs");
    for forbidden in [
        "dispatch_scalar_binop",
        "tensor_numeric_binop",
        "tensor_numeric_unop",
        "runtime_scalar_as_f64",
        "Fn(f64, f64) -> f64",
        "Fn(f64) -> f64",
    ] {
        assert!(
            !source.contains(forbidden),
            "chelis#729 §C5 forbids `{forbidden}` in the host arithmetic boundary; \
             map the callable to a closed typed kernel enum instead"
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
fn host_reductions_can_only_plan_groups_and_call_typed_kernels() {
    let source = include_str!("../src/runtime/host_ops.rs");
    let axis = source_slice(source, "fn tensor_reduce_host(", "fn tensor_permute_host(");
    let window = source_slice(
        source,
        "fn tensor_reduce_window_host(",
        "fn tensor_pad_host(",
    );
    let reduction_boundary = format!("{axis}\n{window}");

    for required in [
        "reduce_tensor_groups",
        "arg_reduce_tensor_groups",
        "TensorReduceOp::Sum",
        "TensorReduceOp::ReduceWindowSum",
        "ArgReduceOp::Argmax",
        "ArgReduceOp::Argmin",
    ] {
        assert!(
            reduction_boundary.contains(required),
            "chelis#729 Phase 2 requires `{required}` inside the host reduction boundary"
        );
    }
    for forbidden in [
        ".to_f64_lossy_vec()",
        ".to_i64_exact_vec()",
        ".from_wide(",
        ".from_wide_int(",
        "checked_add",
        "checked_mul",
        "sum_lanes",
        "best_value",
        "saw_nan",
        "acc.max",
        "acc.min",
    ] {
        assert!(
            !reduction_boundary.contains(forbidden),
            "chelis#729 Phase 2 forbids `{forbidden}` inside host reducers; only ordered index-group planning belongs there"
        );
    }
}
