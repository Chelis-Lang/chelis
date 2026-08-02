//! Structural locks for chelis#729 Phase 2's typed host-runtime boundary.

use chelis_types::{
    FloatBinOp, FloatUnOp, IntBinOp, IntUnOp, NumericKernelError, ScalarValue, TensorStorage,
    float_binop, float_tensor_binop, float_tensor_unop, float_unop, int_binop, int_tensor_binop,
    int_tensor_unop, int_unop,
};

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
