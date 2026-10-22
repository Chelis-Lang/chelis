//! Reduction policy preserves current result types and diagnostic text.

use chelis_numeric_core::prim::{Prim, ReductionPolicyError};

#[test]
fn reduction_policy_preserves_accumulator_and_result_types() {
    for (operand, accumulator, result) in [
        (Prim::F16, Prim::F32, Prim::F16),
        (Prim::Bf16, Prim::F32, Prim::Bf16),
        (Prim::F32, Prim::F32, Prim::F32),
        (Prim::F64, Prim::F64, Prim::F64),
        (Prim::Int8, Prim::Int32, Prim::Int32),
        (Prim::Int16, Prim::Int32, Prim::Int32),
        (Prim::Int32, Prim::Int32, Prim::Int32),
        (Prim::Int64, Prim::Int64, Prim::Int64),
    ] {
        assert_eq!(operand.default_reduce_sum_accumulator(), Ok(accumulator));
        assert_eq!(operand.default_reduce_sum_result_precision(), Ok(result));
        assert_eq!(operand.sum_result_precision(accumulator), result);
    }
}

#[test]
fn reduction_policy_rejects_each_non_numeric_operand_without_owned_text() {
    for (prim, error, text) in [
        (
            Prim::Bool,
            ReductionPolicyError::BoolOperand,
            "reduce_sum is not defined on bool tensors; cast to i32 first",
        ),
        (
            Prim::F8e4m3,
            ReductionPolicyError::DeferredOperand,
            "reduce_sum: operand dtype `f8e4m3` is deferred per spec/04-type-system.md §1.1.1 and is not part of the active numeric primitive set",
        ),
        (
            Prim::String,
            ReductionPolicyError::StringOperand,
            "reduce_sum is not defined for string operands",
        ),
        (
            Prim::Key,
            ReductionPolicyError::RandomKeyOperand,
            "reduce_sum is not defined for random key operands",
        ),
    ] {
        assert_eq!(prim.default_reduce_sum_accumulator(), Err(error));
        assert_eq!(prim.default_reduce_sum_result_precision(), Err(error));
        assert_eq!(error.to_string(), text);
    }
}
