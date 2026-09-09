//! Type checker for the Chelis language.
//!
//! Closed effect identity is owned by [`chelis_vocab::EffectKind`]; this crate
//! does not provide a compatibility re-export.
//!
//! ```compile_fail
//! use chelis_types::EffectKind;
//! ```

pub mod adt;
pub mod agreement;
pub mod cancel;
pub mod context;
pub(crate) mod deep_type;
pub mod dtype_semantics;
pub mod env;
pub mod errors;
pub mod fitness;
pub mod infer;
pub mod invariants;
pub mod known_tags;
pub mod linearity;
pub mod manifest;
pub mod observation;
pub(crate) mod opacity;
pub(crate) mod pipe_stage;
pub(crate) mod session;
#[cfg(feature = "checkpoint-compile-probe")]
#[doc(hidden)]
pub struct DiagnosticCheckpoint(session::DiagnosticCheckpoint);
#[cfg(feature = "checkpoint-compile-probe")]
#[doc(hidden)]
pub fn checkpoint_iter_compile_probe(
    sink: &errors::DiagnosticSink<'_>,
    checkpoint: DiagnosticCheckpoint,
) {
    let _ = sink.iter_since(checkpoint.0);
}
pub mod types;
pub mod unify;
pub mod unsupported;

mod rejection_registry_generated;

pub mod builtin_discovery;
mod builtins;
/// Source architecture guard for the `infer` module tree. Test-only: it
/// inspects source layout, so it has no place in a release build.
#[cfg(test)]
mod source_arch;

pub use builtins::{
    AxisArgumentLayout, BUILTIN_NAMES, BUILTINS, BuiltinCapabilityDecl, BuiltinDecl,
    BuiltinInferenceRule, BuiltinSemanticDomain, BuiltinSiblingCaseDecl, BuiltinSiblingCaseId,
    InferenceDisposition, Realizability, ShapeClass, axis_argument_layout, builtin_decl,
    builtin_env, prelude_adt_defs, realizability, shape_class,
};
pub use cancel::{
    CancelToken, CancelTokenGuard, EVAL_CANCELLED_MSG, cancellation_check_error,
    cancellation_requested, current_cancel_token, install_cancel_token, is_cancellation,
};
pub use context::{LibraryProofId, TypeEnv};
pub use dtype_semantics::{
    ArgReduceOp, CheckedCastKind, CheckedCastPlan, CheckedCastPlanError, CompareOp, FloatBinOp,
    FloatUnOp, IndexedTrapCandidate, IntBinOp, IntUnOp, NUMERIC_TRAP_DIV_ZERO_KIND,
    NUMERIC_TRAP_DOMAIN_KIND, NUMERIC_TRAP_DTYPE_SEPARATOR, NUMERIC_TRAP_OPERATION_SEPARATOR,
    NUMERIC_TRAP_OVERFLOW_KIND, NUMERIC_TRAP_PREFIX, NumericFamily, NumericKernelError,
    NumericTrap, RawScalar, RawTensor, ReduceWindowGradOp, ScalarValue, StorageView,
    TensorReduceOp, TensorStorage, arg_reduce_tensor_groups, bf16_from_f64_rne, cast_raw,
    cast_scalar, cast_trunc_raw, cast_trunc_scalar, cast_trunc_tensor, compare_scalar_tensor,
    compare_scalars, compare_tensor_scalar, compare_tensors, count_tensor_groups, f16_from_f64_rne,
    finalize_scalar, finalize_tensor, float_binop, float_scalar_tensor_binop, float_tensor_binop,
    float_tensor_scalar_binop, float_tensor_unop, float_unop, int_binop, int_scalar_tensor_binop,
    int_tensor_binop, int_tensor_scalar_binop, int_tensor_unop, int_unop, reduce_tensor_groups,
    reduce_window_grad_tensor_groups, scalar_from_f64, scalar_from_i64, tensor_from_scalars,
    uniform_sample,
};
pub use fitness::{
    FitnessReport, StructuralStats, TypeAnalysisOutcome, analyze_ir_program,
    check_ir_program as check_ir_fitness, check_program, clean_fitness_from_stats,
    structural_stats,
};
pub use infer::{
    CheckedProgram, InferResult, InferStats, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_program,
    check_ir_with_context, check_ir_with_signature_context, check_typed_program,
    fold_static_int_expr, infer_ir_program, infer_program, reset_grow_segment_bytes_for_test,
    run_on_grown_stack, set_grow_segment_bytes_for_test,
};
pub use linearity::{LinearityInfo, check_linearity, check_linearity_with_context};
pub use observation::{ElementRef, format_element};
pub use opacity::{
    LinkedProgramGuard, demangle_ident, install_linked_program_guard, is_linker_format_name,
};
