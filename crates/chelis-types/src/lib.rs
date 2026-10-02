//! Type checker for the Chelis language.
//!
//! Closed effect identity is owned by [`chelis_vocab::EffectKind`]; this crate
//! does not provide a compatibility re-export.
//!
//! ```compile_fail
//! use chelis_types::EffectKind;
//! ```

pub mod activation;
pub mod adt;
pub mod agreement;
pub mod bitwise;
pub mod cancel;
pub mod context;
pub(crate) mod deep_type;
pub mod dtype_semantics;
pub mod env;
pub mod errors;
pub mod fitness;
pub mod infer;
pub mod invariants;
pub mod key_admission;
pub mod known_tags;
pub mod linearity;
pub mod manifest;
pub mod observation;
pub(crate) mod opacity;
mod result_scope;
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
pub mod static_seed;
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

pub use bitwise::{BitwiseError, BitwiseKind, bitwise_scalar, bitwise_tensor};
pub use builtins::{
    AxisArgumentLayout, BUILTIN_NAMES, BUILTINS, BuiltinCapabilityDecl, BuiltinDecl,
    BuiltinInferenceRule, BuiltinSemanticDomain, BuiltinSiblingCaseDecl, BuiltinSiblingCaseId,
    COMPARISON_OPS, CaseKeys, InferenceDisposition, KeyParameter, KeyParameterSite, KeyRouting,
    Realizability, ShapeClass, axis_argument_layout, builtin_decl, builtin_env, case_keys,
    prelude_adt_defs, realizability, shape_class,
};
pub use cancel::{
    CancelToken, CancelTokenGuard, EVAL_CANCELLED_MSG, cancellation_check_error,
    cancellation_requested, current_cancel_token, install_cancel_token, is_cancellation,
};
pub use context::{LibraryProofId, TypeEnv};
pub use dtype_semantics::{
    ArgReduceOp, CheckedCastKind, CheckedCastPlan, CheckedCastPlanError, CompareOp, FloatBinOp,
    FloatUnOp, IndexedTrapCandidate, IntBinOp, IntUnOp, KeyBits, NUMERIC_TRAP_DIV_ZERO_KIND,
    NUMERIC_TRAP_DOMAIN_KIND, NUMERIC_TRAP_DTYPE_SEPARATOR, NUMERIC_TRAP_OPERATION_SEPARATOR,
    NUMERIC_TRAP_OVERFLOW_KIND, NUMERIC_TRAP_PREFIX, NumericFamily, NumericKernelError,
    NumericTrap, PreparedDropout, PreparedUniformLike, RandomKey, RawScalar, RawTensor,
    ReduceWindowGradOp, ScalarValue, StorageView, TensorReduceOp, TensorStorage, UniformBound,
    arg_reduce_tensor_groups, bf16_from_f64_rne, cast_raw, cast_scalar, cast_trunc_raw,
    cast_trunc_scalar, cast_trunc_tensor, compare_scalar_tensor, compare_scalars,
    compare_tensor_scalar, compare_tensors, count_tensor_groups, cumsum_tensor_lanes,
    f16_from_f64_rne, finalize_scalar, finalize_tensor, float_binop, float_scalar_tensor_binop,
    float_tensor_binop, float_tensor_scalar_binop, float_tensor_unop, float_unop, int_binop,
    int_scalar_tensor_binop, int_tensor_binop, int_tensor_scalar_binop, int_tensor_unop, int_unop,
    reduce_tensor_groups, reduce_window_grad_tensor_groups, scalar_from_f64, scalar_from_i64,
    scatter_add_tensor_groups, tensor_from_scalars, uniform_like_bound_adjoint,
};
pub use fitness::{
    FitnessReport, StructuralStats, TypeAnalysisOutcome, analyze_ir_program,
    check_ir_program as check_ir_fitness, check_program, clean_fitness_from_stats,
    structural_stats,
};
pub use infer::{
    CheckedLocalTensorAscription, CheckedProgram, DeclaredSignature, DeclaredTypeSurface,
    InferResult, InferStats, LocalAscriptionAxisClaim, LocalAscriptionId,
    LocalTensorAscriptionOrigin, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_program,
    check_ir_with_context, check_ir_with_signature_context, check_typed_program,
    compose_local_tensor_ascriptions, fold_static_int_expr, infer_ir_program, infer_program,
    reset_grow_segment_bytes_for_test, resolve_declared_surface, run_on_grown_stack,
    set_grow_segment_bytes_for_test,
};
pub use linearity::{LinearityInfo, check_linearity, check_linearity_with_context};
pub use observation::{ElementRef, format_element, format_key, format_key_bits};
pub use opacity::{
    LinkedProgramGuard, demangle_ident, install_linked_program_guard, is_linker_format_name,
    linked_binding_in_module_of, linked_constructor_source_name,
};
