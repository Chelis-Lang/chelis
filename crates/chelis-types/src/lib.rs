//! Type checker for the Chelis language.
//!
//! Closed effect identity is owned by [`chelis_vocab::EffectKind`]; this crate
//! does not provide a compatibility re-export.
//!
//! ```compile_fail
//! use chelis_types::EffectKind;
//! ```

pub mod adt;
pub mod bitwise;
pub mod cancel;
pub mod context;
pub(crate) mod deep_type;
pub mod env;
pub mod errors;
pub mod fitness;
pub mod infer;
pub mod invariants;
pub mod key_admission;
pub mod known_tags;
pub mod linearity;
pub mod manifest;
pub(crate) mod opacity;
#[cfg(test)]
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
pub mod unsupported;

mod rejection_registry_generated;

pub mod builtin_discovery;
#[cfg(test)]
mod builtin_value_contract_tests;
mod builtins;
/// Source architecture guard for the `infer` module tree. Test-only: it
/// inspects source layout, so it has no place in a release build.
#[cfg(test)]
mod source_arch;

pub use bitwise::{BitwiseError, bitwise_scalar, bitwise_tensor};
pub use builtins::{
    AdmittedBuiltin, AxisArgumentLayout, BUILTIN_NAMES, BUILTINS, BuiltinCapabilityDecl,
    BuiltinDecl, BuiltinInferenceRule, BuiltinSemanticDomain, BuiltinSiblingCaseDecl,
    BuiltinSiblingCaseId, COMPARISON_OPS, CaseKeys, InferenceDisposition, KeyParameter,
    KeyParameterSite, KeyRouting, Realizability, ShapeClass, axis_argument_layout,
    builtin_call_effect, builtin_decl, builtin_env, builtin_env_names,
    builtin_value_contract_carried, case_keys, prelude_adt_defs, realizability, shape_class,
};
pub use cancel::{
    CancelToken, CancelTokenGuard, EVAL_CANCELLED_MSG, cancellation_check_error,
    cancellation_requested, current_cancel_token, install_cancel_token, is_cancellation,
};
pub use context::{LibraryProofId, TypeEnv};
pub use fitness::{
    FitnessReport, StructuralStats, TypeAnalysisOutcome, analyze_ir_program,
    check_ir_program as check_ir_fitness, check_program, clean_fitness_from_stats,
    structural_stats,
};
pub use infer::{
    CheckedLocalAscription, CheckedProgram, DeclaredSignature, DeclaredTypeSurface, InferResult,
    InferStats, LocalAscriptionAxisClaim, LocalAscriptionComposition, LocalAscriptionId,
    LocalAscriptionOrigin, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_program,
    check_ir_with_context, check_ir_with_signature_context, check_typed_program,
    compose_local_ascriptions, fold_static_int_expr, infer_ir_program, infer_program,
    reset_grow_segment_bytes_for_test, resolve_declared_surface, run_on_grown_stack,
    run_on_grown_stack_of, set_grow_segment_bytes_for_test,
};
pub use linearity::{
    CopyRepair, CopyRepairUse, CopyRepairUseKind, LinearityInfo, check_linearity,
    check_linearity_with_context, copy_repairs, param_names as lexical_param_names,
    pattern_names as lexical_pattern_names,
};
pub use opacity::{
    LinkedProgramGuard, demangle_ident, install_linked_program_guard, is_linker_format_name,
    linked_binding_in_module_of, linked_constructor_source_name,
};
