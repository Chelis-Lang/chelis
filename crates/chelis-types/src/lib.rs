//! Type checker for the Chelis language.

pub mod adt;
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

mod builtins;
/// Source architecture guard for the `infer` module tree. Test-only: it
/// inspects source layout, so it has no place in a release build.
#[cfg(test)]
mod source_arch;

pub use builtins::{
    BUILTIN_NAMES, BUILTINS, BuiltinDecl, Realizability, ShapeClass, builtin_decl, builtin_env,
    realizability, shape_class,
};
pub use cancel::{
    CancelToken, CancelTokenGuard, EVAL_CANCELLED_MSG, current_cancel_token, install_cancel_token,
    is_cancellation,
};
pub use chelis_vocab::EffectKind;
pub use context::TypeEnv;
pub use dtype_semantics::{
    NumericTrap, RawScalar, RawTensor, ScalarValue, StorageView, TensorStorage, cast_raw,
    cast_scalar, finalize_scalar, finalize_tensor, scalar_from_f64, scalar_from_i64,
};
pub use fitness::{
    FitnessReport, StructuralStats, TypeAnalysisOutcome, analyze_ir_program,
    check_ir_program as check_ir_fitness, check_program, clean_fitness_from_stats,
    structural_stats,
};
pub use infer::{
    CheckedProgram, InferResult, InferStats, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_program,
    check_ir_with_context, check_ir_with_signature_context, check_typed_program, infer_ir_program,
    infer_program, run_on_grown_stack, set_grow_segment_bytes_for_test,
};
pub use linearity::{LinearityInfo, check_linearity, check_linearity_with_context};
pub use observation::{ElementRef, format_element};
pub use opacity::{
    LinkedProgramGuard, demangle_ident, install_linked_program_guard, is_linker_format_name,
};
