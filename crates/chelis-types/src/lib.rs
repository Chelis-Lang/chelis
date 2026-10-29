//! Native checker adapters and checked-product authority for Chelis.
//! Portable semantic data and algorithms belong to chelis-checker-core.

#![doc = "```compile_fail\nuse chelis_types::EffectKind;\n```"]

pub mod cancel;
pub mod context;
pub mod fitness;
mod generalize;
pub mod infer;
pub mod linearity;
pub mod manifest;
mod native_stack;
mod opacity;
pub mod types;

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
pub use generalize::without_generalize_sweep_oracle;
pub use infer::{
    CheckedProgram, build_compiled_library_context, build_compiled_library_context_with_base,
    build_type_env_from_library, check_ir_program, check_ir_with_context,
    check_ir_with_signature_context, check_typed_program, fold_static_int_expr, infer_ir_program,
    infer_program, reset_grow_segment_bytes_for_test, resolve_declared_surface, run_on_grown_stack,
    run_on_grown_stack_of, set_grow_segment_bytes_for_test,
};
pub use linearity::{check_linearity, check_linearity_with_context, copy_repairs};
pub use opacity::{LinkedProgramGuard, install_linked_program_guard};
