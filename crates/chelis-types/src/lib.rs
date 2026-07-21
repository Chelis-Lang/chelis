//! Type checker for the Chelis language.

pub mod adt;
pub mod context;
pub mod effect_kind;
pub mod env;
pub mod errors;
pub mod fitness;
pub mod infer;
pub mod invariants;
pub mod linearity;
pub mod observation;
pub(crate) mod opacity;
pub(crate) mod pipe_stage;
pub mod types;
pub mod unify;
pub mod unsupported;

mod builtins;
pub use builtins::{BUILTIN_NAMES, ShapeClass, builtin_env, shape_class};
pub use context::TypeEnv;
pub use effect_kind::EffectKind;
pub use fitness::{
    FitnessReport, StructuralStats, check_ir_program as check_ir_fitness, check_program,
    structural_stats,
};
pub use infer::{
    CheckedProgram, InferResult, build_compiled_library_context,
    build_compiled_library_context_with_base, build_type_env_from_library, check_ir_program,
    check_ir_with_context, check_ir_with_signature_context, check_typed_program, infer_ir_program,
    infer_program, run_on_grown_stack, set_grow_segment_bytes_for_test,
};
pub use linearity::{LinearityInfo, check_linearity, check_linearity_with_context};
pub use observation::{ElementRef, format_element};
pub use opacity::{
    LinkedProgramGuard, demangle_ident, install_linked_program_guard, is_linker_format_name,
};
