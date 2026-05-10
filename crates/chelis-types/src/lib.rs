//! Type checker for the Chelis language.

pub mod adt;
pub mod context;
pub mod env;
pub mod errors;
pub mod fitness;
pub mod infer;
pub mod linearity;
pub mod types;
pub mod unify;

mod builtins;
pub use builtins::{BUILTIN_NAMES, builtin_env};
pub use context::TypeEnv;
pub use fitness::{FitnessReport, check_ir_program as check_ir_fitness, check_program};
pub use infer::{
    CheckedProgram, InferResult, build_compiled_library_context, build_type_env_from_library,
    check_ir_program, check_ir_with_context, check_ir_with_signature_context, check_typed_program,
    infer_ir_program, infer_program,
};
pub use linearity::{LinearityInfo, check_linearity, check_linearity_with_context};
