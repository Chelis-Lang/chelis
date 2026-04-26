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
pub use fitness::{FitnessReport, check_phase0e_program as check_phase0e_fitness, check_program};
pub use infer::{
    CheckedProgram, InferResult, build_type_env_from_library, check_phase0e_program,
    check_phase0e_with_context, check_typed_program, infer_phase0e_program, infer_program,
};
pub use linearity::{LinearityInfo, check_linearity};
