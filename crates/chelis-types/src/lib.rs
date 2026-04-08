//! Type checker for the Chelis language.

pub mod adt;
pub mod env;
pub mod errors;
pub mod fitness;
pub mod infer;
pub mod linearity;
pub mod types;
pub mod unify;

mod builtins;
pub use builtins::{BUILTIN_NAMES, builtin_env};
pub use fitness::{FitnessReport, check_phase0e_program as check_phase0e_fitness, check_program};
pub use infer::{
    CheckedProgram, InferResult, check_phase0e_program, check_typed_program, infer_phase0e_program,
    infer_program,
};
pub use linearity::{LinearityInfo, check_linearity};
