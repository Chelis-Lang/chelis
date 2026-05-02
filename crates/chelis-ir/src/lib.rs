//! RISC DAG intermediate representation for the Chelis language.
//!
//! This crate provides:
//! - [`dag`]: The DAG data structure with RISC primitive operations.
//! - [`lower`]: Deep AST to RISC DAG lowering.
//! - [`tier2`]: Tier 2 (derived op) decomposition into Tier 1 primitives.
//! - [`optimize`]: Basic optimization passes (constant folding, DCE, CSE).
//! - [`verify`]: Structural verification of DAG invariants.

pub mod dag;
pub mod eval;
pub mod fuse;
pub mod grad;
pub mod host;
pub mod lower;
pub mod optimize;
pub mod pipeline;
pub mod span_merge;
pub mod tier2;
pub mod verify;
pub mod vmap;

pub use dag::{Dag, DagNode, DimInfo, NodeId, RiscOp, TensorType};
pub use host::CompiledProgram;
pub use lower::{
    LoweredLibrary, lower_program, lower_program_to_library, lower_program_with_context,
    lower_subexpr_program, tensor_type_from_deep,
};
pub use pipeline::grad_then_fuse;
