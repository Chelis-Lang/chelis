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
pub mod lower;
pub mod optimize;
pub mod tier2;
pub mod verify;

pub use dag::{Dag, DagNode, DimInfo, NodeId, RiscOp, TensorType};
pub use lower::lower_program;
