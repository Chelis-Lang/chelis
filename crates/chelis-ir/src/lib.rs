//! RISC DAG intermediate representation for the Chelis language.
//!
//! This crate provides:
//! - [`dag`]: The DAG data structure with RISC primitive operations.
//! - [`lower`]: Deep AST to RISC DAG lowering.
//! - [`tier2`]: Tier 2 (derived op) decomposition into Tier 1 primitives.
//! - [`optimize`]: Basic optimization passes (constant folding, DCE, CSE).
//! - [`verify`]: Structural verification of DAG invariants.

/// Primitive types the eval target's tensor-DAG evaluator can realize.
/// The evaluator stores all values as `Vec<f64>` and renders at declared
/// precision, so it can deliver any numeric prim.
pub const EVAL_TENSOR_CAPABLE_PRIMS: &[chelis_types::types::Prim] = &[
    chelis_types::types::Prim::F32,
    chelis_types::types::Prim::F64,
    chelis_types::types::Prim::Bool,
    chelis_types::types::Prim::Bf16,
    chelis_types::types::Prim::F16,
    chelis_types::types::Prim::Int8,
    chelis_types::types::Prim::Int16,
    chelis_types::types::Prim::Int32,
    chelis_types::types::Prim::Int64,
];

pub mod analysis;
pub mod axis_sources;
#[expect(
    dead_code,
    reason = "the #893 CapacityKey prerequisite lands before PR #1565 consumes it"
)]
pub mod capacity_key;
pub mod dag;
pub mod eval;
pub mod fuse;
pub mod grad;
pub mod host;
pub mod host_type_state;
pub mod load_store_name;
pub mod lower;
#[cfg(feature = "lowering-trace")]
pub mod lowering_trace;
pub mod optimize;
pub mod ownership;
pub mod pipeline;
pub mod span_merge;
pub mod span_sanitize;
pub mod specialize;
pub mod tier2;
pub mod verify;
pub mod vmap;

pub use analysis::{
    CopyCostSummary, FunctionCopyCost, analyze_copy_costs, analyze_copy_costs_for_roots,
};
pub use axis_sources::{AxisSource, check_axis_sources, node_scopes, output_axis_sources};
pub use dag::{Dag, DagNode, DimInfo, NodeId, RiscOp, TensorType};
pub use grad::{AdError, AdRejectionReason};
pub use host::{
    BlasDimRole, BlasSummaryAttempt, CompiledProgram, HelperPath, HelperSummaryRejection,
    PayloadRole, SparseOpKind, SummaryRejection, SummaryRejectionClass, SummaryRejectionDetail,
    WildcardLocation, host_program_summary_rejections,
};
pub use host_type_state::{
    ConcreteHostType, HostInferenceVar, HostPrecisionTerm, HostShapeSlot, HostShapeTerm,
    HostTensorTypeTerm, HostTypeDecodeError, HostTypeResolutionError, HostTypeTerm,
    decode_host_type, decode_host_type_metadata,
};
pub use load_store_name::{LoadStoreName, LoadStoreNameError};
pub use lower::{
    LoweredLibrary, lower_program, lower_program_to_library, lower_program_with_context,
    lower_subexpr_program, tensor_type_from_deep, try_lower_program, try_lower_program_to_library,
    try_lower_program_with_context, try_lower_subexpr_program,
};
pub use pipeline::{grad_then_fuse, grad_then_fuse_checked};
