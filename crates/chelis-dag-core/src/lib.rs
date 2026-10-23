//! Portable graph models, structural checks, and graph transforms.
//!
//! The embedding supplies an allocator. Graph depth determines stack use.
//! Target compilation does not prove device execution or an unrestricted depth limit.
//! Source lowering, evaluation, host state, and target admission stay in host crates.

#![no_std]
#![forbid(unsafe_code)]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod analysis;
pub mod axis_sources;
pub mod collections;
pub mod dag;
pub mod fuse;
pub mod grad;
pub mod load_store_name;
pub mod optimize;
pub mod ownership;
pub mod span_merge;
pub mod tier2;
pub mod tier2_ordered;
pub mod verify;

pub use analysis::{
    CopyCostSummary, FunctionCopyCost, analyze_copy_costs, analyze_copy_costs_for_roots,
};
pub use dag::{Dag, DagNode, DimInfo, NodeId, RiscOp, TensorType};
pub use grad::{AdError, AdRejectionReason};
pub use load_store_name::{LoadStoreName, LoadStoreNameError};
