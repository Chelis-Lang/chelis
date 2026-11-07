//! Portable IR algorithms with explicit operation resources.
//! Ordinary results do not grant native execution or emission authority.

#![no_std]
#![forbid(unsafe_code)]

#[macro_use]
extern crate alloc;
#[cfg(test)]
#[macro_use(thread_local, eprintln)]
extern crate std;

pub mod anonymous_dims;
pub mod axis_admission;
pub mod callable_value;
pub mod capacity_key;
pub mod claim_pattern;
pub mod compositions;
pub mod contraction_views;
pub mod eval;
pub mod host;
pub mod host_type_state;
pub mod lower;
#[cfg(feature = "lowering-trace")]
pub mod lowering_trace;
pub mod ordered_layout;
pub mod ownership;
pub mod pipeline;
pub(crate) mod prelude;
pub mod resources;
pub mod span_sanitize;
pub mod specialize;
#[cfg(test)]
pub(crate) mod test_support;
pub mod vmap;

/// Primitive types supported by the exact tensor evaluator.
/// Tensor values retain sealed dtype-specific storage.
pub const EVAL_TENSOR_CAPABLE_PRIMS: &[chelis_numeric_core::prim::Prim] = &[
    chelis_numeric_core::prim::Prim::F32,
    chelis_numeric_core::prim::Prim::F64,
    chelis_numeric_core::prim::Prim::Bool,
    chelis_numeric_core::prim::Prim::Bf16,
    chelis_numeric_core::prim::Prim::F16,
    chelis_numeric_core::prim::Prim::Int8,
    chelis_numeric_core::prim::Prim::Int16,
    chelis_numeric_core::prim::Prim::Int32,
    chelis_numeric_core::prim::Prim::Int64,
];
