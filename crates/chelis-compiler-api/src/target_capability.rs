//! Target-to-capability mapping for realizability inference (issue #912).
//!
//! This is a wildcard-free enum match: adding a `Target` variant without
//! mapping its capability set fails to compile. The mapping lives here
//! (compiler-api) because the pipeline is driven from here and library
//! consumers (C Proof notebook, etc.) need manifests without writing their
//! own mapping.

use chelis_types::types::{Prim, Target};

/// Metal backend capability (declared from hardware spec; no f64 on Apple
/// Silicon). Declaration-only — verified in the field via [05-UNS-1] wiring.
const METAL_TENSOR_CAPABLE_PRIMS: &[Prim] = &[
    Prim::F32,
    Prim::Bool,
    Prim::Bf16,
    Prim::F16,
    Prim::Int32,
    Prim::Int64,
];

/// Return the set of primitive types the given target's tensor-DAG path
/// can realize. A def whose declared types exceed this set routes to the
/// host lane for that target.
///
/// # Compile-time enforcement
///
/// This match has NO wildcard arm. Adding a `Target` variant without a
/// corresponding arm is a compile error.
pub fn tensor_capable_prims(target: Target) -> &'static [Prim] {
    match target {
        Target::Eval => chelis_ir::EVAL_TENSOR_CAPABLE_PRIMS,
        Target::C => chelis_backend_c::TENSOR_CAPABLE_PRIMS,
        Target::Hip => chelis_backend_hip::TENSOR_CAPABLE_PRIMS,
        Target::Metal => METAL_TENSOR_CAPABLE_PRIMS,
    }
}
