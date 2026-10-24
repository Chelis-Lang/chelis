//! Host contiguous-layout admission for ordered graph contractions.

use chelis_abi::metadata::ShapeMetadata;
use chelis_dag_core::dag::TensorType;
use chelis_dag_core::tier2_ordered::{EinsumContraction, einsum_contraction_candidate};
use chelis_numeric_core::prim::Prim;

/// Apply host layout admission to a portable contraction candidate.
pub fn einsum_contraction(
    equation: &str,
    lhs: &TensorType,
    rhs: &TensorType,
    accumulator: Option<Prim>,
) -> Option<EinsumContraction> {
    let candidate = einsum_contraction_candidate(equation, lhs, rhs, accumulator)?;
    let dtype = candidate.precision.runtime_dtype().ok()?;
    let admitted = candidate
        .static_extents
        .each_ref()
        .map(|extents| ShapeMetadata::contiguous(extents, dtype).is_ok());
    candidate.admit(admitted)
}
