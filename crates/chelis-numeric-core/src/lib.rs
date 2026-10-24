//! Portable numeric authority.
//!
//! Scalar operations need no allocator. The `alloc` feature adds tensor buffers
//! and owned observation text. The `serde` feature adds exact stored-bit transport.
//! Tensor transport also needs `alloc`. Schema projections belong to host consumers.
//! Existing optimizer f64-image folds use the closed operation types and sealed
//! inputs of [`dtype_semantics::fold_wide_unary`] and [`dtype_semantics::fold_wide_binary`].
#![no_std]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod activation;
pub mod agreement;
pub mod bitwise;
pub mod dtype_semantics;
pub mod ieee;
pub mod key_admission;
mod math;
pub mod named_cast;
pub mod observation;
pub mod prim;
