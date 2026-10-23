//! Allocation-based type models and solver operations.
//!
//! The caller supplies explicit error evidence and an allocator.
//! Hosted diagnostics and witness creation remain outside this crate.
//!
//! Private spin locks need target atomics.
//! A reentrant interrupt must not access a store that its interrupted task holds.
//! A protected operation that unwinds leaves its store unavailable.

#![no_std]

extern crate alloc;
#[cfg(any(test, feature = "test-instrumentation"))]
extern crate std;

pub mod decisions;
pub mod model;
pub mod result_scope;
pub mod scheme;
mod store;
pub mod unify;

pub use model::*;
pub use result_scope::ResultScope;
pub use scheme::*;
pub use unify::*;
