//! Shared descriptor metadata and ABI generation owner.
//!
//! Metadata validates finite descriptor arithmetic, never tensor payloads.
//! Raw generated packets are transport and do not confer validated ownership.
//!
//! The caller supplies an allocator. Native test services stay outside the normal graph.

#![no_std]

extern crate alloc;
#[cfg(test)]
extern crate std;

pub mod failure;
pub mod mapped;
pub mod metadata;
pub mod render;
mod schema;
