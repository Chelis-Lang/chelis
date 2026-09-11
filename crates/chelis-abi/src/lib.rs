//! Shared descriptor metadata and ABI generation owner.
//!
//! Metadata validates finite descriptor arithmetic, never tensor payloads.
//! Raw generated packets are transport and do not confer validated ownership.

pub mod metadata;
pub mod render;
mod schema;
