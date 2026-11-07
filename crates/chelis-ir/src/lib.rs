//! Native IR admission, operation services, and verified emission products.
//! Portable data and algorithms belong to chelis-ir-core.

pub mod capacity_key;
pub mod device_admission;
pub mod eval;
pub mod host;
pub mod lower;
#[cfg(feature = "lowering-trace")]
pub mod lowering_trace;
pub mod ownership;
pub mod resources;

pub use lower::{
    LoweredLibrary, lower_program, lower_program_to_library, lower_program_with_context,
    try_lower_program, try_lower_program_to_library, try_lower_program_with_context,
};
