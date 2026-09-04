//! Private ownership IR boundary for compiled-value-ownership Phase 2.
//!
//! [`lower_ownership`] is the only public constructor for the unverified form,
//! and [`verify_ownership`] is the only transition to a backend-admissible
//! value.
//!
//! ```compile_fail
//! use chelis_ir::ownership::OwnershipProgram;
//! fn bypass() -> OwnershipProgram { OwnershipProgram(()) }
//! ```
//!
//! ```compile_fail
//! use chelis_ir::ownership::{OwnershipProgram, VerifiedOwnershipProgram};
//! fn bypass(p: OwnershipProgram) -> VerifiedOwnershipProgram {
//!     VerifiedOwnershipProgram(p)
//! }
//! ```

#[expect(
    dead_code,
    reason = "the closed heap census includes planner-only TensorStorage"
)]
mod classify;
mod error;
mod ir;
mod lower;
mod render;
mod verify;

pub use error::OwnershipError;

#[derive(Debug, Clone, PartialEq)]
pub struct OwnershipProgram(ir::OwnershipProgram);

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedOwnershipProgram(OwnershipProgram);

/// Lower a checked, concretely typed host program to the private ownership
/// representation. Verification remains a separate mandatory transition.
pub fn lower_ownership(
    checked: &chelis_types::CheckedProgram,
    host: &crate::host::ConcreteHostProgram,
    manifest: &chelis_types::manifest::RootManifest,
) -> Result<OwnershipProgram, OwnershipError> {
    lower::lower(checked, host, manifest).map(OwnershipProgram)
}

pub fn verify_ownership(
    program: OwnershipProgram,
) -> Result<VerifiedOwnershipProgram, OwnershipError> {
    verify::verify(&program.0)?;
    Ok(VerifiedOwnershipProgram(program))
}

impl VerifiedOwnershipProgram {
    pub fn render(&self) -> String {
        render::render(&self.0.0)
    }
}

#[cfg(test)]
mod tests;
