//! Private ownership IR scaffold for compiled-value-ownership Phase 2.
//!
//! Construction remains private until checked-host lowering lands. The only
//! transition to a backend-admissible value is [`verify_ownership`].
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
    reason = "the private IR becomes constructible only when checked-host lowering lands"
)]
mod classify;
mod error;
#[expect(
    dead_code,
    reason = "the private IR becomes constructible only when checked-host lowering lands"
)]
mod ir;
mod render;
mod verify;

pub use error::OwnershipError;

#[derive(Debug, Clone, PartialEq)]
pub struct OwnershipProgram(ir::OwnershipProgram);

#[derive(Debug, Clone, PartialEq)]
pub struct VerifiedOwnershipProgram(OwnershipProgram);

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
