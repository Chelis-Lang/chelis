//! Explicit stack services for portable syntax operations.
//!
//! The caller supplies the platform implementation and its resource policy.
//! No implementation or host default exists in this crate.

#![no_std]
#![forbid(unsafe_code)]

/// Stack resources for one syntax operation.
///
/// On a normal return, growth methods execute the operation exactly once and return its result.
/// The implementation must preserve panic propagation.
pub trait StackServices {
    /// Run an operation on a stack with the requested reservation.
    fn grow<R>(&self, bytes: usize, operation: impl FnOnce() -> R) -> R;

    /// Extend the stack when its remaining space enters the red zone.
    fn maybe_grow<R>(&self, red_zone: usize, bytes: usize, operation: impl FnOnce() -> R) -> R;

    /// Return the remaining native stack space, if the platform can measure it.
    fn remaining_stack(&self) -> Option<usize>;
}
