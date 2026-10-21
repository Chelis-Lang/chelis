//! Native stack and path services outside the portable syntax graph.

#![forbid(unsafe_code)]

pub mod module_identity;

use chelis_stack_services::StackServices;

/// Native stack resources with the existing stacker policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct NativeStack;

impl StackServices for NativeStack {
    #[inline]
    fn grow<R>(&self, bytes: usize, operation: impl FnOnce() -> R) -> R {
        stacker::grow(bytes, operation)
    }

    #[inline]
    fn maybe_grow<R>(&self, red_zone: usize, bytes: usize, operation: impl FnOnce() -> R) -> R {
        stacker::maybe_grow(red_zone, bytes, operation)
    }

    #[inline]
    fn remaining_stack(&self) -> Option<usize> {
        stacker::remaining_stack()
    }
}
