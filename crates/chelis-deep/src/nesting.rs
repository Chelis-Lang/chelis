//! Native-stack bound for the recursive Deep parser (chelis#2425).
//!
//! The raw parser is recursive descent over the input's nesting, a few native
//! frames per level. A native stack overflow aborts the process, so it cannot
//! become a diagnostic after the fact. The parser therefore runs on a stack
//! segment of its own, [`PARSE_SEGMENT_BYTES`], and every level asks
//! [`Descent::enter`] before it descends; when the segment is nearly spent the
//! parser reports a located nesting error instead of overflowing.
//!
//! The dedicated segment makes the parser's depth limit a function of the
//! input and the compiler build alone, whatever stack the caller runs on, so
//! `chelis check`, `chelis prove` and every other Deep ingress accept and
//! reject the same inputs with the same diagnostic. The segment is sized well
//! below the checker's grown segment (`chelis_types::run_on_grown_stack`,
//! 512 MiB), because the later front-end passes (stamping, validation, and the
//! checker walks up to the checker's own stack guard) spend more stack per
//! nesting level than the parser does and are not all guarded. That margin is
//! measured, not proven: on a debug build the parser accepts about 23,000
//! levels of nested application, and `chelis check` and `chelis prove` run to
//! a verdict at every depth up to that limit for nested applications, `let`
//! chains and function types.

use core::cell::Cell;

/// Size of the fresh stack segment that the supplied service gives each parse.
pub const PARSE_SEGMENT_BYTES: usize = 64 * 1024 * 1024;

/// Native stack (in bytes) the parser keeps in reserve within its segment. It
/// matches the type checker's red zone and leaves room to unwind and build
/// the diagnostic.
pub const STACK_RED_ZONE_BYTES: usize = 128 * 1024;

/// Nesting depth accepted when the service cannot measure the remaining stack.
/// The guard uses this static cap when no measured budget is available.
const FALLBACK_MAX_DEPTH: usize = 512;

/// Live recursive depth belongs to one parse, not to its caller or thread.
pub(crate) type ParseDepth = Cell<usize>;

/// Run `parse` on a fresh parser segment.
pub(crate) fn on_parse_segment<R>(
    parse: impl FnOnce() -> R,
    stack: &impl chelis_stack_services::StackServices,
) -> R {
    stack.grow(PARSE_SEGMENT_BYTES, parse)
}

/// One level of recursive descent. Holding it keeps the level counted; the
/// level ends when it drops.
pub(crate) struct Descent<'a>(&'a ParseDepth);

impl<'a> Descent<'a> {
    /// Enter one more level, or return `None` if the stack cannot hold it.
    pub(crate) fn enter(
        depth: &'a ParseDepth,
        stack: &impl chelis_stack_services::StackServices,
    ) -> Option<Self> {
        let exhausted = match stack.remaining_stack() {
            Some(remaining) => remaining < STACK_RED_ZONE_BYTES,
            None => depth.get() >= FALLBACK_MAX_DEPTH,
        };
        if exhausted {
            return None;
        }
        depth.set(depth.get() + 1);
        Some(Self(depth))
    }
}

impl Drop for Descent<'_> {
    fn drop(&mut self) {
        self.0.set(self.0.get() - 1);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec::Vec;

    struct MeasuredStack(Option<usize>);

    impl chelis_stack_services::StackServices for MeasuredStack {
        fn grow<R>(&self, bytes: usize, operation: impl FnOnce() -> R) -> R {
            chelis_stack_services::StackServices::grow(
                &chelis_syntax_host::NativeStack,
                bytes,
                operation,
            )
        }

        fn maybe_grow<R>(&self, red_zone: usize, bytes: usize, operation: impl FnOnce() -> R) -> R {
            chelis_stack_services::StackServices::maybe_grow(
                &chelis_syntax_host::NativeStack,
                red_zone,
                bytes,
                operation,
            )
        }

        fn remaining_stack(&self) -> Option<usize> {
            self.0
        }
    }

    #[test]
    fn measured_guard_accepts_the_red_zone_boundary_and_rejects_less() {
        let depth = Cell::new(0);
        assert!(Descent::enter(&depth, &MeasuredStack(Some(STACK_RED_ZONE_BYTES))).is_some());
        assert!(Descent::enter(&depth, &MeasuredStack(Some(STACK_RED_ZONE_BYTES - 1))).is_none());
        assert_eq!(depth.get(), 0);
    }

    #[test]
    fn unmeasured_guard_counts_each_parse_separately_and_releases_levels() {
        let first = Cell::new(0);
        let second = Cell::new(0);
        let stack = MeasuredStack(None);
        let mut levels: Vec<_> = (0..FALLBACK_MAX_DEPTH)
            .map(|_| Descent::enter(&first, &stack).expect("fallback level is available"))
            .collect();
        assert!(Descent::enter(&first, &stack).is_none());
        assert!(Descent::enter(&second, &stack).is_some());
        levels.pop();
        assert!(Descent::enter(&first, &stack).is_some());
        drop(levels);
        assert_eq!(first.get(), 0);
        assert_eq!(second.get(), 0);
    }
}
