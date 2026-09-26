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

use std::cell::Cell;

/// Size of the stack segment the raw parser runs on. `stacker::grow` reserves
/// it as address space; only the pages a parse touches are committed.
pub const PARSE_SEGMENT_BYTES: usize = 64 * 1024 * 1024;

/// Native stack (in bytes) the parser keeps in reserve within its segment. It
/// matches the type checker's red zone and leaves room to unwind and build
/// the diagnostic.
pub const STACK_RED_ZONE_BYTES: usize = 128 * 1024;

/// Nesting depth accepted on a platform whose remaining stack cannot be
/// measured (`stacker::remaining_stack` returns `None`). Such a platform has
/// no budget to consult, so the guard falls back to this static cap.
const FALLBACK_MAX_DEPTH: usize = 512;

thread_local! {
    /// Live recursive-descent depth on this thread, consulted only where the
    /// remaining stack cannot be measured.
    static DEPTH: Cell<usize> = const { Cell::new(0) };
}

/// Run `parse` on a fresh parser segment.
pub(crate) fn on_parse_segment<R>(parse: impl FnOnce() -> R) -> R {
    stacker::grow(PARSE_SEGMENT_BYTES, parse)
}

/// One level of recursive descent. Holding it keeps the level counted; the
/// level ends when it drops.
pub(crate) struct Descent(());

impl Descent {
    /// Enter one more level, or `None` when the stack cannot afford it.
    pub(crate) fn enter() -> Option<Self> {
        let exhausted = match stacker::remaining_stack() {
            Some(remaining) => remaining < STACK_RED_ZONE_BYTES,
            None => DEPTH.with(Cell::get) >= FALLBACK_MAX_DEPTH,
        };
        if exhausted {
            return None;
        }
        DEPTH.with(|depth| depth.set(depth.get() + 1));
        Some(Self(()))
    }
}

impl Drop for Descent {
    fn drop(&mut self) {
        DEPTH.with(|depth| depth.set(depth.get() - 1));
    }
}
