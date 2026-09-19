//! Lexical binding frames for the host interpreter (chelis#2204).
//!
//! This module carries the counted receipt for frame copies. Every time the
//! interpreter deep-copies the values of a binding frame (capturing a frame
//! into a closure, cloning a callable per combinator element, or saving a
//! caller frame around an application), the number of binding entries copied
//! is added to a per-thread counter. The receipt test in `runtime/tests.rs`
//! asserts that this count does not scale with the number of closure
//! applications, which is the asymptotic promise chelis#2204 makes.

thread_local! {
    /// Binding entries deep-copied by frame clones on this thread since the
    /// last reset. Counted at the copy, never estimated.
    static FRAME_VALUE_COPIES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Record that a frame clone deep-copied `entries` bound values.
pub(crate) fn record_frame_copy(entries: usize) {
    FRAME_VALUE_COPIES.with(|copies| copies.set(copies.get() + entries as u64));
}

/// Binding entries deep-copied by frame clones on this thread since the last
/// reset.
#[cfg(test)]
pub(crate) fn frame_value_copies() -> u64 {
    FRAME_VALUE_COPIES.with(std::cell::Cell::get)
}

/// Reset [`frame_value_copies`] for this thread.
#[cfg(test)]
pub(crate) fn reset_frame_value_copies() {
    FRAME_VALUE_COPIES.with(|copies| copies.set(0));
}
