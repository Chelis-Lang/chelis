//! Native stack reservations for host entry points.

use std::cell::Cell;

pub(super) const GROW_SEGMENT_BYTES: usize = 512 * 1024 * 1024;

thread_local! {
                                                            static GROW_SEGMENT_OVERRIDE: Cell<Option<usize>> = const { Cell::new(None) };
}

#[doc(hidden)]
pub fn set_grow_segment_bytes_for_test(bytes: usize) {
    GROW_SEGMENT_OVERRIDE.with(|cell| cell.set(Some(bytes)));
}

#[doc(hidden)]
pub fn reset_grow_segment_bytes_for_test() {
    GROW_SEGMENT_OVERRIDE.with(|cell| cell.set(None));
}

pub(super) fn grow_segment_bytes() -> usize {
    GROW_SEGMENT_OVERRIDE
        .with(Cell::get)
        .filter(|&n| n > 0)
        .unwrap_or(GROW_SEGMENT_BYTES)
}

pub(super) fn with_grown_stack<R>(f: impl FnOnce() -> R) -> R {
    stacker::grow(grow_segment_bytes(), f)
}

pub fn run_on_grown_stack<R>(f: impl FnOnce() -> R) -> R {
    with_grown_stack(f)
}

pub fn run_on_grown_stack_of<R>(bytes: usize, f: impl FnOnce() -> R) -> R {
    let bytes = GROW_SEGMENT_OVERRIDE
        .with(Cell::get)
        .filter(|&n| n > 0)
        .unwrap_or(bytes);
    stacker::grow(bytes, f)
}

pub(crate) struct NativeStack;
impl chelis_stack_services::StackServices for NativeStack {
    fn grow<R>(&self, bytes: usize, run: impl FnOnce() -> R) -> R {
        stacker::grow(bytes, run)
    }
    fn maybe_grow<R>(&self, red_zone: usize, bytes: usize, run: impl FnOnce() -> R) -> R {
        stacker::maybe_grow(red_zone, bytes, run)
    }
    fn remaining_stack(&self) -> Option<usize> {
        stacker::remaining_stack()
    }
}
