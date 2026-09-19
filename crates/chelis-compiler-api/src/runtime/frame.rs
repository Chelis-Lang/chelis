//! Lexical binding frames for the host interpreter (chelis#2204).
//!
//! An anonymous `fn` must retain every local that was bound when it was
//! created, and the list combinators apply their callback once per element.
//! When the frame was one flat map, capturing it copied every bound value and
//! every per-element clone of the callback copied them again, so a `fold` was
//! quadratic in whatever happened to be in scope. A [`Frame`] keeps the
//! innermost scope owned and every captured scope shared behind a reference
//! count: capturing freezes the current locals into a shared scope instead of
//! copying them, and cloning a frame copies only its innermost scope.
//!
//! The counted receipt for this promise lives here as well. Every deep copy of
//! a frame's own bindings adds the entries copied to a per-thread counter, and
//! the receipt test in `runtime/tests.rs` asserts that the count does not
//! scale with the number of closure applications.

use std::collections::BTreeMap;
use std::sync::Arc;

use chelis_unord::UnordMap;

use super::RuntimeValue;

thread_local! {
    /// Binding entries deep-copied by frame clones on this thread since the
    /// last reset. Counted at the copy, never estimated.
    static FRAME_VALUE_COPIES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Record that a frame clone deep-copied `entries` bound values.
fn record_frame_copy(entries: usize) {
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

/// A lexical binding frame: an owned innermost scope over a chain of shared
/// captured scopes.
///
/// Lookup walks from the innermost scope outward, so a local shadows a
/// captured binding of the same name exactly as it did when the frame was one
/// flat map. The visible bindings are the same set in the same precedence;
/// only the representation of "everything in scope" changed.
#[derive(Debug, Default)]
pub struct Frame {
    /// Bindings made in this scope since it was last captured.
    locals: UnordMap<String, RuntimeValue>,
    /// Every scope captured below this one, innermost first. Shared with the
    /// closures that captured it and never mutated after freezing.
    captured: Option<Arc<Frame>>,
}

impl Clone for Frame {
    /// Copies the innermost scope's values and shares the captured chain.
    fn clone(&self) -> Self {
        record_frame_copy(self.locals.len());
        Self {
            locals: self.locals.clone(),
            captured: self.captured.clone(),
        }
    }
}

impl Frame {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// The innermost binding of `name`, if any scope binds it.
    pub(crate) fn get(&self, name: &str) -> Option<&RuntimeValue> {
        let mut frame = self;
        loop {
            if let Some(value) = frame.locals.get(name) {
                return Some(value);
            }
            frame = frame.captured.as_deref()?;
        }
    }

    pub(crate) fn contains_key(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    /// Bind `name` in the innermost scope, shadowing any captured binding.
    /// Returns the value this scope previously bound to `name`, if any.
    pub(crate) fn insert(&mut self, name: String, value: RuntimeValue) -> Option<RuntimeValue> {
        self.locals.insert(name, value)
    }

    /// Whether no scope binds anything. Test-only receipt support.
    #[cfg(test)]
    pub(crate) fn is_empty(&self) -> bool {
        self.locals.is_empty()
            && self
                .captured
                .as_ref()
                .is_none_or(|parent| parent.is_empty())
    }

    /// Number of distinct visible names. Test-only receipt support.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.to_sorted().len()
    }

    /// Every visible binding, innermost definition winning, sorted by name.
    pub(crate) fn to_sorted(&self) -> Vec<(&String, &RuntimeValue)> {
        let mut chain = Vec::new();
        let mut frame = self;
        loop {
            chain.push(frame);
            match &frame.captured {
                Some(parent) => frame = parent,
                None => break,
            }
        }
        let mut merged: BTreeMap<&String, &RuntimeValue> = BTreeMap::new();
        // Outermost first, so an inner scope's binding overwrites an outer one.
        for frame in chain.into_iter().rev() {
            for (name, value) in frame.locals.to_sorted() {
                merged.insert(name, value);
            }
        }
        merged.into_iter().collect()
    }

    /// Snapshot this frame for a closure without copying a value.
    ///
    /// The current locals are frozen into a shared scope that both this frame
    /// and the returned one resolve through. Bindings made in this frame
    /// afterwards land in fresh locals above the frozen scope, so the closure
    /// keeps seeing exactly the values that were bound when it was created,
    /// as the by-value snapshot did. A frame with no new locals shares its
    /// existing chain, so repeated captures do not deepen it.
    pub(crate) fn capture(&mut self) -> Frame {
        if !self.locals.is_empty() {
            let frozen = Frame {
                locals: std::mem::take(&mut self.locals),
                captured: self.captured.take(),
            };
            self.captured = Some(Arc::new(frozen));
        }
        Frame {
            locals: UnordMap::new(),
            captured: self.captured.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int(value: i64) -> RuntimeValue {
        RuntimeValue::int64(value)
    }

    fn read(frame: &Frame, name: &str) -> Option<String> {
        frame.get(name).map(super::super::host_ops::render_value)
    }

    #[test]
    fn capture_snapshots_values_bound_before_it_and_not_after() {
        let mut frame = Frame::new();
        frame.insert("x".to_string(), int(1));
        let captured = frame.capture();
        frame.insert("x".to_string(), int(2));
        frame.insert("y".to_string(), int(3));
        assert_eq!(
            read(&captured, "x"),
            read(&Frame::from_value("x", int(1)), "x")
        );
        assert_eq!(
            read(&frame, "x"),
            read(&Frame::from_value("x", int(2)), "x")
        );
        assert!(captured.get("y").is_none());
        assert!(frame.contains_key("y"));
    }

    #[test]
    fn locals_shadow_captured_bindings_in_lookup_and_listing() {
        let mut frame = Frame::new();
        frame.insert("a".to_string(), int(1));
        frame.insert("b".to_string(), int(2));
        let mut inner = frame.capture();
        inner.insert("a".to_string(), int(10));
        let listed: Vec<&String> = inner
            .to_sorted()
            .into_iter()
            .map(|(name, _)| name)
            .collect();
        assert_eq!(listed, ["a", "b"]);
        assert_eq!(inner.len(), 2);
        assert_eq!(
            read(&inner, "a"),
            read(&Frame::from_value("a", int(10)), "a")
        );
        assert_eq!(
            read(&inner, "b"),
            read(&Frame::from_value("b", int(2)), "b")
        );
        assert!(!inner.is_empty());
        assert!(Frame::new().is_empty());
        assert!(Frame::new().capture().is_empty());
    }

    #[test]
    fn capturing_and_cloning_copy_no_frozen_value() {
        let mut frame = Frame::new();
        for index in 0..8 {
            frame.insert(format!("v{index}"), int(index));
        }
        reset_frame_value_copies();
        let captured = frame.capture();
        let again = captured.clone();
        let saved = frame.clone();
        assert_eq!(frame_value_copies(), 0);
        assert_eq!(again.len(), 8);
        assert_eq!(saved.len(), 8);
        frame.insert("w".to_string(), int(9));
        let _copied = frame.clone();
        assert_eq!(frame_value_copies(), 1, "only the new local is copied");
    }

    impl Frame {
        fn from_value(name: &str, value: RuntimeValue) -> Frame {
            let mut frame = Frame::new();
            frame.insert(name.to_string(), value);
            frame
        }
    }
}
