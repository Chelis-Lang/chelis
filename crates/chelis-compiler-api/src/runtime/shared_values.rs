//! The element sequence of a container [`RuntimeValue`] (chelis#2567).
//!
//! A runtime value is immutable, so a list, tuple, dict or data-type value
//! shares its elements instead of owning a private copy. Reading a binding
//! clones its value, and with owned elements that clone copied the whole
//! nested value, one native frame group per level: a data-type chain built by
//! `fold` overflowed the stack a few thousand links deep, and every step of
//! the fold copied the chain built so far. With shared elements a clone is one
//! reference-count increment, and a write (`DerefMut`) copies only the one
//! sequence it changes, when another owner still shares it.
//!
//! Release is the other walk the derived code made recursive: dropping the
//! last owner of a chain dropped each level from inside the one above. The
//! [`Drop`] here drains every element it solely owns onto a worklist and
//! releases the value there, so a release uses the same native depth whatever
//! the value's nesting. An element another owner still shares is only
//! decremented; its own last release drains it the same way.

use std::fmt;
use std::ops::{Deref, DerefMut};
use std::sync::Arc;

use super::RuntimeValue;

/// An element type a [`Shared`] sequence can hold: something that hands the
/// runtime values it owns to a release worklist.
pub trait Nested: Clone {
    fn release_into(self, pending: &mut Vec<RuntimeValue>);
}

impl Nested for RuntimeValue {
    fn release_into(self, pending: &mut Vec<RuntimeValue>) {
        pending.push(self);
    }
}

impl Nested for (RuntimeValue, RuntimeValue) {
    fn release_into(self, pending: &mut Vec<RuntimeValue>) {
        pending.push(self.0);
        pending.push(self.1);
    }
}

thread_local! {
    /// Container elements copied out of a shared sequence on this thread
    /// since the last reset. Counted at the copy, never estimated.
    static ELEMENT_COPIES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Record that `elements` container elements were copied.
pub(super) fn record_element_copies(elements: usize) {
    ELEMENT_COPIES.with(|copies| copies.set(copies.get() + elements as u64));
}

/// Container elements copied on this thread since the last reset.
#[cfg(test)]
pub(crate) fn element_copies() -> u64 {
    ELEMENT_COPIES.with(std::cell::Cell::get)
}

/// Reset [`element_copies`] for this thread.
#[cfg(test)]
pub(crate) fn reset_element_copies() {
    ELEMENT_COPIES.with(|copies| copies.set(0));
}

/// A shared, copy-on-write sequence of container elements.
pub struct Shared<T: Nested>(Arc<Vec<T>>);

/// The element sequence of a list, tuple or data-type value.
pub type Values = Shared<RuntimeValue>;

/// The entry sequence of a dict value.
pub type Entries = Shared<(RuntimeValue, RuntimeValue)>;

impl<T: Nested> Shared<T> {
    /// The elements as an owned vector: moved out when this is their only
    /// owner, copied (each element a shallow clone) otherwise.
    pub fn into_vec(mut self) -> Vec<T> {
        let items = std::mem::replace(&mut self.0, Arc::new(Vec::new()));
        Arc::try_unwrap(items).unwrap_or_else(|shared| {
            record_element_copies(shared.len());
            shared.as_ref().clone()
        })
    }

    /// Move out the elements this is the only owner of, leaving it empty.
    fn take_unique(&mut self) -> Option<Vec<T>> {
        Arc::get_mut(&mut self.0).map(std::mem::take)
    }
}

impl RuntimeValue {
    /// Hand every element this value solely owns to `pending`, leaving the
    /// value with no nested value to release in place.
    fn release_children_into(&mut self, pending: &mut Vec<RuntimeValue>) {
        match self {
            RuntimeValue::List(items)
            | RuntimeValue::Tuple(items)
            | RuntimeValue::Adt { fields: items, .. } => {
                for item in items.take_unique().into_iter().flatten() {
                    item.release_into(pending);
                }
            }
            RuntimeValue::Dict(entries) => {
                for entry in entries.take_unique().into_iter().flatten() {
                    entry.release_into(pending);
                }
            }
            _ => {}
        }
    }
}

impl<T: Nested> Drop for Shared<T> {
    fn drop(&mut self) {
        let Some(items) = self.take_unique() else {
            return;
        };
        let mut pending = Vec::new();
        for item in items {
            item.release_into(&mut pending);
        }
        while let Some(mut value) = pending.pop() {
            value.release_children_into(&mut pending);
        }
    }
}

impl<T: Nested> Clone for Shared<T> {
    fn clone(&self) -> Self {
        Self(Arc::clone(&self.0))
    }
}

impl<T: Nested> Default for Shared<T> {
    fn default() -> Self {
        Self(Arc::new(Vec::new()))
    }
}

impl<T: Nested> Deref for Shared<T> {
    type Target = Vec<T>;

    fn deref(&self) -> &Vec<T> {
        &self.0
    }
}

impl<T: Nested> DerefMut for Shared<T> {
    fn deref_mut(&mut self) -> &mut Vec<T> {
        if Arc::get_mut(&mut self.0).is_none() {
            record_element_copies(self.0.len());
        }
        Arc::make_mut(&mut self.0)
    }
}

impl<T: Nested> From<Vec<T>> for Shared<T> {
    fn from(items: Vec<T>) -> Self {
        Self(Arc::new(items))
    }
}

impl<T: Nested> FromIterator<T> for Shared<T> {
    fn from_iter<I: IntoIterator<Item = T>>(items: I) -> Self {
        Self(Arc::new(items.into_iter().collect()))
    }
}

impl<T: Nested> IntoIterator for Shared<T> {
    type Item = T;
    type IntoIter = std::vec::IntoIter<T>;

    fn into_iter(self) -> Self::IntoIter {
        self.into_vec().into_iter()
    }
}

impl<'a, T: Nested> IntoIterator for &'a Shared<T> {
    type Item = &'a T;
    type IntoIter = std::slice::Iter<'a, T>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl<T: Nested + fmt::Debug> fmt::Debug for Shared<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.as_ref().fmt(f)
    }
}

#[cfg(test)]
mod tests {
    use super::super::eval::runtime_values_equal;
    use super::super::host_ops::render_value;
    use super::*;

    /// Far deeper than the small stack each test below runs on.
    const DEPTH: i64 = 200_000;
    const SMALL_STACK: usize = 256 * 1024;

    fn on_small_stack(test: impl FnOnce() + Send + 'static) {
        std::thread::Builder::new()
            .stack_size(SMALL_STACK)
            .spawn(test)
            .expect("spawn a small-stack thread")
            .join()
            .expect("the small-stack walk completes");
    }

    fn link(head: RuntimeValue, tail: RuntimeValue) -> RuntimeValue {
        RuntimeValue::Adt {
            ctor: "Link".to_string(),
            source_name: "Link".to_string(),
            fields: vec![head, tail].into(),
            field_names: None,
        }
    }

    fn end() -> RuntimeValue {
        RuntimeValue::Adt {
            ctor: "End".to_string(),
            source_name: "End".to_string(),
            fields: Values::default(),
            field_names: None,
        }
    }

    /// `Link(depth - 1, ... Link(0, <bottom>))`, each level through `wrap`.
    fn chain(
        depth: i64,
        bottom: RuntimeValue,
        wrap: fn(RuntimeValue) -> RuntimeValue,
    ) -> RuntimeValue {
        (0..depth).fold(bottom, |tail, index| {
            wrap(link(RuntimeValue::int64(index), tail))
        })
    }

    fn direct(value: RuntimeValue) -> RuntimeValue {
        value
    }

    fn in_list(value: RuntimeValue) -> RuntimeValue {
        RuntimeValue::List(vec![value].into())
    }

    fn in_tuple(value: RuntimeValue) -> RuntimeValue {
        RuntimeValue::Tuple(vec![value].into())
    }

    fn in_dict(value: RuntimeValue) -> RuntimeValue {
        RuntimeValue::Dict(vec![(RuntimeValue::String("k".to_string()), value)].into())
    }

    #[test]
    fn a_deep_value_of_every_container_kind_releases_on_a_small_stack() {
        for wrap in [direct, in_list, in_tuple, in_dict] {
            on_small_stack(move || drop(chain(DEPTH, end(), wrap)));
        }
    }

    #[test]
    fn a_shared_deep_value_releases_when_its_last_owner_goes() {
        on_small_stack(|| {
            let value = chain(DEPTH, end(), direct);
            let alias = value.clone();
            drop(value);
            // Releasing one owner leaves the other's whole value intact.
            assert!(
                render_value(&alias)
                    .ends_with(&format!("Link(0, End{}", ")".repeat(DEPTH as usize)))
            );
            drop(alias);
        });
    }

    #[test]
    fn a_clone_shares_and_a_write_copies_only_the_written_sequence() {
        let items: Values = vec![RuntimeValue::int64(1), RuntimeValue::int64(2)].into();
        let mut written = items.clone();
        assert!(
            Arc::ptr_eq(&items.0, &written.0),
            "a clone shares the elements"
        );
        written.push(RuntimeValue::int64(3));
        assert!(!Arc::ptr_eq(&items.0, &written.0), "a write detaches");
        assert_eq!(items.len(), 2);
        assert_eq!(written.len(), 3);
        assert_eq!(items.clone().into_vec().len(), 2);
    }

    #[test]
    fn a_deep_value_renders_on_a_small_stack() {
        on_small_stack(|| {
            let text = render_value(&chain(DEPTH, end(), direct));
            assert!(text.starts_with(&format!("Link({}, Link(", DEPTH - 1)));
            assert!(text.ends_with(&format!("Link(0, End{}", ")".repeat(DEPTH as usize))));
        });
        assert_eq!(
            render_value(&in_dict(in_tuple(in_list(chain(2, end(), direct))))),
            "dict(k: ([Link(1, Link(0, End))]))"
        );
    }

    #[test]
    fn deep_values_compare_on_a_small_stack() {
        for wrap in [direct, in_list, in_tuple, in_dict] {
            on_small_stack(move || {
                let lhs = chain(DEPTH, end(), wrap);
                assert_eq!(
                    runtime_values_equal(&lhs, &chain(DEPTH, end(), wrap)),
                    Ok(true)
                );
                let differs_at_the_bottom = chain(DEPTH, RuntimeValue::Unit, wrap);
                assert_eq!(
                    runtime_values_equal(&lhs, &differs_at_the_bottom),
                    Ok(false)
                );
                let refused_at_the_bottom =
                    chain(DEPTH, RuntimeValue::MappedFile(Vec::new()), wrap);
                assert!(runtime_values_equal(&lhs, &refused_at_the_bottom).is_err());
            });
        }
    }

    /// The first difference in visiting order decides, as a recursive
    /// comparison's would: a refused value after a difference is never
    /// reached, and one before it is.
    #[test]
    fn comparison_order_is_left_to_right_depth_first() {
        let refused = || RuntimeValue::MappedFile(Vec::new());
        let lhs = RuntimeValue::Tuple(vec![RuntimeValue::int64(1), refused()].into());
        let rhs = RuntimeValue::Tuple(vec![RuntimeValue::int64(2), refused()].into());
        assert_eq!(runtime_values_equal(&lhs, &rhs), Ok(false));
        let lhs = RuntimeValue::Tuple(vec![in_list(refused()), RuntimeValue::int64(1)].into());
        let rhs = RuntimeValue::Tuple(vec![in_list(refused()), RuntimeValue::int64(2)].into());
        assert!(runtime_values_equal(&lhs, &rhs).is_err());
    }

    #[test]
    fn dictionary_search_preserves_entry_matching_and_refusal_order() {
        let entry = |n| {
            (
                RuntimeValue::String("k".to_string()),
                RuntimeValue::int64(n),
            )
        };
        let lhs = RuntimeValue::Dict(vec![entry(1), entry(2)].into());
        let reordered = RuntimeValue::Dict(vec![entry(2), entry(1)].into());
        let missing = RuntimeValue::Dict(vec![entry(2), entry(3)].into());
        assert_eq!(runtime_values_equal(&lhs, &reordered), Ok(true));
        assert_eq!(runtime_values_equal(&lhs, &missing), Ok(false));

        let refused = RuntimeValue::Dict(
            vec![(RuntimeValue::MappedFile(Vec::new()), RuntimeValue::int64(1))].into(),
        );
        assert!(runtime_values_equal(&refused, &refused).is_err());
    }
}
