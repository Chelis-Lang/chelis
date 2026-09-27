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

/// Private producer provenance paired with a runtime value. Aggregates retain
/// one entry per value slot so a later projection cannot inherit a sibling's
/// producer or collapse distinct transform roots to a guessed common name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ResultProducer {
    Tensor(String),
    Aggregate(Vec<Option<ResultProducer>>),
    /// A list combinator's result: every tensor nested in it, at any depth,
    /// has that combinator as its producer (spec/04 section 4.7), so a
    /// projection of any child is the same stamp.
    Uniform(String),
}

/// One pending step of [`ResultProducer::interface_load`]'s walk.
enum InterfaceStep<'a> {
    Visit(&'a RuntimeValue),
    /// The last `usize` stamps are one aggregate's children, in order.
    Assemble(usize),
}

impl ResultProducer {
    pub(crate) fn tensor(operation: impl Into<String>) -> Self {
        Self::Tensor(operation.into())
    }

    pub(crate) fn operation(&self) -> Option<&str> {
        match self {
            Self::Tensor(operation) | Self::Uniform(operation) => Some(operation),
            Self::Aggregate(_) => None,
        }
    }

    pub(crate) fn child(&self, index: usize) -> Option<Self> {
        match self {
            Self::Aggregate(children) => children.get(index).cloned().flatten(),
            Self::Uniform(_) => Some(self.clone()),
            Self::Tensor(_) => None,
        }
    }

    pub(crate) fn aggregate_prefix(&self, count: usize) -> Option<Self> {
        match self {
            Self::Aggregate(children) => {
                Self::aggregate(children.iter().take(count).cloned().collect())
            }
            Self::Uniform(_) => Some(self.clone()),
            Self::Tensor(_) => None,
        }
    }

    pub(crate) fn aggregate_suffix(&self, start: usize) -> Option<Self> {
        match self {
            Self::Aggregate(children) => {
                Self::aggregate(children.iter().skip(start).cloned().collect())
            }
            Self::Uniform(_) => Some(self.clone()),
            Self::Tensor(_) => None,
        }
    }

    pub(crate) fn aggregate(children: Vec<Option<Self>>) -> Option<Self> {
        children
            .iter()
            .any(Option::is_some)
            .then_some(Self::Aggregate(children))
    }

    /// Stamp a value crossing a genuine runtime interface. Every tensor leaf
    /// is observed through `load`; aggregate shape is retained so a later
    /// projection cannot lose the interface origin or borrow a sibling's.
    /// The value is walked from a worklist, so a value nested far deeper than
    /// the native stack is stamped with bounded native depth (chelis#2567).
    pub(crate) fn interface_load(value: &RuntimeValue) -> Option<Self> {
        if let Some(stamp) = Self::shallow_interface_load(value) {
            return stamp;
        }
        let mut steps = vec![InterfaceStep::Visit(value)];
        let mut stamped: Vec<Option<Self>> = Vec::new();
        while let Some(step) = steps.pop() {
            match step {
                InterfaceStep::Visit(value) => match value {
                    value if let Some(stamp) = Self::shallow_interface_load(value) => {
                        stamped.push(stamp)
                    }
                    RuntimeValue::Tensor(_) => stamped.push(Some(Self::tensor("load"))),
                    RuntimeValue::Tuple(values)
                    | RuntimeValue::List(values)
                    | RuntimeValue::Adt { fields: values, .. } => {
                        steps.push(InterfaceStep::Assemble(values.len()));
                        steps.extend(values.iter().rev().map(InterfaceStep::Visit));
                    }
                    _ => stamped.push(None),
                },
                InterfaceStep::Assemble(count) => {
                    let children = stamped.split_off(stamped.len() - count);
                    stamped.push(Self::aggregate(children));
                }
            }
        }
        stamped.pop().flatten()
    }

    /// The stamp of a leaf, or of an aggregate whose children are all leaves,
    /// made directly; `None` when a child is itself an aggregate. An aggregate
    /// with no tensor child is stamped without allocating.
    fn shallow_interface_load(value: &RuntimeValue) -> Option<Option<Self>> {
        let values = match value {
            RuntimeValue::Tensor(_) => return Some(Some(Self::tensor("load"))),
            RuntimeValue::Tuple(values)
            | RuntimeValue::List(values)
            | RuntimeValue::Adt { fields: values, .. } => values,
            _ => return Some(None),
        };
        let mut has_tensor = false;
        for value in values {
            match value {
                RuntimeValue::Tensor(_) => has_tensor = true,
                RuntimeValue::Tuple(_) | RuntimeValue::List(_) | RuntimeValue::Adt { .. } => {
                    return None;
                }
                _ => {}
            }
        }
        if !has_tensor {
            return Some(None);
        }
        let children = values
            .iter()
            .map(|value| matches!(value, RuntimeValue::Tensor(_)).then(|| Self::tensor("load")))
            .collect();
        Some(Some(Self::Aggregate(children)))
    }

    pub(crate) fn matches_value(&self, value: &RuntimeValue) -> bool {
        match (self, value) {
            (Self::Uniform(_), _) => true,
            (Self::Tensor(_), RuntimeValue::Tensor(_)) => true,
            (Self::Aggregate(children), RuntimeValue::Tuple(values))
            | (Self::Aggregate(children), RuntimeValue::List(values)) => {
                children.len() == values.len()
                    && children.iter().zip(values).all(|(producer, value)| {
                        producer.as_ref().is_none_or(|p| p.matches_value(value))
                    })
            }
            (Self::Aggregate(children), RuntimeValue::Adt { fields, .. }) => {
                children.len() == fields.len()
                    && children.iter().zip(fields).all(|(producer, value)| {
                        producer.as_ref().is_none_or(|p| p.matches_value(value))
                    })
            }
            _ => false,
        }
    }
}

#[derive(Debug, Clone)]
struct Binding {
    value: RuntimeValue,
    result_producer: Option<ResultProducer>,
}

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
    locals: UnordMap<String, Binding>,
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
            if let Some(binding) = frame.locals.get(name) {
                return Some(&binding.value);
            }
            frame = frame.captured.as_deref()?;
        }
    }

    /// Producer provenance retained beside a lexical value. It is private
    /// execution metadata, not part of the value or its public representation.
    pub(crate) fn result_producer(&self, name: &str) -> Option<&ResultProducer> {
        let mut frame = self;
        loop {
            if let Some(binding) = frame.locals.get(name) {
                return binding.result_producer.as_ref();
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
        self.insert_with_result_producer(name, value, None)
    }

    /// Bind a value and the operation that produced this exact tensor result.
    /// Shadowing replaces both together, so an alias captured before a later
    /// same-named binding retains its original producer.
    pub(crate) fn insert_with_result_producer(
        &mut self,
        name: String,
        value: RuntimeValue,
        result_producer: Option<ResultProducer>,
    ) -> Option<RuntimeValue> {
        self.locals
            .insert(
                name,
                Binding {
                    value,
                    result_producer,
                },
            )
            .map(|binding| binding.value)
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
            for (name, binding) in frame.locals.to_sorted() {
                merged.insert(name, &binding.value);
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
        let listed: Vec<(&String, String)> = inner
            .to_sorted()
            .into_iter()
            .map(|(name, value)| (name, super::super::host_ops::render_value(value)))
            .collect();
        let expected_a = super::super::host_ops::render_value(&int(10));
        let expected_b = super::super::host_ops::render_value(&int(2));
        assert_eq!(
            listed,
            [
                (&"a".to_string(), expected_a),
                (&"b".to_string(), expected_b)
            ],
            "the listing carries the innermost definition of a shadowed name"
        );
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

    #[test]
    fn captured_alias_producer_survives_later_same_name_shadowing() {
        let mut frame = Frame::new();
        frame.insert_with_result_producer(
            "value".to_string(),
            int(1),
            Some(ResultProducer::tensor("diagonal")),
        );
        let captured = frame.capture();
        frame.insert_with_result_producer(
            "value".to_string(),
            int(2),
            Some(ResultProducer::tensor("cumsum")),
        );
        assert_eq!(
            captured
                .result_producer("value")
                .and_then(ResultProducer::operation),
            Some("diagonal")
        );
        assert_eq!(
            frame
                .result_producer("value")
                .and_then(ResultProducer::operation),
            Some("cumsum")
        );
    }

    #[test]
    fn aggregate_producer_keeps_each_projected_field_distinct() {
        let producer = ResultProducer::aggregate(vec![
            Some(ResultProducer::tensor("diagonal")),
            Some(ResultProducer::tensor("cumsum")),
            None,
        ])
        .expect("two fields carry producer provenance");

        assert_eq!(
            producer.operation(),
            None,
            "an aggregate has no guessed common op"
        );
        assert_eq!(
            producer
                .child(0)
                .as_ref()
                .and_then(ResultProducer::operation),
            Some("diagonal")
        );
        assert_eq!(
            producer
                .child(1)
                .as_ref()
                .and_then(ResultProducer::operation),
            Some("cumsum")
        );
        assert_eq!(producer.child(2), None);
        assert_eq!(producer.child(3), None);
        assert_eq!(ResultProducer::aggregate(vec![None, None]), None);
    }

    #[test]
    fn aggregate_suffix_preserves_selected_children_and_null_slots() {
        let producer = ResultProducer::aggregate(vec![
            Some(ResultProducer::tensor("diagonal")),
            None,
            Some(ResultProducer::tensor("cumsum")),
        ])
        .expect("two fields carry producer provenance");
        let suffix = producer.aggregate_suffix(1).expect("cumsum remains");
        assert_eq!(suffix.child(0), None);
        assert_eq!(
            suffix.child(1).as_ref().and_then(ResultProducer::operation),
            Some("cumsum")
        );
        assert_eq!(producer.aggregate_suffix(3), None);
        assert_eq!(ResultProducer::tensor("load").aggregate_suffix(0), None);
    }

    #[test]
    fn aggregate_interface_load_stamps_only_tensor_leaves() {
        let tensor = RuntimeValue::Tensor(super::super::RuntimeTensorValue::new(
            chelis_ir::eval::TensorValue::from_vec(vec![1], vec![1.0]),
        ));
        let value = RuntimeValue::Tuple(vec![
            int(7),
            tensor.clone(),
            RuntimeValue::Adt {
                ctor: "Some".to_string(),
                fields: vec![tensor],
                field_names: None,
            },
        ]);
        let producer = ResultProducer::interface_load(&value).expect("tensor leaves exist");
        assert_eq!(producer.child(0), None);
        assert_eq!(
            producer
                .child(1)
                .as_ref()
                .and_then(ResultProducer::operation),
            Some("load")
        );
        assert_eq!(
            producer
                .child(2)
                .and_then(|adt| adt.child(0))
                .as_ref()
                .and_then(ResultProducer::operation),
            Some("load")
        );
    }

    impl Frame {
        fn from_value(name: &str, value: RuntimeValue) -> Frame {
            let mut frame = Frame::new();
            frame.insert(name.to_string(), value);
            frame
        }
    }
}
