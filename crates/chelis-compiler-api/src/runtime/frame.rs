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
//!
//! A binding in the innermost scope can be released at its last read
//! (chelis#3720): the read takes the value instead of sharing it, so a
//! container written next is solely owned and changes in place. The released
//! name stays in its scope, so it still shadows every captured binding of the
//! same name, and a later read finds it released instead of resolving
//! outward.

use std::collections::BTreeMap;
use std::sync::Arc;

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
    /// A runtime interface stamps every nested tensor as a load without
    /// walking a potentially deep container at entry.
    Interface,
}

impl ResultProducer {
    pub(crate) fn tensor(operation: impl Into<String>) -> Self {
        Self::Tensor(operation.into())
    }

    pub(crate) fn operation(&self) -> Option<&str> {
        match self {
            Self::Tensor(operation) | Self::Uniform(operation) => Some(operation),
            Self::Interface => Some("load"),
            Self::Aggregate(_) => None,
        }
    }

    pub(crate) fn child(&self, index: usize) -> Option<Self> {
        self.child_ref(index).cloned()
    }

    /// [`Self::child`], borrowed: a walk that only reads a child's
    /// provenance copies no subtree.
    pub(crate) fn child_ref(&self, index: usize) -> Option<&Self> {
        match self {
            Self::Aggregate(children) => children.get(index).and_then(Option::as_ref),
            Self::Uniform(_) | Self::Interface => Some(self),
            Self::Tensor(_) => None,
        }
    }

    pub(crate) fn aggregate_prefix(&self, count: usize) -> Option<Self> {
        match self {
            Self::Aggregate(children) => {
                Self::aggregate(children.iter().take(count).cloned().collect())
            }
            Self::Uniform(_) | Self::Interface => Some(self.clone()),
            Self::Tensor(_) => None,
        }
    }

    pub(crate) fn aggregate_suffix(&self, start: usize) -> Option<Self> {
        match self {
            Self::Aggregate(children) => {
                Self::aggregate(children.iter().skip(start).cloned().collect())
            }
            Self::Uniform(_) | Self::Interface => Some(self.clone()),
            Self::Tensor(_) => None,
        }
    }

    pub(crate) fn aggregate(children: Vec<Option<Self>>) -> Option<Self> {
        children
            .iter()
            .any(Option::is_some)
            .then_some(Self::Aggregate(children))
    }

    /// Stamp a value crossing a runtime interface without visiting every
    /// nested element. An interface stamp follows projections and applies
    /// only where the projected value can contain a tensor.
    pub(crate) fn interface_load(value: &RuntimeValue) -> Option<Self> {
        Self::Interface
            .matches_value(value)
            .then_some(Self::Interface)
    }

    pub(crate) fn matches_value(&self, value: &RuntimeValue) -> bool {
        match (self, value) {
            (Self::Uniform(_), _) => true,
            (Self::Tensor(_), RuntimeValue::Tensor(_)) => true,
            (
                Self::Interface,
                RuntimeValue::Tensor(_)
                | RuntimeValue::Tuple(_)
                | RuntimeValue::List(_)
                | RuntimeValue::Adt { .. },
            ) => true,
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

/// What one scope binds a name to.
#[derive(Debug)]
enum Slot {
    /// Held by this scope alone. A binding stays inline until a snapshot
    /// first copies its scope, so a scope that is never snapshotted makes
    /// no allocation per binding.
    Owned(Binding),
    /// Shared behind a reference count with the snapshots of this scope, so
    /// each later snapshot copies a pointer, not the binding.
    Shared(Arc<Binding>),
    /// The value was taken at the binding's last read.
    Released,
}

impl Clone for Slot {
    /// Copies an owned binding and shares a shared one.
    fn clone(&self) -> Self {
        match self {
            Self::Owned(binding) => {
                record_binding_copy();
                Self::Owned(binding.clone())
            }
            Self::Shared(binding) => Self::Shared(Arc::clone(binding)),
            Self::Released => Self::Released,
        }
    }
}

impl Slot {
    fn binding(&self) -> Option<&Binding> {
        match self {
            Self::Owned(binding) => Some(binding),
            Self::Shared(binding) => Some(binding),
            Self::Released => None,
        }
    }

    /// Move an owned binding behind a reference count so copies share it.
    fn share(&mut self) {
        if let Self::Owned(_) = self {
            let Self::Owned(binding) = std::mem::replace(self, Self::Released) else {
                unreachable!("the slot was just matched as owned");
            };
            record_binding_copy();
            *self = Self::Shared(Arc::new(binding));
        }
    }
}

/// A binding taken from a scope at its last read. Every snapshot that shared
/// it releases its own copy before [`Self::into_parts`] runs, so the value
/// then moves out solely owned.
pub(crate) struct ReleasedBinding(Slot);

impl ReleasedBinding {
    /// The value and its producer: moved out when no scope still shares the
    /// binding, cloned otherwise.
    pub(crate) fn into_parts(self) -> (RuntimeValue, Option<ResultProducer>) {
        let Binding {
            value,
            result_producer,
        } = match self.0 {
            Slot::Owned(binding) => binding,
            Slot::Shared(binding) => Arc::unwrap_or_clone(binding),
            Slot::Released => unreachable!("only a live slot is released"),
        };
        (value, result_producer)
    }
}

/// One scope's bindings, sorted by name.
///
/// A sorted vector rather than an ordered map: a map node reserves room for
/// eleven entries whatever the scope holds, and a binding is large, so a
/// scope of two bindings held about 2.5 KB while a recursion kept its frame
/// alive (chelis#3842). The vector's first allocation holds four entries.
#[derive(Clone, Default)]
struct Scope {
    entries: Vec<(String, Slot)>,
}

impl std::fmt::Debug for Scope {
    /// The length only, as the ordered map it replaced reported.
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "Scope {{ len: {} }}", self.entries.len())
    }
}

impl Scope {
    fn position(&self, name: &str) -> Result<usize, usize> {
        self.entries
            .binary_search_by(|(bound, _)| bound.as_str().cmp(name))
    }

    fn get(&self, name: &str) -> Option<&Slot> {
        let index = self.position(name).ok()?;
        Some(&self.entries[index].1)
    }

    fn get_mut(&mut self, name: &str) -> Option<&mut Slot> {
        let index = self.position(name).ok()?;
        Some(&mut self.entries[index].1)
    }

    fn insert(&mut self, name: String, slot: Slot) {
        match self.position(&name) {
            Ok(index) => self.entries[index].1 = slot,
            Err(index) => self.entries.insert(index, (name, slot)),
        }
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

/// What a name resolves to: the innermost binding of it, and whether that
/// binding was released at its last read.
pub(crate) enum Lookup<'a> {
    Live(&'a RuntimeValue),
    Released,
    Unbound,
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

thread_local! {
    /// Bindings copied, or moved behind a reference count, on this thread
    /// since the last reset. Counted at the copy, never estimated.
    static BINDING_COPIES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

fn record_binding_copy() {
    BINDING_COPIES.with(|copies| copies.set(copies.get() + 1));
}

/// Bindings copied, or moved behind a reference count, on this thread since
/// the last reset.
#[cfg(test)]
pub(crate) fn binding_copies() -> u64 {
    BINDING_COPIES.with(std::cell::Cell::get)
}

/// Reset [`binding_copies`] for this thread.
#[cfg(test)]
pub(crate) fn reset_binding_copies() {
    BINDING_COPIES.with(|copies| copies.set(0));
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
    locals: Scope,
    /// Every scope captured below this one, innermost first. Shared with the
    /// closures that captured it and never mutated after freezing.
    captured: Option<Arc<Frame>>,
}

impl Clone for Frame {
    /// Copies the innermost scope's entries and shares the captured chain. An
    /// owned binding is copied; a shared one is shared.
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

    /// A copy of this frame to restore later, sharing every binding with it.
    /// Each binding moves behind a reference count the first time a
    /// snapshot copies it, so it is never copied twice.
    pub(crate) fn snapshot(&mut self) -> Frame {
        for (_, slot) in &mut self.locals.entries {
            slot.share();
        }
        self.clone()
    }

    /// What the innermost scope that binds `name` binds it to.
    fn slot(&self, name: &str) -> Option<&Slot> {
        let mut frame = self;
        loop {
            if let Some(slot) = frame.locals.get(name) {
                return Some(slot);
            }
            frame = frame.captured.as_deref()?;
        }
    }

    /// What `name` resolves to. A released binding still shadows every
    /// outer binding of the same name.
    pub(crate) fn lookup(&self, name: &str) -> Lookup<'_> {
        match self.slot(name).map(Slot::binding) {
            Some(Some(binding)) => Lookup::Live(&binding.value),
            Some(None) => Lookup::Released,
            None => Lookup::Unbound,
        }
    }

    /// The innermost binding of `name`, if any scope binds it and it was not
    /// released.
    pub(crate) fn get(&self, name: &str) -> Option<&RuntimeValue> {
        match self.lookup(name) {
            Lookup::Live(value) => Some(value),
            Lookup::Released | Lookup::Unbound => None,
        }
    }

    /// Producer provenance retained beside a lexical value. It is private
    /// execution metadata, not part of the value or its public representation.
    pub(crate) fn result_producer(&self, name: &str) -> Option<&ResultProducer> {
        self.slot(name)
            .and_then(Slot::binding)
            .and_then(|binding| binding.result_producer.as_ref())
    }

    /// Whether any scope binds `name`, released or not: a released local is
    /// still the binding that name denotes in its scope.
    pub(crate) fn contains_key(&self, name: &str) -> bool {
        self.slot(name).is_some()
    }

    /// Take the binding this frame's innermost scope binds to `name`,
    /// leaving the name bound and released. A binding in a captured scope is
    /// shared with the closures that captured it and is never released, so
    /// this returns `None` for it, as it does for a name the innermost scope
    /// does not bind or already released.
    pub(crate) fn release_local(&mut self, name: &str) -> Option<ReleasedBinding> {
        let slot = self.locals.get_mut(name)?;
        match std::mem::replace(slot, Slot::Released) {
            Slot::Released => None,
            live => Some(ReleasedBinding(live)),
        }
    }

    /// Bind `name` in the innermost scope, shadowing any captured binding.
    pub(crate) fn insert(&mut self, name: String, value: RuntimeValue) {
        self.insert_with_result_producer(name, value, None);
    }

    /// Bind a value and the operation that produced this exact tensor result.
    /// Shadowing replaces both together, so an alias captured before a later
    /// same-named binding retains its original producer.
    pub(crate) fn insert_with_result_producer(
        &mut self,
        name: String,
        value: RuntimeValue,
        result_producer: Option<ResultProducer>,
    ) {
        let binding = Binding {
            value,
            result_producer,
        };
        self.locals.insert(name, Slot::Owned(binding));
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
        let mut merged: BTreeMap<&String, Option<&RuntimeValue>> = BTreeMap::new();
        // Outermost first, so an inner scope's binding overwrites an outer one.
        // A released binding hides the outer ones and is not listed itself.
        for frame in chain.into_iter().rev() {
            for (name, slot) in &frame.locals.entries {
                merged.insert(name, slot.binding().map(|binding| &binding.value));
            }
        }
        merged
            .into_iter()
            .filter_map(|(name, value)| Some((name, value?)))
            .collect()
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
            locals: Scope::default(),
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
    fn a_released_local_shadows_its_captured_name_and_reads_as_released() {
        let mut frame = Frame::new();
        frame.insert("xs".to_string(), int(1));
        let mut inner = frame.capture();
        inner.insert("xs".to_string(), int(2));
        let (value, _) = inner
            .release_local("xs")
            .expect("a live local releases")
            .into_parts();
        assert_eq!(
            super::super::host_ops::render_value(&value),
            super::super::host_ops::render_value(&int(2))
        );
        assert!(matches!(inner.lookup("xs"), Lookup::Released));
        assert!(
            inner.get("xs").is_none(),
            "the captured `xs` stays shadowed"
        );
        assert!(inner.contains_key("xs"));
        assert!(inner.to_sorted().is_empty());
        assert!(
            inner.release_local("xs").is_none(),
            "a release happens once"
        );
        // A later binding of the name is live again.
        inner.insert("xs".to_string(), int(3));
        assert_eq!(
            read(&inner, "xs"),
            read(&Frame::from_value("xs", int(3)), "xs")
        );
    }

    #[test]
    fn a_captured_binding_is_never_released() {
        let mut frame = Frame::new();
        frame.insert("xs".to_string(), int(1));
        let closure_env = frame.capture();
        assert!(
            frame.release_local("xs").is_none(),
            "a captured binding is shared with the closure that captured it"
        );
        assert_eq!(read(&closure_env, "xs"), read(&frame, "xs"));
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
        let value = RuntimeValue::Tuple(
            vec![
                int(7),
                tensor.clone(),
                RuntimeValue::Adt {
                    ctor: "Some".to_string(),
                    source_name: "Some".to_string(),
                    fields: vec![tensor].into(),
                    field_names: None,
                },
            ]
            .into(),
        );
        let producer = ResultProducer::interface_load(&value).expect("tensor leaves exist");
        // A projection keeps a producer only where it matches the value.
        assert_eq!(
            producer
                .child(0)
                .filter(|child| child.matches_value(&int(7))),
            None
        );
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
