use std::cell::RefCell;
use std::cmp::Ordering;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

use chelis_unord::{UnordMap, UnordSet};
use serde::{Serialize, Serializer};

#[derive(Default)]
struct Trace(RefCell<Vec<String>>);

impl Trace {
    fn record(&self, event: impl Into<String>) {
        self.0.borrow_mut().push(event.into());
    }

    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.0.borrow_mut())
    }
}

struct Key {
    value: u8,
    trace: Rc<Trace>,
}

impl Clone for Key {
    fn clone(&self) -> Self {
        self.trace.record(format!("key-clone: {}", self.value));
        Self {
            value: self.value,
            trace: Rc::clone(&self.trace),
        }
    }
}

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.trace
            .record(format!("key-eq: {} {}", self.value, other.value));
        self.value == other.value
    }
}

impl Eq for Key {}

impl PartialOrd for Key {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Key {
    fn cmp(&self, other: &Self) -> Ordering {
        self.trace
            .record(format!("key-cmp: {} {}", self.value, other.value));
        self.value.cmp(&other.value)
    }
}

impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl Serialize for Key {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.trace.record(format!("key-serialize: {}", self.value));
        self.value.serialize(serializer)
    }
}

struct Value {
    value: u8,
    trace: Rc<Trace>,
}

impl Clone for Value {
    fn clone(&self) -> Self {
        self.trace.record(format!("value-clone: {}", self.value));
        Self {
            value: self.value,
            trace: Rc::clone(&self.trace),
        }
    }
}

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.trace
            .record(format!("value-eq: {} {}", self.value, other.value));
        self.value == other.value
    }
}

impl Serialize for Value {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.trace
            .record(format!("value-serialize: {}", self.value));
        self.value.serialize(serializer)
    }
}

fn map(trace: &Rc<Trace>, insertion_order: &[u8]) -> UnordMap<Key, Value> {
    insertion_order
        .iter()
        .map(|value| {
            (
                Key {
                    value: *value,
                    trace: Rc::clone(trace),
                },
                Value {
                    value: *value,
                    trace: Rc::clone(trace),
                },
            )
        })
        .collect()
}

fn expected_pair_events(kind: &str) -> Vec<String> {
    (0..5)
        .flat_map(|value| {
            [
                format!("key-{kind}: {value}"),
                format!("value-{kind}: {value}"),
            ]
        })
        .collect()
}

#[test]
fn public_trait_callbacks_never_observe_raw_table_order() {
    let insertion_orders = [
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
        [2, 4, 0, 3, 1],
        [1, 3, 0, 4, 2],
    ];

    let mut clone_trace = None;
    for iteration in 0..64 {
        let insertion_order = insertion_orders[iteration % insertion_orders.len()];
        let trace = Rc::new(Trace::default());
        let value = map(&trace, &insertion_order);
        trace.take();

        let cloned = value.clone();
        let observed_clone = trace.take();
        if let Some(expected) = &clone_trace {
            assert_eq!(
                &observed_clone, expected,
                "Clone callbacks changed with insertion order or randomized hash state"
            );
        } else {
            clone_trace = Some(observed_clone);
        }

        assert!(value == cloned);
        let expected_eq = (0..5)
            .flat_map(|item| {
                [
                    format!("key-eq: {item} {item}"),
                    format!("value-eq: {item} {item}"),
                ]
            })
            .collect::<Vec<_>>();
        assert_eq!(trace.take(), expected_eq);

        let ordered = value.to_sorted();
        assert_eq!(
            ordered
                .iter()
                .map(|(key, item)| (key.value, item.value))
                .collect::<Vec<_>>(),
            [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        );
        assert!(
            trace.take().is_empty(),
            "the ordered exit must use the maintained canonical index without comparison callbacks"
        );

        assert_eq!(
            serde_json::to_vec(&value).unwrap(),
            br#"[[0,0],[1,1],[2,2],[3,3],[4,4]]"#
        );
        assert_eq!(trace.take(), expected_pair_events("serialize"));
    }
}

#[test]
fn set_trait_surfaces_share_the_structural_order_guarantee() {
    let insertion_orders = [
        [0, 1, 2, 3, 4],
        [4, 3, 2, 1, 0],
        [2, 4, 0, 3, 1],
        [1, 3, 0, 4, 2],
    ];
    let mut clone_trace = None;
    for iteration in 0..64 {
        let trace = Rc::new(Trace::default());
        let set = insertion_orders[iteration % insertion_orders.len()]
            .map(|value| Key {
                value,
                trace: Rc::clone(&trace),
            })
            .into_iter()
            .collect::<UnordSet<_>>();
        trace.take();
        let _cloned = set.clone();
        let observed = trace.take();
        if let Some(expected) = &clone_trace {
            assert_eq!(
                &observed, expected,
                "set Clone callbacks changed with insertion order or randomized hash state"
            );
        } else {
            clone_trace = Some(observed);
        }
    }

    let trace = Rc::new(Trace::default());
    let set = [4, 2, 0, 3, 1]
        .map(|value| Key {
            value,
            trace: Rc::clone(&trace),
        })
        .into_iter()
        .collect::<UnordSet<_>>();
    trace.take();
    let cloned = set.clone();
    trace.take();
    assert!(set == cloned);
    assert_eq!(
        trace.take(),
        (0..5)
            .map(|value| format!("key-eq: {value} {value}"))
            .collect::<Vec<_>>()
    );
    assert_eq!(serde_json::to_vec(&set).unwrap(), br#"[0,1,2,3,4]"#);
    assert_eq!(
        trace.take(),
        (0..5)
            .map(|value| format!("key-serialize: {value}"))
            .collect::<Vec<_>>()
    );
}

struct DropKey {
    value: u8,
    trace: Rc<Trace>,
}

impl PartialEq for DropKey {
    fn eq(&self, other: &Self) -> bool {
        self.value == other.value
    }
}

impl Eq for DropKey {}

impl PartialOrd for DropKey {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for DropKey {
    fn cmp(&self, other: &Self) -> Ordering {
        self.value.cmp(&other.value)
    }
}

impl Hash for DropKey {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.value.hash(state);
    }
}

impl Drop for DropKey {
    fn drop(&mut self) {
        self.trace.record(format!("key-drop: {}", self.value));
    }
}

struct DropValue {
    value: u8,
    trace: Rc<Trace>,
}

impl Drop for DropValue {
    fn drop(&mut self) {
        self.trace.record(format!("value-drop: {}", self.value));
    }
}

fn drop_map(trace: &Rc<Trace>) -> UnordMap<DropKey, DropValue> {
    [4, 2, 0, 3, 1]
        .map(|value| {
            (
                DropKey {
                    value,
                    trace: Rc::clone(trace),
                },
                DropValue {
                    value,
                    trace: Rc::clone(trace),
                },
            )
        })
        .into_iter()
        .collect()
}

fn expected_drop_events() -> Vec<String> {
    (0..5)
        .flat_map(|value| [format!("key-drop: {value}"), format!("value-drop: {value}")])
        .collect()
}

#[test]
fn clear_and_drop_retire_user_values_in_canonical_order() {
    let trace = Rc::new(Trace::default());
    let mut cleared = drop_map(&trace);
    cleared.clear();
    assert_eq!(trace.take(), expected_drop_events());

    drop(drop_map(&trace));
    assert_eq!(trace.take(), expected_drop_events());
}
