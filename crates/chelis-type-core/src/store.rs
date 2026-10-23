use chelis_unord::{PersistentMap, UnordMap};
use core::fmt;
use core::ops::{Deref, DerefMut};
use spin::Mutex;

/// Logical aliases and transient terminal-owner lookups have separate storage.
#[derive(Debug)]
pub(crate) struct Bindings<K, V> {
    pub(crate) bindings: PersistentMap<K, V>,
    pub(crate) resolved: UnordMap<K, K>,
}

impl<K, V> Default for Bindings<K, V> {
    fn default() -> Self {
        Self {
            bindings: PersistentMap::new(),
            resolved: UnordMap::new(),
        }
    }
}

impl<K, V> Clone for Bindings<K, V> {
    fn clone(&self) -> Self {
        Self {
            bindings: self.bindings.clone(),
            resolved: UnordMap::new(),
        }
    }
}

impl<K, V> Deref for Bindings<K, V> {
    type Target = PersistentMap<K, V>;

    fn deref(&self) -> &Self::Target {
        &self.bindings
    }
}

impl<K, V> DerefMut for Bindings<K, V> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.resolved.clear();
        &mut self.bindings
    }
}

#[cfg(feature = "serde")]
impl<K: Ord + serde::Serialize, V: serde::Serialize> serde::Serialize for Bindings<K, V> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        self.bindings.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, K, V> serde::Deserialize<'de> for Bindings<K, V>
where
    K: Ord + Clone + serde::Deserialize<'de>,
    V: Clone + serde::Deserialize<'de>,
{
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        Ok(Self {
            bindings: PersistentMap::deserialize(deserializer)?,
            resolved: UnordMap::new(),
        })
    }
}

/// A private guard never escapes a protected operation.
pub(crate) struct Store<T> {
    state: Mutex<State<T>>,
}

struct State<T> {
    incomplete: bool,
    value: T,
}

impl<T> Store<T> {
    pub(crate) fn new(value: T) -> Self {
        Self {
            state: Mutex::new(State {
                incomplete: false,
                value,
            }),
        }
    }

    /// An interrupted operation leaves the marker set and rejects later access.
    pub(crate) fn with<R>(&self, operation: impl FnOnce(&mut T) -> R) -> R {
        let mut state = self.state.lock();
        assert!(!state.incomplete, "substitution store operation incomplete");
        state.incomplete = true;
        let result = operation(&mut state.value);
        state.incomplete = false;
        result
    }

    /// Release the guard before an operation can access another store.
    pub(crate) fn with_detached<R>(&self, operation: impl FnOnce(&mut T) -> R) -> R
    where
        T: Default,
    {
        let mut value = {
            let mut state = self.state.lock();
            assert!(!state.incomplete, "substitution store operation incomplete");
            state.incomplete = true;
            core::mem::take(&mut state.value)
        };
        let result = operation(&mut value);
        let mut state = self.state.lock();
        state.value = value;
        state.incomplete = false;
        result
    }
}

impl<T: Default> Default for Store<T> {
    fn default() -> Self {
        Self::new(T::default())
    }
}

impl<T: Clone> Clone for Store<T> {
    fn clone(&self) -> Self {
        Self::new(self.with(|value| value.clone()))
    }
}

impl<T> fmt::Debug for Store<T> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Store").finish_non_exhaustive()
    }
}

#[cfg(feature = "serde")]
impl<T: Clone + serde::Serialize> serde::Serialize for Store<T> {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        let snapshot = self.with(|value| value.clone());
        snapshot.serialize(serializer)
    }
}

#[cfg(feature = "serde")]
impl<'de, T: serde::Deserialize<'de>> serde::Deserialize<'de> for Store<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        T::deserialize(deserializer).map(Self::new)
    }
}

#[cfg(test)]
mod tests {
    use super::{Bindings, Store};
    use alloc::vec;
    use std::panic::{AssertUnwindSafe, catch_unwind};

    #[test]
    fn logical_mutation_clears_terminal_cache() {
        let store = Store::new(Bindings::<u32, u32>::default());
        store.with(|map| {
            map.insert(1, 2);
            map.resolved.insert(1, 3);
            assert_eq!(map.resolved.get(&1), Some(&3));
            assert_eq!(map.get(&1), Some(&2));
            map.insert(2, 4);
            assert!(map.resolved.is_empty());
        });
    }

    #[test]
    fn clone_retains_independent_logical_state_without_cache() {
        let original = Store::new(Bindings::<u32, u32>::default());
        original.with(|map| {
            map.insert(1, 2);
            map.resolved.insert(1, 3);
        });
        let cloned = original.clone();
        cloned.with(|map| {
            assert!(map.resolved.is_empty());
            assert_eq!(map.get(&1), Some(&2));
            map.insert(1, 4);
        });
        original.with(|map| {
            assert_eq!(map.get(&1), Some(&2));
            assert_eq!(map.resolved.get(&1), Some(&3));
        });
    }

    #[test]
    fn interrupted_operation_rejects_later_access_and_clone() {
        let store = Store::new(vec![1]);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                store.with(|values| {
                    values.push(2);
                    panic!("interrupt protected operation");
                });
            }))
            .is_err()
        );
        assert!(catch_unwind(AssertUnwindSafe(|| store.with(|values| values.len()))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| store.clone())).is_err());
    }

    #[test]
    fn interrupted_detached_operation_rejects_later_access() {
        let store = Store::new(vec![1]);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                store.with_detached(|values| {
                    values.push(2);
                    panic!("interrupt detached operation");
                });
            }))
            .is_err()
        );
        assert!(catch_unwind(AssertUnwindSafe(|| store.with(|values| values.len()))).is_err());
        assert!(catch_unwind(AssertUnwindSafe(|| store.with_detached(|_| ()))).is_err());
    }

    #[test]
    fn detached_operation_releases_guard_before_callback() {
        let store = Store::new(vec![1]);
        store.with_detached(|_| {
            assert!(catch_unwind(AssertUnwindSafe(|| store.with(|_| ()))).is_err());
        });
        assert_eq!(store.with(|values| values.clone()), vec![1]);
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_retains_only_logical_bindings() {
        let store = Store::new(Bindings::<u32, u32>::default());
        store.with(|map| {
            map.insert(1, 2);
            map.resolved.insert(1, 3);
        });
        let bytes = bincode::serialize(&store).unwrap();
        let logical = store.with(|map| map.bindings.clone());
        assert_eq!(bytes, bincode::serialize(&logical).unwrap());
        let decoded: Store<Bindings<u32, u32>> = bincode::deserialize(&bytes).unwrap();
        decoded.with(|map| {
            assert_eq!(map.get(&1), Some(&2));
            assert!(map.resolved.is_empty());
        });
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serialization_callbacks_can_access_the_original_store() {
        use alloc::sync::{Arc, Weak};

        #[derive(Clone)]
        struct Callback(Weak<Store<Callback>>);

        impl serde::Serialize for Callback {
            fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
                self.0.upgrade().unwrap().with(|_| ());
                serializer.serialize_unit()
            }
        }

        let store = Arc::new_cyclic(|owner| Store::new(Callback(owner.clone())));
        bincode::serialize(store.as_ref()).unwrap();
        store.with(|_| ());
    }
}
